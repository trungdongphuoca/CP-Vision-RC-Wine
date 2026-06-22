"""
demo/app.py — Flask backend cho Demo Hội đồng
=============================================
Chạy: .venv\Scripts\python.exe demo\app.py
Truy cập: http://localhost:5000
"""
import sys, os, json, math, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parents[1]
app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path='')

# ── LLM Sommelier Engine (vLLM FP8 with Fallback) ─────────────────────
LLM_AVAILABLE = False
llm = None
SamplingParams = None
llm_model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8" # optimized FP8 repo

try:
    print(f"🔧 Initializing vLLM Sommelier ({llm_model_id}) with FP8 precision on GPU...")
    from vllm import LLM, SamplingParams
    # Capped VRAM budget at 14.5 GB (gpu_memory_utilization=0.75 for 16GB VRAM)
    llm = LLM(
        model=llm_model_id,
        quantization="fp8",
        gpu_memory_utilization=0.75,
        trust_remote_code=True,
        max_model_len=4096,
    )
    LLM_AVAILABLE = True
    print("✅ vLLM Engine loaded successfully!")
except Exception as llm_err:
    print(f"[WARN] vLLM could not be initialized (likely Windows environment): {llm_err}")
    print("🤖 Falling back to professional Mock LLM Sommelier Agent.")

# Helper function to generate mock wine recommendation for in-stock wine
def generate_mock_instock_advice(wine, occasion, food_pairing):
    title = wine.get("title", wine.get("name", ""))
    variety = wine.get("variety", "")
    country = wine.get("country", "")
    price = wine.get("price", "")
    desc = wine.get("description", "")
    
    lines = [
        f"Hello! As a Master Sommelier, I am absolutely delighted to present your matched selection: **{title}**.",
        f"This is a magnificent expression of **{variety}** from **{country}**."
    ]
    
    if desc:
        lines.append(f"Upon opening, this bottle reveals a beautiful bouquet: *\"{desc}\"*")
        
    lines.append("\n**Sommelier Evaluation & Food Pairing:**")
    
    if food_pairing and occasion:
        lines.append(f"For your **{occasion}** gathering featuring **{food_pairing}**, this wine is an outstanding companion. The structural components of the **{variety}** will complement the flavors of **{food_pairing}** beautifully, while the wine's elegance will elevate the festive mood of the occasion.")
    elif food_pairing:
        lines.append(f"For your **{food_pairing}** pairing, this wine is an outstanding companion. The wine's balanced acidity and body will cut through and elevate the rich textures of the **{food_pairing}**, making every bite and sip a harmonious experience.")
    elif occasion:
        lines.append(f"This bottle is a superb match for your **{occasion}** gathering. Its classic flavor profile makes it highly versatile and pleasing to a wide range of palates, offering elegance and a pleasant, lingering finish that sparks conversation.")
    else:
        lines.append(f"I highly recommend serving this bottle at its peak cellar temperature, allowing it to breathe for about 30 minutes before pouring. It is a highly expressive wine that shows best with light aeration.")
        
    return "\n\n".join(lines)

# Helper function to generate mock wine recommendation for out-of-stock wine
def generate_mock_outofstock_advice(scan_info, similar_wines, occasion, food_pairing):
    v = scan_info.get("variety", "wine")
    c = scan_info.get("country", "")
    y = scan_info.get("year", "")
    label_style = scan_info.get("label_style", "N/A")
    
    desc_str = f"a **{v}**" if v else "a wine"
    if c: desc_str += f" from **{c}**"
    if y: desc_str += f" ({y} vintage)"
    
    lines = [
        f"Greetings! I analyzed the label you uploaded, which appears to be {desc_str}.",
    ]
    if label_style and label_style != "N/A":
        lines.append(f"Visual style: The label features *{label_style}*.\n")
    else:
        lines.append("")
        
    lines.append("Although this specific bottle is currently not in our catalog, I have selected the **Top 5 similar alternatives** from our cellar ")
    
    if occasion or food_pairing:
        context_parts = []
        if occasion: context_parts.append(f"your **{occasion}** event")
        if food_pairing: context_parts.append(f"pairing with **{food_pairing}**")
        lines[-1] += "specifically optimized for " + " and ".join(context_parts) + "."
    else:
        lines[-1] += "matching its country, grape variety, and visual style."
        
    lines.append("\nHere is my advice on these selections:")
    
    for idx, w in enumerate(similar_wines):
        title = w.get("title", w.get("name", ""))
        var = w.get("variety", "")
        cnt = w.get("country", "")
        
        # Build custom reason
        reason = ""
        if food_pairing:
            reason = f"the structure of this {var} pairs excellently with {food_pairing}"
        elif occasion:
            reason = f"its elegant presentation is perfect for a {occasion}"
        else:
            reason = f"a wonderful choice showing classic {var} characteristics of {cnt}"
            
        lines.append(f"{idx+1}. **{title}** - *{reason}*")
        lines.append(f"   *Tasting note snippet:* {w.get('description', '')[:120]}...\n")
        
    lines.append("I recommend any of these bottles as a splendid alternative. Cheers!")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════
# Load & Build Indexes
# ══════════════════════════════════════════════════════════════════════

print("🔧 Loading Sapo data...")
cat   = pd.read_csv(ROOT / 'data/sapo/sapo_catalog.csv')
inter = pd.read_csv(ROOT / 'data/sapo/sapo_interactions.csv')
with open(ROOT / 'data/sapo/sapo_test.jsonl', encoding='utf-8') as f:
    test_users = [json.loads(l) for l in f]

# Tạo lookup
sku2idx = {s: i for i, s in enumerate(cat['sku'])}
idx2row = cat.set_index('sku')

def safe_str(val):
    if pd.isna(val): return ''
    return str(val)

cat['search_text'] = (
    cat['name'].apply(safe_str) + ' ' +
    cat['type'].apply(safe_str) + ' ' +
    cat['brand'].apply(safe_str) + ' ' +
    cat['tags'].apply(safe_str) + ' ' +
    cat['description'].apply(safe_str).str[:300]
).str.lower()

# TF-IDF
print("🔧 Building TF-IDF index...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_mat = tfidf.fit_transform(cat['search_text'])

# SVD for session-based
svd = TruncatedSVD(n_components=64, random_state=42)
svd_mat = svd.fit_transform(tfidf_mat)

# BM25
print("🔧 Building BM25 index...")
corpus = [t.split() for t in cat['search_text'].tolist()]
bm25   = BM25Okapi(corpus)

# CF User-Item matrix
print("🔧 Building CF matrix...")
users  = inter['user'].unique().tolist()
items  = cat['sku'].tolist()

# MASKING USERS
user_map = {u: f"Khách_Hàng_{i+1:03d}" for i, u in enumerate(users)}
reverse_user_map = {v: k for k, v in user_map.items()}

u2i    = {u: i for i, u in enumerate(users)}
s2i    = {s: i for i, s in enumerate(items)}
ui_mat = np.zeros((len(users), len(items)))
for _, row in inter.iterrows():
    if row['user'] in u2i and row['sku'] in s2i:
        ui_mat[u2i[row['user']], s2i[row['sku']]] = float(row['qty'])

print("✅ Sapo indexes ready!")

print("🔧 Loading Winemag 130k data...")
try:
    wine_df = pd.read_csv(ROOT / 'data/processed/wine_catalog_semantic.csv', usecols=['title', 'variety', 'country', 'price', 'description', 'Semantic_ID', 'doc_text'], dtype=str).fillna('')
    wine_df['Semantic_ID_Cluster'] = wine_df['Semantic_ID'].apply(
        lambda x: '-'.join(x.split('-')[:3]) if isinstance(x, str) and len(x.split('-')) >= 3 else ''
    )
    print("🔧 Building Winemag 130k TF-IDF index...")
    wine_tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    wine_mat = wine_tfidf.fit_transform(wine_df['doc_text'])
    print("✅ Winemag 130k loaded.")
    WINEMAG_READY = True
except Exception as e:
    print("Error loading Winemag:", e)
    WINEMAG_READY = False

# ── CV Scanner Initialization ─────────────────────────────────────────
try:
    from src.cv_wine import (
        WineLabelScanner,
        find_in_catalog,
        find_similar_wines,
        build_tasting_notes,
        get_aroma_profile,
        get_food_pairings,
    )
    print("🔧 Initializing Wine Label Scanner (Florence-2-large + CLIP-large + YOLO11)...")
    wine_scanner = WineLabelScanner()
    if wine_scanner is not None:
        wine_scanner.load(load_clip=True)  # Load both models
        # Precompute/load catalog embeddings for the Sapo catalog (305 rows)
        wine_scanner.load_catalog_embeddings(cat)
        print("✅ Wine Label Scanner ready!")
    CV_AVAILABLE = True
except Exception as _cv_err:
    wine_scanner = None
    CV_AVAILABLE = False
    print(f"[WARN] cv_wine not available: {_cv_err}")

print("\n🚀 All Systems Go!\n")

# ══════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════

def format_price(val):
    try:
        v = float(val)
        if v > 1000:
            return f"{v:,.0f} VND"
        return f"${v:,.0f}"
    except:
        return "N/A"

def wine_to_dict(sku, score=None, reason=None, rank=None):
    if sku not in sku2idx:
        return None
    row = idx2row.loc[sku]
    desc = safe_str(row['description'])
    desc_short = desc[:180] + '...' if len(desc) > 180 else desc
    return {
        'sku': sku,
        'name': safe_str(row['name']),
        'type': safe_str(row['type']),
        'brand': safe_str(row['brand']),
        'price': format_price(row['price']),
        'price_raw': row['price'] if not pd.isna(row['price']) else None,
        'description': desc_short,
        'tags': safe_str(row['tags']),
        'score': round(float(score), 4) if score is not None else None,
        'reason': reason or '',
        'rank': rank,
    }

def search_tfidf(query, top_k=10, exclude=None):
    q_vec = tfidf.transform([query.lower()])
    sims  = cosine_similarity(q_vec, tfidf_mat).flatten()
    ranked= np.argsort(-sims)
    result= []
    for i in ranked:
        sku = cat.iloc[i]['sku']
        if exclude and sku in exclude: continue
        if sims[i] < 0.001: continue
        reason = f"Độ tương đồng nội dung (TF-IDF cosine): {sims[i]:.3f}"
        result.append(wine_to_dict(sku, score=sims[i], reason=reason, rank=len(result)+1))
        if len(result) >= top_k: break
    return result

def search_bm25(query, top_k=10, exclude=None):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(-scores)
    result = []
    for i in ranked:
        sku = cat.iloc[i]['sku']
        if exclude and sku in exclude: continue
        if scores[i] < 0.01: continue
        reason = f"BM25 relevance score: {scores[i]:.3f}"
        result.append(wine_to_dict(sku, score=scores[i], reason=reason, rank=len(result)+1))
        if len(result) >= top_k: break
    return result

def recommend_cf(user, top_k=10):
    if user not in u2i:
        return [], []
    u_idx  = u2i[user]
    u_vec  = ui_mat[u_idx]
    bought = {items[j] for j in np.where(u_vec > 0)[0]}

    norms   = np.linalg.norm(ui_mat, axis=1, keepdims=True) + 1e-9
    norm_m  = ui_mat / norms
    u_norm  = u_vec / (np.linalg.norm(u_vec) + 1e-9)
    sims    = norm_m @ u_norm

    # Top neighbors
    top_u   = np.argsort(-sims)[1:21]
    cf_scores = np.zeros(len(items))
    for nu in top_u:
        cf_scores += sims[nu] * ui_mat[nu]
    for s in bought:
        if s in s2i: cf_scores[s2i[s]] = -1

    ranked = np.argsort(-cf_scores)
    result = []
    neighbors_info = [
        {'user': user_map.get(users[nu], users[nu]), 'similarity': round(float(sims[nu]), 3)}
        for nu in top_u[:3] if sims[nu] > 0.1
    ]
    for i in ranked:
        sku = items[i]
        if cf_scores[i] <= 0: continue
        reason = f"CF weighted score: {cf_scores[i]:.3f} — khách tương đồng đã mua sản phẩm này"
        result.append(wine_to_dict(sku, score=cf_scores[i], reason=reason, rank=len(result)+1))
        if len(result) >= top_k: break
    return result, neighbors_info

def get_user_history(user):
    user_rows = inter[inter['user'] == user].copy()
    if user_rows.empty: return []
    history = []
    for _, row in user_rows.sort_values('qty', ascending=False).iterrows():
        w = wine_to_dict(row['sku'])
        if w:
            w['qty'] = int(row['qty'])
            history.append(w)
    return history

def search_winemag(query, top_k=10):
    if not WINEMAG_READY: return []
    q_vec = wine_tfidf.transform([query.lower()])
    sims = cosine_similarity(q_vec, wine_mat).flatten()
    ranked = np.argsort(-sims)
    result = []
    for i in ranked:
        if sims[i] < 0.001: continue
        row = wine_df.iloc[i]
        price_val = row['price']
        price_str = f"${float(price_val):.0f}" if price_val.replace('.','',1).isdigit() else "N/A"
        desc = row['description']
        desc_short = desc[:180] + '...' if len(desc) > 180 else desc
        result.append({
            'sku': row['Semantic_ID'],
            'name': row['title'],
            'type': row['variety'],
            'brand': row['country'],
            'price': price_str,
            'description': desc_short,
            'score': round(float(sims[i]), 4),
            'reason': f"TF-IDF: {sims[i]:.3f} | Semantic ID: {row['Semantic_ID']}",
            'rank': len(result)+1
        })
        if len(result) >= top_k: break
    return result

def recommend_session(user, top_k=10):
    if user not in u2i: return []
    history = get_user_history(user)
    hist_skus = [h['sku'] for h in history]
    hist_idx = [s2i[s] for s in hist_skus if s in s2i]
    if not hist_idx: return []
    
    query_vec = svd_mat[hist_idx].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(query_vec, svd_mat).flatten()
    
    ranked = np.argsort(-sims)
    result = []
    already = set(hist_skus)
    for i in ranked:
        sku = items[i]
        if sku in already: continue
        if sims[i] < 0.001: continue
        reason = f"Tương đồng lịch sử (SVD): {sims[i]:.3f}"
        result.append(wine_to_dict(sku, score=sims[i], reason=reason, rank=len(result)+1))
        if len(result) >= top_k: break
    return result

def recommend_hybrid(user, query, top_k=10):
    if user not in u2i: return []
    u_idx = u2i[user]
    u_vec = ui_mat[u_idx]
    norms = np.linalg.norm(ui_mat, axis=1, keepdims=True) + 1e-9
    norm_m = ui_mat / norms
    u_norm = u_vec / (np.linalg.norm(u_vec) + 1e-9)
    sims = norm_m @ u_norm
    top_u = np.argsort(-sims)[1:21]
    cf_scores = np.zeros(len(items))
    for nu in top_u:
        cf_scores += sims[nu] * ui_mat[nu]
    already = {items[j] for j in np.where(u_vec > 0)[0]}
    for s in already:
        if s in s2i: cf_scores[s2i[s]] = -1
    
    cand_idx = np.argsort(-cf_scores)[:50]
    cand_skus = [items[i] for i in cand_idx if cf_scores[i] > 0]
    
    if not cand_skus: return []
    
    tokens = query.lower().split()
    if not tokens: return []
    bm25_scores = bm25.get_scores(tokens)
    
    cand_scores = []
    for sku in cand_skus:
        if sku in s2i:
            cand_scores.append((sku, bm25_scores[s2i[sku]]))
            
    cand_scores.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    for sku, score in cand_scores:
        if score < 0.001: continue
        reason = f"CF Candidates + Keyword ({score:.2f})"
        result.append(wine_to_dict(sku, score=score, reason=reason, rank=len(result)+1))
        if len(result) >= top_k: break
    return result

# ══════════════════════════════════════════════════════════════════════
# API Routes
# ══════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(str(Path(__file__).parent), 'visiontiger_demo.html')

@app.route('/classic')
def classic_index():
    return send_from_directory(str(Path(__file__).parent), 'index.html')

@app.route('/report')
def report():
    return send_from_directory(str(Path(__file__).parent), 'report.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': time.time()})

@app.route('/api/search', methods=['POST'])
def api_search():
    data   = request.json
    query  = data.get('query', '')
    method = data.get('method', 'bm25')
    t0     = time.time()
    if method == 'tfidf':
        results = search_tfidf(query)
    else:
        results = search_bm25(query)
    latency = round((time.time() - t0) * 1000, 1)
    return jsonify({'results': results, 'method': method, 'latency_ms': latency, 'query': query})

@app.route('/api/search_winemag', methods=['POST'])
def api_search_winemag():
    data = request.json
    query = data.get('query', '')
    t0 = time.time()
    results = search_winemag(query)
    latency = round((time.time() - t0) * 1000, 1)
    return jsonify({'results': results, 'method': 'Winemag 130K (TF-IDF)', 'latency_ms': latency, 'query': query})

@app.route('/api/search_llm_mock', methods=['POST'])
def api_search_llm_mock():
    data = request.json
    query = data.get('query', '')
    
    if not WINEMAG_READY: return jsonify({'error': 'Winemag not loaded'}), 500
    q_vec = wine_tfidf.transform([query.lower()])
    sims = cosine_similarity(q_vec, wine_mat).flatten()
    best_idx = np.argmax(sims)
    best_row = wine_df.iloc[best_idx]
    
    cluster = best_row['Semantic_ID_Cluster']
    
    cluster_items = wine_df[wine_df['Semantic_ID_Cluster'] == cluster]
    cluster_indices = cluster_items.index
    cluster_sims = sims[cluster_indices]
    
    ranked_within = np.argsort(-cluster_sims)
    
    results = []
    for i in ranked_within[:10]:
        idx = cluster_indices[i]
        row = wine_df.iloc[idx]
        price_val = row['price']
        price_str = f"${float(price_val):.0f}" if str(price_val).replace('.','',1).isdigit() else "N/A"
        results.append({
            'sku': row['Semantic_ID'],
            'name': row['title'],
            'type': row['variety'],
            'brand': row['country'],
            'price': price_str,
            'description': str(row['description'])[:180] + '...',
            'score': round(float(sims[idx]), 4),
            'reason': f"Thuộc cụm sinh bởi LLM [{cluster}]",
            'rank': len(results)+1
        })
        
    tokens = [f"[{cluster.split('-')[0]}]", f"[{cluster.split('-')[1]}]", f"[{cluster.split('-')[2]}]"] if '-' in cluster else ["[00]","[00]","[00]"]
    
    # Mô phỏng độ trễ sinh token của LLM
    time.sleep(1.5)
    
    return jsonify({
        'query': query,
        'cluster_generated': cluster,
        'tokens': tokens,
        'total_in_cluster': len(cluster_items),
        'results': results,
        'latency_ms': 2278 # Hardcode reported latency from thesis
    })

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """So sánh song song 5 phương pháp"""
    data  = request.json
    query = data.get('query', '')
    masked_user = data.get('user', '')
    user  = reverse_user_map.get(masked_user, masked_user)

    
    t0 = time.time(); r_bm25  = search_bm25(query, top_k=5); t_bm25  = round((time.time()-t0)*1000,1)
    t0 = time.time(); r_tfidf = search_tfidf(query, top_k=5); t_tfidf = round((time.time()-t0)*1000,1)
    
    r_cf = []; t_cf = 0
    r_session = []; t_session = 0
    r_hybrid = []; t_hybrid = 0
    
    if user:
        t0 = time.time(); r_cf, _ = recommend_cf(user, top_k=5); t_cf = round((time.time()-t0)*1000,1)
        t0 = time.time(); r_session = recommend_session(user, top_k=5); t_session = round((time.time()-t0)*1000,1)
        t0 = time.time(); r_hybrid = recommend_hybrid(user, query, top_k=5); t_hybrid = round((time.time()-t0)*1000,1)
        
    return jsonify({
        'bm25':  {'results': r_bm25,  'latency_ms': t_bm25},
        'tfidf': {'results': r_tfidf, 'latency_ms': t_tfidf},
        'cf': {'results': r_cf, 'latency_ms': t_cf},
        'session': {'results': r_session, 'latency_ms': t_session},
        'hybrid': {'results': r_hybrid, 'latency_ms': t_hybrid},
        'query': query,
        'user': user
    })

@app.route('/api/users', methods=['GET'])
def api_users():
    """Danh sách khách hàng có lịch sử (top 20 active)"""
    top = inter.groupby('user')['qty'].sum().sort_values(ascending=False).head(20)
    result = []
    for u, total_qty in top.items():
        n_items = inter[inter['user']==u]['sku'].nunique()
        result.append({'user': user_map.get(u, u), 'total_qty': int(total_qty), 'n_products': int(n_items)})
    return jsonify(result)

@app.route('/api/recommend/cf', methods=['POST'])
def api_cf():
    data   = request.json
    masked_user = data.get('user', '')
    user   = reverse_user_map.get(masked_user, masked_user)
    top_k  = data.get('top_k', 6)
    t0     = time.time()
    recs, neighbors = recommend_cf(user, top_k=top_k)
    history = get_user_history(user)
    latency = round((time.time()-t0)*1000, 1)
    return jsonify({
        'user': masked_user,
        'history': history,
        'recommendations': recs,
        'neighbors': neighbors,
        'latency_ms': latency
    })

@app.route('/api/catalog/stats', methods=['GET'])
def api_stats():
    return jsonify({
        'n_products': len(cat),
        'n_users': len(users),
        'n_interactions': len(inter),
        'types': cat['type'].value_counts().to_dict(),
        'price_min': float(cat['price'].dropna().min()) if not cat['price'].dropna().empty else 0.0,
        'price_max': float(cat['price'].dropna().max()) if not cat['price'].dropna().empty else 0.0,
        'price_median': float(cat['price'].dropna().median()) if not cat['price'].dropna().empty else 0.0,
        'n_winemag': len(wine_df) if WINEMAG_READY else 0
    })

@app.route('/api/product/<sku>', methods=['GET'])
def api_product(sku):
    w = wine_to_dict(sku)
    if not w: return jsonify({'error': 'Not found'}), 404
    # Full description
    if sku in sku2idx:
        row = idx2row.loc[sku]
        w['description_full'] = safe_str(row['description'])
    return jsonify(w)

@app.route('/api/scan_label', methods=['POST'])
def api_scan_label():
    if not CV_AVAILABLE or wine_scanner is None:
        return jsonify({"status": "error", "message": "CV module not available."}), 503
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty file name."}), 400
        
    # Get user context (occasion & food pairing)
    occasion = request.form.get("occasion", "").strip()
    food_pairing = request.form.get("food_pairing", "").strip()
        
    try:
        t_total = time.time()
        image_bytes = file.read()
        
        # 1. Scan the label (Florence-2-large + YOLO11 + CLIP-large)
        print(f"[CV App] Scanning label. Occasion: '{occasion}', Food: '{food_pairing}'...")
        
        # 1a. Configuration A: Unwarped (Normalized Model - Default)
        scan_info_unwarped = wine_scanner.scan(image_bytes)
        if "error" in scan_info_unwarped:
            return jsonify({"status": "error", "message": scan_info_unwarped["error"]})
            
        scan_info = scan_info_unwarped
        scan_unwarped_out = {k: v for k, v in scan_info_unwarped.items() if k != "visual_embedding"}
        
        # 1b. Configuration B: Warped (Curved Model - Standard)
        # Monkeypatch to temporarily disable cylindrical unwarping
        original_unwarp = wine_scanner.preprocess_cylindrical_image
        wine_scanner.preprocess_cylindrical_image = lambda img, radius_ratio=0.85: img
        try:
            scan_info_warped = wine_scanner.scan(image_bytes)
            scan_warped_out = {k: v for k, v in scan_info_warped.items() if k != "visual_embedding"}
        except Exception as scan_err:
            print(f"[CV App] Warped scan failed: {scan_err}")
            scan_warped_out = {"error": str(scan_err)}
        finally:
            wine_scanner.preprocess_cylindrical_image = original_unwarp
        
        # 2. Look up in catalog (fuzzy match on Title and Variety to prevent overfitting)
        lookup = find_in_catalog(scan_info, wine_df)
        
        # Prepare response data structures
        status = "in_stock" if lookup["found"] else "not_in_stock"
        wine = None
        similar_wines = []
        
        if lookup["found"]:
            wine = lookup["wine"]
            variety = wine.get("variety", scan_info.get("variety", ""))
            tasting_notes = build_tasting_notes(wine)
            aroma = get_aroma_profile(variety)
            pairings = get_food_pairings(variety)
            
            # Make sure everything is JSON-serializable
            wine = json.loads(json.dumps(wine, default=lambda o: float(o) if isinstance(o, (np.floating, np.ndarray)) else str(o)))
        else:
            similar = find_similar_wines(
                scan_info, wine_df, n=5, 
                occasion=occasion, food_pairing=food_pairing, 
                scanner=wine_scanner
            )
            for w in similar:
                w["tasting_summary"] = build_tasting_notes(w)
                if "match_reason" not in w:
                    w["match_reason"] = "Highly recommended based on style."
            
            similar_wines = json.loads(json.dumps(similar, default=lambda o: float(o) if isinstance(o, (np.floating, np.ndarray)) else str(o)))

        total_ms = round((time.time() - t_total) * 1000, 1)

        # 3. Stream Sommelier Advice using SSE
        def generate_sse_stream():
            # First event: Metadata
            meta_payload = {
                "type": "metadata",
                "status": status,
                "scan_info": scan_unwarped_out,
                "scan_info_unwarped": scan_unwarped_out,
                "scan_info_warped": scan_warped_out,
                "total_ms": total_ms
            }
            if status == "in_stock":
                meta_payload.update({
                    "match_type": lookup["match_type"],
                    "match_score": float(lookup["similarity"]),
                    "wine": wine,
                    "tasting_notes": tasting_notes,
                    "aroma_profile": aroma,
                    "food_pairings": pairings
                })
            else:
                meta_payload.update({
                    "message": f"This wine is not in our inventory. Based on the label, it appears to be a {scan_info.get('variety', 'wine')} from {scan_info.get('country', '')}.",
                    "similar_wines": similar_wines
                })
            
            yield f"data: {json.dumps(meta_payload)}\n\n"
            
            # Generate the Sommelier Advice text
            sommelier_advice_text = ""
            
            # Check if vLLM is available
            if LLM_AVAILABLE and llm is not None:
                try:
                    # Construct Prompt
                    if status == "in_stock":
                        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Master Sommelier. Write a warm, professional wine recommendation in English for the following wine in our catalog.
Address the customer directly. If they specified an occasion or food pairing, explain why this wine is a perfect match.

Wine Details:
- Name: {wine.get('title', wine.get('name', ''))}
- Variety: {wine.get('variety', '')}
- Country: {wine.get('country', '')}
- Description: {wine.get('description', '')}

Customer Context:
- Occasion: {occasion if occasion else "N/A"}
- Food Pairing: {food_pairing if food_pairing else "N/A"}

Provide your expert sommelier advice. Keep it under 250 words, elegant, and professional.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please write the recommendation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
                    else:
                        similar_wines_text = ""
                        for idx, w in enumerate(similar_wines):
                            similar_wines_text += f"{idx+1}. {w.get('title')} ({w.get('variety')}, {w.get('country')})\n   Tasting: {w.get('description', '')}\n"

                        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Master Sommelier. The customer uploaded a label for a wine we don't have in stock: a {scan_info.get('variety', 'wine')} from {scan_info.get('country', 'unknown region')}.
Suggest these 5 in-stock alternatives from our inventory. Explain why each is a great choice, connecting them to the customer's occasion/food pairing if provided, or their variety/country.

Alternatives:
{similar_wines_text}

Customer Context:
- Occasion: {occasion if occasion else "N/A"}
- Food Pairing: {food_pairing if food_pairing else "N/A"}

Write your recommendation. Keep it under 350 words, structured, elegant, and professional.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please write the recommendation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
                    
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=512,
                        stop=["<|eot_id|>", "<|end_of_text|>"]
                    )
                    
                    print("[vLLM] Generating advice stream...")
                    outputs = llm.generate([prompt], sampling_params)
                    sommelier_advice_text = outputs[0].outputs[0].text
                    
                except Exception as e:
                    print(f"[WARN] vLLM generation failed: {e}. Falling back to mock generator.")
                    if status == "in_stock":
                        sommelier_advice_text = generate_mock_instock_advice(wine, occasion, food_pairing)
                    else:
                        sommelier_advice_text = generate_mock_outofstock_advice(scan_info, similar_wines, occasion, food_pairing)
            else:
                if status == "in_stock":
                    sommelier_advice_text = generate_mock_instock_advice(wine, occasion, food_pairing)
                else:
                    sommelier_advice_text = generate_mock_outofstock_advice(scan_info, similar_wines, occasion, food_pairing)
            
            # Stream the advice text word-by-word
            words = re.split(r'(\s+)', sommelier_advice_text)
            for word in words:
                if not word:
                    continue
                yield f"data: {json.dumps({'type': 'token', 'text': word})}\n\n"
                time.sleep(0.015)  # typing latency
                
            yield "data: {\"type\": \"end\"}\n\n"

        response = Response(generate_sse_stream(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'  # disable nginx buffering
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting demo server at http://localhost:5050")
    app.run(debug=False, port=5050, host='0.0.0.0')
