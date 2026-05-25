"""
baseline_eval.py  — Baseline Evaluation Suite
==============================================
Methods evaluated:
  1.  BM25 (Okapi BM25, tuned k1/b)
  2.  BM25+ Enhanced (field-weighted query expansion)
  3.  TF-IDF Content-Based CF (ngram, sublinear_tf)
  4.  TF-IDF + LSA / SVD (Latent Semantic Analysis)
  5.  Hybrid BM25 + TF-IDF (score fusion)
  6.  Popularity-Based (country+variety frequency prior)
  7.  Random Baseline (lower bound)
  --- Improved-Recall methods (address region-ambiguity problem) ---
  8.  Structured Filter + TF-IDF
       Root cause of low Recall: queries only specify country+variety+price;
       the region part of the Semantic ID (e.g. RHNE, MART) is never
       mentioned in the query. BM25/TF-IDF wastes budget retrieving wines
       from the wrong country or variety. Fix: hard-filter catalog by the
       extracted country + variety, then rank within that filtered set.
  9.  Structured Filter + BM25 (within-group BM25)
  10. Structured Filter + Price Re-rank
       After filtering by country+variety, sort candidates by price
       proximity to the budget hint in the query. Price is the strongest
       discriminator within a country+variety group.

Dataset: Kaggle Wine Reviews (winemag-data-130k-v2.csv)
Cite   : Wine Enthusiast Magazine (2017). Kaggle.
"""

import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))
import config as cfg

import json, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CATALOG_CSV = str(cfg.WINE_CSV)
TEST_FILE   = str(cfg.TEST_JSONL)
OUTPUT_CSV  = str(cfg.BASELINE_CSV)
K_VALUES    = [1, 5, 10]

import argparse
parser = argparse.ArgumentParser(description="Baseline Evaluation Suite")
parser.add_argument("--eval_size", type=int, default=1000,
                    help="Number of test queries to evaluate (default: 1000, use 0 for ALL)")
args = parser.parse_args()
EVAL_SIZE = args.eval_size if args.eval_size > 0 else 999999
print(f"[CONFIG] eval_size={EVAL_SIZE:,}")

# ─── AUTO-INSTALL ─────────────────────────────────────────────────────────────
def _install(pkg, imp=None):
    try: __import__(imp or pkg)
    except ImportError: os.system(f"{sys.executable} -m pip install {pkg} -q")

_install("rank_bm25")
_install("scikit-learn", "sklearn")

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ─── SEMANTIC ID HELPERS ─────────────────────────────────────────────────────
def clean_text(t):
    if pd.isna(t): return "UNKN"
    return re.sub(r"[^A-Za-z0-9]", "", str(t)).upper()[:4]

def extract_year(title):
    if pd.isna(title): return "NV"
    m = re.search(r"(19|20)\d{2}", str(title))
    return m.group(0) if m else "NV"

def make_semantic_id(row):
    return (f"{clean_text(row['country'])}-"
            f"{clean_text(row.get('province',''))}-"
            f"{clean_text(row['variety'])}-"
            f"{row['vintage']}")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
def load_catalog(csv_path):
    print(f"\n{'='*60}\nLoading catalog: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["country","variety","description","title"])
    df["vintage"]     = df["title"].apply(extract_year)
    df["Semantic_ID"] = df.apply(make_semantic_id, axis=1)
    # Rich document text for retrieval
    df["doc_text"] = df.apply(
        lambda r: (f"{r['variety']} {r['country']} "
                   f"{r.get('province','') or ''} "
                   f"{r.get('region_1','') or ''} "
                   f"{r.get('winery','') or ''} "
                   f"{r['description']}"), axis=1)
    # Lightweight text for BM25+ field weighting (repeat key fields 3x)
    df["doc_text_bm25p"] = df.apply(
        lambda r: (f"{r['variety']} {r['variety']} {r['variety']} "
                   f"{r['country']} {r['country']} "
                   f"{r.get('province','') or ''} "
                   f"{r.get('region_1','') or ''} "
                   f"{r['description']}"), axis=1)
    df = df.reset_index(drop=True)
    print(f"  Catalog : {len(df):,} wines | Unique IDs: {df['Semantic_ID'].nunique():,}")
    return df

def load_test(fp, n):
    data = [json.loads(l) for l in open(fp)][:n]
    print(f"  Test queries: {len(data):,}")
    return data

# ─── METRICS ─────────────────────────────────────────────────────────────────
def recall_at_k(ret, tgt, k):
    return 1.0 if tgt in ret[:k] else 0.0

def ndcg_at_k(ret, tgt, k):
    for i, r in enumerate(ret[:k]):
        if r == tgt: return 1.0 / np.log2(i + 2)
    return 0.0

def mrr_score(ret, tgt, K=10):
    for i, r in enumerate(ret[:K]):
        if r == tgt: return 1.0 / (i + 1)
    return 0.0

def intent_match(ret, tgt, k):
    tc, tv = tgt.split('-')[0], (tgt.split('-')[2] if len(tgt.split('-'))>2 else "")
    for r in ret[:k]:
        p = r.split('-')
        if len(p)>2 and p[0]==tc and p[2]==tv: return 1.0
    return 0.0

def country_match(ret, tgt, k):
    tc = tgt.split('-')[0]
    return 1.0 if any(r.split('-')[0]==tc for r in ret[:k]) else 0.0

def variety_match(ret, tgt, k):
    tv = tgt.split('-')[2] if len(tgt.split('-'))>2 else ""
    return 1.0 if any(len(r.split('-'))>2 and r.split('-')[2]==tv for r in ret[:k]) else 0.0

def compute_metrics(ret, tgt, k_values):
    row = {}
    for k in k_values:
        row[f"Recall@{k}"]       = recall_at_k(ret, tgt, k)
        row[f"NDCG@{k}"]         = ndcg_at_k(ret, tgt, k)
        row[f"IntentMatch@{k}"]  = intent_match(ret, tgt, k)
        row[f"CountryMatch@{k}"] = country_match(ret, tgt, k)
        row[f"VarietyMatch@{k}"] = variety_match(ret, tgt, k)
    row["MRR"] = mrr_score(ret, tgt, max(k_values))
    return row

def aggregate(records, k_values):
    df = pd.DataFrame(records)
    s  = {}
    for k in k_values:
        for col in [f"Recall@{k}",f"NDCG@{k}",f"IntentMatch@{k}",
                    f"CountryMatch@{k}",f"VarietyMatch@{k}"]:
            s[col] = df[col].mean()
    s["MRR"] = df["MRR"].mean()
    return df, s

# ─── METHOD 1: BM25 (tuned) ──────────────────────────────────────────────────
def run_bm25(cat, test, k_values):
    print(f"\n{'─'*60}\nMETHOD 1: BM25 (Okapi, k1=1.2 b=0.65 — tuned for wine vocab)")
    t0 = time.time()
    corpus = [d.lower().split() for d in cat["doc_text"]]
    bm25   = BM25Okapi(corpus, k1=1.2, b=0.65)
    print(f"  Index: {time.time()-t0:.1f}s")
    K = max(k_values); records=[]; lats=[]
    for item in tqdm(test, desc="  BM25"):
        t_q = time.time()
        sc  = bm25.get_scores(item["instruction"].lower().split())
        top = np.argsort(sc)[::-1][:K]
        lats.append((time.time()-t_q)*1000)
        ret = cat.iloc[top]["Semantic_ID"].tolist()
        r   = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── METHOD 2: BM25+ Enhanced (field weighting) ──────────────────────────────
def run_bm25_enhanced(cat, test, k_values):
    print(f"\n{'─'*60}\nMETHOD 2: BM25+ Enhanced (variety/country fields repeated ×3)")
    t0 = time.time()
    corpus = [d.lower().split() for d in cat["doc_text_bm25p"]]
    bm25   = BM25Okapi(corpus, k1=1.5, b=0.75)
    print(f"  Index: {time.time()-t0:.1f}s")
    K = max(k_values); records=[]; lats=[]

    # Build query expansion: extract explicit variety/country from instruction
    variety_keywords = [
        "cabernet sauvignon","pinot noir","chardonnay","sauvignon blanc",
        "merlot","syrah","shiraz","riesling","malbec","grenache","tempranillo",
        "zinfandel","viognier","prosecco","champagne","rosé","rose",
        "red blend","white blend","sparkling",
    ]
    country_keywords = [
        "france","italy","spain","us","usa","argentina","chile","australia",
        "germany","austria","portugal","new zealand","south africa",
    ]

    for item in tqdm(test, desc="  BM25+"):
        instr = item["instruction"].lower()
        # Repeat detected entities for field boosting
        boosted = instr
        for v in variety_keywords:
            if v in instr: boosted += f" {v} {v}"
        for c in country_keywords:
            if c in instr: boosted += f" {c} {c}"

        t_q = time.time()
        sc  = bm25.get_scores(boosted.split())
        top = np.argsort(sc)[::-1][:K]
        lats.append((time.time()-t_q)*1000)
        ret = cat.iloc[top]["Semantic_ID"].tolist()
        r   = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── METHOD 3: TF-IDF CF ─────────────────────────────────────────────────────
def run_tfidf(cat, test, k_values):
    print(f"\n{'─'*60}\nMETHOD 3: TF-IDF Content-Based CF (max_feat=50k, ngram(1,2))")
    t0  = time.time()
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1,2), sublinear_tf=True,
                          min_df=2, strip_accents="unicode")
    mat = vec.fit_transform(cat["doc_text"])
    print(f"  Matrix: {mat.shape}  |  built: {time.time()-t0:.1f}s")
    K=max(k_values); records=[]; lats=[]; BATCH=128
    for b in tqdm(range(0, len(test), BATCH), desc="  TF-IDF"):
        batch = test[b:b+BATCH]
        t_q   = time.time()
        qmat  = vec.transform([x["instruction"] for x in batch])
        sims  = cosine_similarity(qmat, mat)
        lat   = (time.time()-t_q)*1000/len(batch)
        lats.extend([lat]*len(batch))
        for i, item in enumerate(batch):
            top = np.argsort(sims[i])[::-1][:K]
            ret = cat.iloc[top]["Semantic_ID"].tolist()
            r   = {"target_id": item["target_id"]}
            r.update(compute_metrics(ret, item["target_id"], k_values))
            records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── METHOD 4: TF-IDF + LSA (SVD 200 dims) ──────────────────────────────────
def run_lsa(cat, test, k_values):
    print(f"\n{'─'*60}\nMETHOD 4: TF-IDF + LSA (SVD 200 dims, Latent Semantic Analysis)")
    t0  = time.time()
    vec = TfidfVectorizer(max_features=80_000, ngram_range=(1,2), sublinear_tf=True,
                          min_df=2, strip_accents="unicode")
    mat = vec.fit_transform(cat["doc_text"])
    svd = TruncatedSVD(n_components=200, random_state=42)
    cat_lsa = normalize(svd.fit_transform(mat))
    print(f"  LSA shape: {cat_lsa.shape}  |  built: {time.time()-t0:.1f}s")
    K=max(k_values); records=[]; lats=[]; BATCH=256
    for b in tqdm(range(0, len(test), BATCH), desc="  LSA"):
        batch = test[b:b+BATCH]
        t_q   = time.time()
        qmat  = normalize(svd.transform(vec.transform([x["instruction"] for x in batch])))
        sims  = qmat @ cat_lsa.T
        lat   = (time.time()-t_q)*1000/len(batch)
        lats.extend([lat]*len(batch))
        for i, item in enumerate(batch):
            top = np.argsort(sims[i])[::-1][:K]
            ret = cat.iloc[top]["Semantic_ID"].tolist()
            r   = {"target_id": item["target_id"]}
            r.update(compute_metrics(ret, item["target_id"], k_values))
            records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── METHOD 5: Hybrid BM25 + TF-IDF (score fusion) ──────────────────────────
def run_hybrid(cat, test, k_values, bm25_obj, tfidf_vec, tfidf_mat):
    print(f"\n{'─'*60}\nMETHOD 5: Hybrid BM25 + TF-IDF (0.5 × BM25_norm + 0.5 × TF-IDF cosine)")
    K=max(k_values); records=[]; lats=[]; BATCH=64
    # Pre-normalise BM25 scores row-wise: compute for all queries
    for b in tqdm(range(0, len(test), BATCH), desc="  Hybrid"):
        batch = test[b:b+BATCH]
        t_q   = time.time()
        # TF-IDF scores
        qmat = tfidf_vec.transform([x["instruction"] for x in batch])
        tfidf_scores = cosine_similarity(qmat, tfidf_mat)   # (batch, catalog)
        # BM25 scores
        bm25_scores = np.array([
            bm25_obj.get_scores(x["instruction"].lower().split()) for x in batch
        ])
        # Min-max normalize each row
        def _norm(arr):
            mn, mx = arr.min(axis=1, keepdims=True), arr.max(axis=1, keepdims=True)
            return np.where(mx>mn, (arr-mn)/(mx-mn+1e-9), 0.0)
        fused = 0.5 * _norm(bm25_scores) + 0.5 * tfidf_scores
        lat   = (time.time()-t_q)*1000/len(batch)
        lats.extend([lat]*len(batch))
        for i, item in enumerate(batch):
            top = np.argsort(fused[i])[::-1][:K]
            ret = cat.iloc[top]["Semantic_ID"].tolist()
            r   = {"target_id": item["target_id"]}
            r.update(compute_metrics(ret, item["target_id"], k_values))
            records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── METHOD 6: Popularity-Based ──────────────────────────────────────────────
def run_popularity(cat, test, k_values):
    """
    For each query, detect country+variety from text, then rank wines
    matching those fields by their frequency (popularity) in the catalog.
    Simulates a non-personalised recommendation system.
    """
    print(f"\n{'─'*60}\nMETHOD 6: Popularity-Based (country+variety frequency rank)")
    # Build popularity index: group by (country_code, variety_code) → sorted list of IDs
    cat["_ckey"] = cat["Semantic_ID"].apply(lambda x: (x.split('-')[0], x.split('-')[2]) if len(x.split('-'))>2 else ("",""))
    pop_index = cat.groupby("_ckey")["Semantic_ID"].apply(list).to_dict()

    variety_map = {
        "cabernet sauvignon":"CABE","cabernet franc":"CABE","chardonnay":"CHAR",
        "pinot noir":"PINO","pinot grigio":"PINO","sauvignon blanc":"SAUV",
        "merlot":"MERL","syrah":"SYRA","shiraz":"SYRA","malbec":"MALB",
        "zinfandel":"ZINF","riesling":"RIES","grenache":"GREN",
        "rosé":"ROS","rose":"ROS","tempranillo":"TEMP","prosecco":"PROS",
        "red blend":"REDB","white blend":"WHIT","sparkling":"SPAR",
        "malbec":"MALB","viognier":"VIOG",
    }
    country_map = {
        "france":"FRAN","italy":"ITAL","spain":"SPAI","us":"US","usa":"US",
        "argentina":"ARGE","chile":"CHIL","australia":"AUST","germany":"GERM",
        "austria":"AUST","portugal":"PORT","new zealand":"NEWZ",
        "south africa":"SOUT","israel":"ISRA","canada":"CANA",
    }
    K=max(k_values); records=[]; lats=[]
    for item in tqdm(test, desc="  Popularity"):
        instr = item["instruction"].lower()
        c_code = next((v for k,v in country_map.items() if k in instr), "US")
        v_code = next((v for k,v in variety_map.items() if k in instr), "REDB")
        t_q = time.time()
        key = (c_code, v_code)
        candidates = pop_index.get(key, [])
        # Fallback: same country, any variety
        if len(candidates) < K:
            for (cc, vc), ids in pop_index.items():
                if cc == c_code: candidates = candidates + ids
        # Fallback: global popular
        if len(candidates) < K:
            candidates = cat["Semantic_ID"].tolist()
        # Deduplicate preserving order
        seen=set(); ret=[]
        for cid in candidates:
            if cid not in seen: seen.add(cid); ret.append(cid)
        lats.append((time.time()-t_q)*1000)
        r = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret[:K], item["target_id"], k_values))
        records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── METHOD 7: Random Baseline ───────────────────────────────────────────────
def run_random(cat, test, k_values, seed=42):
    print(f"\n{'─'*60}\nMETHOD 7: Random Baseline (lower bound reference)")
    rng = np.random.default_rng(seed)
    K=max(k_values); records=[]; lats=[]
    all_ids = cat["Semantic_ID"].tolist()
    for item in tqdm(test, desc="  Random"):
        t_q = time.time()
        idx = rng.choice(len(all_ids), size=K, replace=False)
        lats.append((time.time()-t_q)*1000)
        ret = [all_ids[i] for i in idx]
        r   = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s

# ─── PRINT TABLE ─────────────────────────────────────────────────────────────
def print_table(summaries, k_values):
    cols = []
    for k in k_values: cols += [f"Recall@{k}", f"NDCG@{k}", f"IntentMatch@{k}"]
    cols += ["MRR", "Latency_ms"]
    hdr = f"{'Method':<28}" + "".join(f"{c:>13}" for c in cols)
    sep = "─" * len(hdr)
    print(f"\n{'='*60}\n  BASELINE EVALUATION RESULTS\n  Eval size={EVAL_SIZE} | K={k_values}\n{'='*60}")
    print(hdr); print(sep)
    for m, v in summaries.items():
        print(f"{m:<28}" + "".join(f"{v.get(c,0.0):>13.4f}" for c in cols))
    print(sep)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
# ─── STRUCTURED EXTRACTION HELPERS ──────────────────────────────────────────
# These maps let us extract country + variety directly from the free-text query.
# This is the KEY to improving Recall: filter first, then rank within the group.

_VARIETY_MAP = {
    "cabernet sauvignon":"CABE","cabernet franc":"CABE",
    "chardonnay":"CHAR","pinot noir":"PINO","pinot grigio":"PINO",
    "sauvignon blanc":"SAUV","merlot":"MERL","syrah":"SYRA","shiraz":"SYRA",
    "malbec":"MALB","zinfandel":"ZINF","riesling":"RIES","grenache":"GREN",
    "ros\u00e9":"ROS","rose ":"ROS","tempranillo":"TEMP","prosecco":"PROS",
    "red blend":"REDB","white blend":"WHIT","sparkling":"SPAR",
    "viognier":"VIOG","gew\u00fcrztraminer":"GEWR","gewurztraminer":"GEWR",
    "gr\u00fcner veltliner":"GRNE","gruner veltliner":"GRNE",
    "albarino":"ALBA","albar\u00edno":"ALBA","moscato":"MOSC",
    "sangiovese":"SANG","brunello":"SANG","bordeaux":"BORD",
    "blaufr\u00e4nkisch":"BLAU","blaufrankisch":"BLAU",
    "zweigelt":"ZWEI","dolcetto":"DOLC","primitivo":"PRIM",
    "carmenere":"CARM","carmenere":"CARM","godello":"GODE",
    "port":"PORT","port ":"PORT","champagne":"CHAM",
    "corvina":"CORV","nebbiolo":"NEBB","torrontes":"TORR",
    "touriga":"TOUR","vermentino":"VERM","grillo":"GRIL",
    "garnacha":"GARN","monastrell":"MONA","bobal":"BOBA",
    "verdejo":"VERD","airen":"AIRE","txakoli":"TXAK",
    "g-s-m":"GSM","gsm":"GSM",
}
_COUNTRY_MAP = {
    "france":"FRAN","french":"FRAN",
    "italy":"ITAL","italian":"ITAL",
    "spain":"SPAI","spanish":"SPAI",
    "us ":"US","usa":"US","united states":"US","american":"US"," us,":"US",
    "california":"US","oregon":"US","washington":"US",
    "argentina":"ARGE","argentinian":"ARGE",
    "chile":"CHIL","chilean":"CHIL",
    "australia":"AUST","australian":"AUST",
    "germany":"GERM","german":"GERM",
    "austria":"AUST",  # note: AUST used for both Australia and Austria in the dataset
    "portugal":"PORT","portuguese":"PORT",
    "new zealand":"NEWZ",
    "south africa":"SOUT",
    "israel":"ISRA","canada":"CANA",
    "greece":"GREE","greek":"GREE",
    "hungary":"HUNG","romania":"ROMA",
    "slovenia":"SLOV","croatia":"CROA",
    "bulgaria":"BULG","moldova":"MOLD",
}

def extract_query_fields(instruction: str):
    """Extract (country_code, variety_code, price) from a free-text query."""
    q = instruction.lower()
    # Country — try longest matches first
    c_code = None
    for k in sorted(_COUNTRY_MAP, key=len, reverse=True):
        if k in q:
            c_code = _COUNTRY_MAP[k]; break
    # Variety — longest matches first
    v_code = None
    for k in sorted(_VARIETY_MAP, key=len, reverse=True):
        if k in q:
            v_code = _VARIETY_MAP[k]; break
    # Price hint (e.g. "costs around $32.0")
    price = None
    pm = re.search(r'\$([\d.]+)', instruction)
    if pm:
        try: price = float(pm.group(1))
        except: pass
    return c_code, v_code, price


# ─── METHOD 8: Structured Filter + TF-IDF ────────────────────────────────────
def run_structured_tfidf(cat, test, k_values):
    """
    HIGH-RECALL method: solves the region-ambiguity problem.
    Step 1: extract (country, variety) from query.
    Step 2: hard-filter catalog to that country+variety subset.
    Step 3: rank within subset by TF-IDF cosine similarity of description.
    Falls back to full catalog when country/variety not detected.
    """
    print(f"\n{'─'*60}")
    print("METHOD 8: Structured Filter + TF-IDF (country+variety hard filter)")
    print("  Addresses root cause: region code never in query → filter first")

    t0  = time.time()
    # Index for full-catalog fallback
    vec_full = TfidfVectorizer(max_features=80_000, ngram_range=(1,2),
                               sublinear_tf=True, min_df=2, strip_accents="unicode")
    mat_full = vec_full.fit_transform(cat["doc_text"])
    print(f"  Full index: {mat_full.shape}  |  built: {time.time()-t0:.1f}s")

    # Build per-(country,variety) description sub-matrices
    cat["_cv"] = cat["Semantic_ID"].apply(
        lambda x: (x.split('-')[0], x.split('-')[2]) if len(x.split('-'))>2 else ("",""))

    K=max(k_values); records=[]; lats=[]
    hits_filtered = 0
    for item in tqdm(test, desc="  Struct-TF-IDF"):
        t_q = time.time()
        c_code, v_code, _ = extract_query_fields(item["instruction"])

        if c_code and v_code:
            mask   = cat["_cv"] == (c_code, v_code)
            subset = cat[mask].reset_index(drop=True)
            hits_filtered += 1
        else:
            subset = None

        if subset is not None and len(subset) >= K:
            # Build mini TF-IDF on the subset
            sub_mat = vec_full.transform(subset["doc_text"])
            q_vec   = vec_full.transform([item["instruction"]])
            sims    = cosine_similarity(q_vec, sub_mat)[0]
            top_idx = np.argsort(sims)[::-1][:K]
            ret     = subset.iloc[top_idx]["Semantic_ID"].tolist()
        else:
            # Fallback to full catalog
            q_vec = vec_full.transform([item["instruction"]])
            sims  = cosine_similarity(q_vec, mat_full)[0]
            top   = np.argsort(sims)[::-1][:K]
            ret   = cat.iloc[top]["Semantic_ID"].tolist()

        lats.append((time.time()-t_q)*1000)
        r = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)

    print(f"  Queries filtered by country+variety: {hits_filtered}/{len(test)} "
          f"({hits_filtered/len(test)*100:.1f}%)")
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s


# ─── METHOD 9: Structured Filter + BM25 ──────────────────────────────────────
def run_structured_bm25(cat, test, k_values):
    """
    Structured Filter + in-group BM25.
    After filtering by country+variety, builds a per-group BM25 index
    and ranks by description match within the group.
    Faster than TF-IDF on large subsets; better recall on small subsets.
    """
    print(f"\n{'─'*60}")
    print("METHOD 9: Structured Filter + BM25 (within country+variety group)")

    # Pre-build full BM25 for fallback
    t0 = time.time()
    full_corpus = [d.lower().split() for d in cat["doc_text"]]
    bm25_full   = BM25Okapi(full_corpus, k1=1.2, b=0.65)
    print(f"  Full BM25 index: {time.time()-t0:.1f}s")

    cat["_cv"] = cat["Semantic_ID"].apply(
        lambda x: (x.split('-')[0], x.split('-')[2]) if len(x.split('-'))>2 else ("",""))

    K=max(k_values); records=[]; lats=[]
    for item in tqdm(test, desc="  Struct-BM25"):
        t_q = time.time()
        c_code, v_code, _ = extract_query_fields(item["instruction"])

        if c_code and v_code:
            mask   = cat["_cv"] == (c_code, v_code)
            subset = cat[mask].reset_index()
        else:
            subset = None

        if subset is not None and len(subset) >= K:
            sub_corpus = [d.lower().split() for d in subset["doc_text"]]
            sub_bm25   = BM25Okapi(sub_corpus, k1=1.2, b=0.65)
            sc         = sub_bm25.get_scores(item["instruction"].lower().split())
            top_idx    = np.argsort(sc)[::-1][:K]
            ret        = subset.iloc[top_idx]["Semantic_ID"].tolist()
        else:
            sc  = bm25_full.get_scores(item["instruction"].lower().split())
            top = np.argsort(sc)[::-1][:K]
            ret = cat.iloc[top]["Semantic_ID"].tolist()

        lats.append((time.time()-t_q)*1000)
        r = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)

    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s


# ─── METHOD 10: Structured Filter + Price Re-rank ────────────────────────────
def run_structured_price(cat, test, k_values):
    """
    Structured Filter + Price Proximity Re-rank.
    Within the country+variety filtered subset, rank wines by how close
    their price is to the budget hint in the query.
    Price is the strongest signal within a country+variety group.
    """
    print(f"\n{'─'*60}")
    print("METHOD 10: Structured Filter + Price Re-rank (budget proximity)")

    # Ensure price column exists
    if "price" not in cat.columns:
        print("  WARNING: 'price' column not found — falling back to Structured TF-IDF")
        return run_structured_tfidf(cat, test, k_values)

    cat["_price"] = pd.to_numeric(cat["price"], errors="coerce").fillna(cat["price"].median())
    cat["_cv"]   = cat["Semantic_ID"].apply(
        lambda x: (x.split('-')[0], x.split('-')[2]) if len(x.split('-'))>2 else ("",""))

    # Full TF-IDF for hybrid scoring and fallback
    t0  = time.time()
    vec = TfidfVectorizer(max_features=80_000, ngram_range=(1,2),
                          sublinear_tf=True, min_df=2, strip_accents="unicode")
    mat = vec.fit_transform(cat["doc_text"])
    print(f"  TF-IDF fallback index: {time.time()-t0:.1f}s")

    K=max(k_values); records=[]; lats=[]
    for item in tqdm(test, desc="  Struct-Price"):
        t_q = time.time()
        c_code, v_code, price = extract_query_fields(item["instruction"])

        if c_code and v_code:
            mask   = cat["_cv"] == (c_code, v_code)
            subset = cat[mask].reset_index(drop=True)
        else:
            subset = None

        if subset is not None and len(subset) >= K:
            if price is not None:
                # Primary sort: price proximity; secondary: TF-IDF similarity
                sub_mat  = vec.transform(subset["doc_text"])
                q_vec    = vec.transform([item["instruction"]])
                tfidf_sc = cosine_similarity(q_vec, sub_mat)[0]
                # Normalise price distance to [0,1]
                price_dist = np.abs(subset["_price"].values - price)
                max_dist   = price_dist.max() + 1e-9
                price_sc   = 1.0 - price_dist / max_dist
                # Combined score: 60% price proximity + 40% description
                combined = 0.60 * price_sc + 0.40 * tfidf_sc
            else:
                # No price hint → pure TF-IDF within subset
                sub_mat  = vec.transform(subset["doc_text"])
                q_vec    = vec.transform([item["instruction"]])
                combined = cosine_similarity(q_vec, sub_mat)[0]
            top_idx = np.argsort(combined)[::-1][:K]
            ret     = subset.iloc[top_idx]["Semantic_ID"].tolist()
        else:
            # Fallback
            q_vec = vec.transform([item["instruction"]])
            sims  = cosine_similarity(q_vec, mat)[0]
            top   = np.argsort(sims)[::-1][:K]
            ret   = cat.iloc[top]["Semantic_ID"].tolist()

        lats.append((time.time()-t_q)*1000)
        r = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)

    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s



# ─── METHOD 11: GNN-Enhanced Retrieval ───────────────────────────────────────
def run_gnn_retrieval(cat, test, k_values):
    print(f"\n{'─'*60}\nMETHOD 11: GNN-Enhanced Retrieval (LightGCN Embeddings)")
    embeddings_file = str(cfg.RESULTS / "gnn_wine_embeddings.npy")
    tfidf_file = str(cfg.RESULTS / "gnn_tfidf.pkl")
    svd_file = str(cfg.RESULTS / "gnn_svd.pkl")
    
    if not (os.path.exists(embeddings_file) and os.path.exists(tfidf_file) and os.path.exists(svd_file)):
        print(f"  ERROR: GNN files not found. Run gnn_indexer.py first.")
        return pd.DataFrame(), {}
    
    import pickle
    t0 = time.time()
    final_wine_embeddings = np.load(embeddings_file)
    with open(tfidf_file, "rb") as f:
        vec = pickle.load(f)
    with open(svd_file, "rb") as f:
        svd = pickle.load(f)
    print(f"  Loaded GNN embeddings and projection objects in {time.time()-t0:.2f}s")
    
    K = max(k_values); records=[]; lats=[]
    for item in tqdm(test, desc="  GNN-Retrieval"):
        t_q = time.time()
        # Project query
        q_vec = vec.transform([item["instruction"]])
        q_emb = svd.transform(q_vec)  # shape (1, 128)
        
        # Compute cosine similarity
        sims = cosine_similarity(q_emb, final_wine_embeddings)[0]
        top = np.argsort(sims)[::-1][:K]
        
        lats.append((time.time()-t_q)*1000)
        ret = cat.iloc[top]["Semantic_ID"].tolist()
        
        r = {"target_id": item["target_id"]}
        r.update(compute_metrics(ret, item["target_id"], k_values))
        records.append(r)
        
    df, s = aggregate(records, k_values)
    s["Latency_ms"] = float(np.mean(lats))
    return df, s



def main():
    print("="*60)
    print("  Wine Recommendation — Baseline Evaluation (11 methods)")
    print("="*60)
    if not os.path.exists(CATALOG_CSV):
        print(f"ERROR: {CATALOG_CSV} not found."); sys.exit(1)
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: {TEST_FILE} not found. Run data_prep.py first."); sys.exit(1)

    cat  = load_catalog(CATALOG_CSV)
    print(f"\nLoading test data...")
    test = load_test(TEST_FILE, EVAL_SIZE)

    summaries = {}

    # 1. BM25 (tuned)
    _, s = run_bm25(cat, test, K_VALUES)
    summaries["BM25"] = s

    # 2. BM25+ Enhanced
    _, s = run_bm25_enhanced(cat, test, K_VALUES)
    summaries["BM25+ Enhanced"] = s

    # 3. TF-IDF CF
    _, s = run_tfidf(cat, test, K_VALUES)
    summaries["TF-IDF CF"] = s

    # 4. LSA
    _, s = run_lsa(cat, test, K_VALUES)
    summaries["TF-IDF + LSA"] = s

    # 5. Hybrid (reuse BM25 + TF-IDF objects)
    print(f"\n{'─'*60}\nBuilding shared indexes for Hybrid method...")
    corpus_bm25 = [d.lower().split() for d in cat["doc_text"]]
    bm25_obj    = BM25Okapi(corpus_bm25, k1=1.2, b=0.65)
    vec2        = TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=2, strip_accents="unicode")
    mat2        = vec2.fit_transform(cat["doc_text"])
    _, s = run_hybrid(cat, test, K_VALUES, bm25_obj, vec2, mat2)
    summaries["Hybrid BM25+TF-IDF"] = s

    # 6. Popularity
    _, s = run_popularity(cat, test, K_VALUES)
    summaries["Popularity-Based"] = s

    # 7. Random
    _, s = run_random(cat, test, K_VALUES)
    summaries["Random Baseline"] = s

    # 8. Structured Filter + TF-IDF  (HIGH-RECALL)
    _, s = run_structured_tfidf(cat, test, K_VALUES)
    summaries["Struct-Filter TF-IDF"] = s

    # 9. Structured Filter + BM25  (HIGH-RECALL)
    _, s = run_structured_bm25(cat, test, K_VALUES)
    summaries["Struct-Filter BM25"] = s

    # 10. Structured Filter + Price Re-rank  (HIGH-RECALL)
    _, s = run_structured_price(cat, test, K_VALUES)
    summaries["Struct-Filter+Price"] = s

    # 11. GNN-Enhanced Retrieval (GNN-Filter)
    _, s = run_gnn_retrieval(cat, test, K_VALUES)
    summaries["GNN-Filter"] = s

    # Print & Save
    print_table(summaries, K_VALUES)
    summary_df = pd.DataFrame(summaries).T
    summary_df.index.name = "Method"
    summary_df.to_csv(OUTPUT_CSV)
    print(f"\n  ✅ Saved: {OUTPUT_CSV}")
    print("  Next: run base_rag_eval.py --update_baseline then merge_results.py")


if __name__ == "__main__":
    main()

