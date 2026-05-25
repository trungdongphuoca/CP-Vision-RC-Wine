"""
demo_pipeline.py — Tuần 5: Demo Pipeline Cơ Bản
Truy xuất rượu vang từ câu truy vấn văn bản (không cần GPU).

Pipeline: Query → BM25/TF-IDF Semantic Search → XAI Scoring → Top-K Results

Usage:
    python3 demo_pipeline.py
    python3 demo_pipeline.py --query "Bold red Cabernet from Italy under $50"
    python3 demo_pipeline.py --batch        # 5 example queries
    python3 demo_pipeline.py --method bm25  # only BM25
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

import argparse, json, re, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# ─── Colors ───────────────────────────────────────────────────────────────────
class C:
    RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; YELLOW="\033[93m"
    CYAN="\033[96m"; MAGENTA="\033[95m"; DIM="\033[2m"; WHITE="\033[97m"

def hdr(t): print(f"\n{C.BOLD}{C.CYAN}{'─'*68}{C.RESET}\n{C.BOLD}  {t}{C.RESET}")
def ok(t):  print(f"  {C.GREEN}✓{C.RESET} {t}")
def info(t):print(f"  {C.CYAN}ℹ{C.RESET}  {t}")
def sep():  print(f"  {C.DIM}{'─'*64}{C.RESET}")

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--query",        default=None)
parser.add_argument("--top_k",        type=int, default=3)
parser.add_argument("--method",       choices=["bm25","tfidf","both"], default="both")
parser.add_argument("--catalog_size", type=int, default=5000)
parser.add_argument("--no_xai",       action="store_true")
parser.add_argument("--batch",        action="store_true")
parser.add_argument("--out",          default=None)
args = parser.parse_args()

BATCH_QUERIES = [
    "Bold red Cabernet Sauvignon from Napa Valley under $60 for steak dinner",
    "Light crisp white wine from France around $25 with seafood",
    "Sweet dessert wine from Italy for cheese pairing",
    "Affordable Malbec from Argentina under $20",
    "Sparkling Champagne from France for celebration",
]

# ─── Catalog ──────────────────────────────────────────────────────────────────
def load_catalog(path, size):
    df = pd.read_csv(path).dropna(subset=["country","variety","description","title"])
    def year(t):
        m = re.search(r"(19|20)\d{2}", str(t))
        return m.group(0) if m else "NV"
    def clean(s, n=4):
        return re.sub(r"[^A-Za-z0-9]","",str(s)).upper()[:n]
    df["vintage"] = df["title"].apply(year)
    df["Semantic_ID"] = df.apply(
        lambda r: f"{clean(r['country'])}-{clean(r.get('province',''))}-{clean(r['variety'])}-{r['vintage']}",
        axis=1)
    df["doc_text"] = df.apply(
        lambda r: f"{r['variety']} {r['country']} {r.get('province','') or ''} {r['description']}",
        axis=1)
    return df.head(size).reset_index(drop=True)

# ─── Indices ──────────────────────────────────────────────────────────────────
def build_bm25(df):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([d.lower().split() for d in df["doc_text"]])

def build_tfidf(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=30_000, ngram_range=(1,2), sublinear_tf=True)
    m = v.fit_transform(df["doc_text"])
    return v, m

# ─── Retrieval ─────────────────────────────────────────────────────────────────
def ret_bm25(bm25, df, q, k):
    sc = bm25.get_scores(q.lower().split())
    idx = np.argsort(sc)[::-1][:k]
    return df.iloc[idx], sc[idx]

def ret_tfidf(vec, mat, df, q, k):
    from sklearn.metrics.pairwise import cosine_similarity
    sc = cosine_similarity(vec.transform([q]), mat)[0]
    idx = np.argsort(sc)[::-1][:k]
    return df.iloc[idx], sc[idx]

# ─── XAI scoring ──────────────────────────────────────────────────────────────
FOOD_KW = ["steak","seafood","fish","cheese","chicken","pasta","dessert","lamb","beef"]
STYLE_KW= {"bold":["cabernet","malbec","syrah"], "light":["pinot","riesling"],
            "sweet":["moscato","port","dessert"], "sparkling":["champagne","prosecco","brut"]}

def xai_score(query, row):
    q = query.lower()
    # price
    m = re.search(r'\$\s*(\d+)', q)
    try:    wp = float(row.get("price", 0) or 0)
    except: wp = 0
    price_f = (min(float(m.group(1)),wp)/max(float(m.group(1)),wp,1)) if m and wp else 0.5
    # style
    v = str(row.get("variety","")).lower()
    direct = len(set(v.split()) & set(q.split())) / max(len(v.split()),1)
    style_f = min(direct,1.0) if direct else next(
        (0.8 for sty,grps in STYLE_KW.items() if sty in q and any(g in v for g in grps)), 0.1)
    # pairing
    foods = [k for k in FOOD_KW if k in q]
    desc = str(row.get("description","")).lower()
    pair_f = (sum(1 for f in foods if f in desc)/len(foods)) if foods else 0.5
    # region
    country = str(row.get("country","")).lower()
    reg_f = 1.0 if country and country in q else 0.0
    # weighted
    score = 0.30*price_f + 0.25*style_f + 0.20*pair_f + 0.15*reg_f
    return {"price_match":round(price_f,3),"style_match":round(style_f,3),
            "pairing_match":round(pair_f,3),"region_match":round(reg_f,3),
            "xai_score":round(score,4)}

# ─── Display ──────────────────────────────────────────────────────────────────
def print_wine(rank, row, score, xai=None, method=""):
    price = f"${row.get('price',0):.0f}" if pd.notna(row.get("price")) else "N/A"
    pts   = f"{row.get('points','?')} pts" if pd.notna(row.get("points")) else ""
    print(f"\n  {C.BOLD}{C.YELLOW}#{rank}{C.RESET}  {C.WHITE}{str(row.get('title','?'))[:65]}{C.RESET}")
    print(f"      Semantic ID : {C.CYAN}{row.get('Semantic_ID','?')}{C.RESET}")
    print(f"      Variety     : {row.get('variety','?')}  |  {row.get('country','?')}")
    print(f"      Price       : {C.GREEN}{price}{C.RESET}  {C.DIM}{pts}{C.RESET}")
    print(f"      Score [{method:6s}]: {C.MAGENTA}{score:.4f}{C.RESET}")
    desc = str(row.get("description",""))[:110]
    print(f"      Notes       : {C.DIM}{desc}...{C.RESET}")
    if xai:
        top = max(["price_match","style_match","pairing_match","region_match"],
                  key=lambda k: xai[k])
        bars = " | ".join(f"{k.split('_')[0]}={xai[k]:.2f}" for k in
                          ["price_match","style_match","pairing_match","region_match"])
        print(f"      XAI Score   : {C.GREEN}{xai['xai_score']:.3f}{C.RESET}  "
              f"Top factor: {C.MAGENTA}{top}{C.RESET}")
        print(f"      XAI Factors : {C.DIM}{bars}{C.RESET}")

# ─── Run one query ─────────────────────────────────────────────────────────────
def run_query(query, df, bm25=None, tv=None, tm=None, top_k=3, method="both", do_xai=True):
    hdr(f"QUERY: \"{query[:65]}\"")
    t0 = time.time(); out = {"query": query, "results": {}}

    if method in ("bm25","both") and bm25:
        sep(); print(f"  {C.BOLD}▶ Method: BM25 (Okapi BM25 keyword retrieval){C.RESET}")
        t1 = time.time(); rows, sc = ret_bm25(bm25, df, query, top_k)
        lat = (time.time()-t1)*1000; info(f"Latency: {lat:.1f}ms")
        wines = []
        for i,(_, row) in enumerate(rows.iterrows()):
            x = xai_score(query, row) if do_xai else None
            print_wine(i+1, row, sc[i], x, "BM25")
            wines.append({"rank":i+1,"title":row.get("title"),"semantic_id":row.get("Semantic_ID"),
                          "score":float(sc[i]),"xai":x})
        out["results"]["BM25"] = {"wines":wines,"latency_ms":lat}

    if method in ("tfidf","both") and tv:
        sep(); print(f"  {C.BOLD}▶ Method: TF-IDF Content-Based Filtering{C.RESET}")
        t1 = time.time(); rows, sc = ret_tfidf(tv, tm, df, query, top_k)
        lat = (time.time()-t1)*1000; info(f"Latency: {lat:.1f}ms")
        wines = []
        for i,(_, row) in enumerate(rows.iterrows()):
            x = xai_score(query, row) if do_xai else None
            print_wine(i+1, row, sc[i], x, "TF-IDF")
            wines.append({"rank":i+1,"title":row.get("title"),"semantic_id":row.get("Semantic_ID"),
                          "score":float(sc[i]),"xai":x})
        out["results"]["TF-IDF"] = {"wines":wines,"latency_ms":lat}

    out["total_latency_ms"] = round((time.time()-t0)*1000, 1)
    sep(); ok(f"Total pipeline latency: {out['total_latency_ms']}ms")
    return out

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{C.BOLD}{'='*68}{C.RESET}")
    print(f"{C.BOLD}  🍷  Wine Recommendation Demo Pipeline  (Tuần 3 + 5 + 6){C.RESET}")
    print(f"{'='*68}")
    print(f"  Tuần 3: Semantic Search Index (BM25 + TF-IDF)")
    print(f"  Tuần 5: Demo Pipeline — Truy xuất sản phẩm từ văn bản")
    print(f"  Tuần 6: Explainable XAI (feature attribution)")
    print(f"  Dataset: Kaggle Wine Reviews  |  {args.catalog_size:,} wines indexed")

    info(f"Loading catalog ({args.catalog_size:,} wines)...")
    t0 = time.time()
    df = load_catalog(str(cfg.WINE_CSV), args.catalog_size)
    ok(f"Loaded {len(df):,} wines  [{(time.time()-t0)*1000:.0f}ms]")

    bm25 = tv = tm = None
    if args.method in ("bm25","both"):
        info("Building BM25 index..."); t0=time.time()
        bm25 = build_bm25(df); ok(f"BM25 ready [{(time.time()-t0)*1000:.0f}ms]")
    if args.method in ("tfidf","both"):
        info("Building TF-IDF index..."); t0=time.time()
        tv, tm = build_tfidf(df); ok(f"TF-IDF ready  shape={tm.shape}  [{(time.time()-t0)*1000:.0f}ms]")

    queries  = BATCH_QUERIES if args.batch else ([args.query] if args.query else None)
    all_res  = []

    if queries is None:
        # Interactive
        print(f"\n{C.CYAN}  Interactive mode (type 'quit' to exit){C.RESET}")
        print(f"  Ví dụ: 'Bold red Cabernet from Italy under $50 for steak'\n")
        while True:
            try:
                q = input(f"{C.BOLD}  Query > {C.RESET}").strip()
                if not q or q.lower() in ("quit","q","exit"): break
                all_res.append(run_query(q, df, bm25, tv, tm, args.top_k, args.method, not args.no_xai))
            except (KeyboardInterrupt, EOFError):
                break
    else:
        for q in queries:
            all_res.append(run_query(q, df, bm25, tv, tm, args.top_k, args.method, not args.no_xai))

    # Summary
    print(f"\n{C.BOLD}{'='*68}")
    print(f"  TỔNG KẾT PIPELINE")
    print(f"{'='*68}{C.RESET}")
    print(f"  Queries processed : {len(all_res)}")
    print(f"  Catalog size      : {len(df):,} wines")
    print(f"  XAI enabled       : {'No' if args.no_xai else 'Yes'}")
    if all_res:
        avg = np.mean([r["total_latency_ms"] for r in all_res])
        print(f"  Avg total latency : {avg:.1f}ms/query")

    out_path = args.out or (str(cfg.RESULTS/"demo_pipeline_results.json") if args.batch else None)
    if out_path and all_res:
        os.makedirs(str(cfg.RESULTS), exist_ok=True)
        with open(out_path,"w") as f:
            json.dump(all_res, f, indent=2, default=str)
        ok(f"Results saved → {out_path}")

    print(f"\n{C.GREEN}{C.BOLD}  ✓ Demo hoàn thành!{C.RESET}")
    print(f"  Đánh giá đầy đủ : python3 evaluation/baseline_eval.py")
    print(f"  Biểu đồ kết quả : python3 evaluation/plot_results.py")
    print(f"  Toàn bộ pipeline : bash run_report.sh\n")

if __name__ == "__main__":
    main()
