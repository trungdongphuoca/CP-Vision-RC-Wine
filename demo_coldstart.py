"""
demo_coldstart.py — Tuần 7: Cold-Start & Fallback Mechanism
============================================================
Minh họa cơ chế xử lý sản phẩm mới (cold-start) và fallback
khi không có embedding hoặc lịch sử tương tác.

Chiến lược:
  - Tier 1: Vector Search (ChromaDB embedding) — sản phẩm có dữ liệu
  - Tier 2: Content-Based BM25/TF-IDF — fallback khi không có embedding
  - Tier 3: Popularity-Based — cold-start hoàn toàn (sản phẩm mới nhất)

Usage:
    python3 demo_coldstart.py
    python3 demo_coldstart.py --query "Light white wine from France"
    python3 demo_coldstart.py --demo_new  # giả lập thêm sản phẩm mới
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

import argparse, re, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

class C:
    RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; YELLOW="\033[93m"
    CYAN="\033[96m"; RED="\033[91m"; DIM="\033[2m"; WHITE="\033[97m"; MAGENTA="\033[95m"

def hdr(t): print(f"\n{C.BOLD}{C.CYAN}{'─'*64}{C.RESET}\n{C.BOLD}  {t}{C.RESET}")
def ok(t):  print(f"  {C.GREEN}✓{C.RESET} {t}")
def warn(t):print(f"  {C.YELLOW}⚠{C.RESET}  {t}")
def info(t):print(f"  {C.CYAN}ℹ{C.RESET}  {t}")
def err(t): print(f"  {C.RED}✗{C.RESET} {t}")
def sep():  print(f"  {C.DIM}{'─'*60}{C.RESET}")

parser = argparse.ArgumentParser()
parser.add_argument("--query",    default="Bold red wine from France for dinner")
parser.add_argument("--demo_new", action="store_true",
                    help="Simulate adding 5 brand-new wines (no embeddings)")
parser.add_argument("--catalog_size", type=int, default=3000)
args = parser.parse_args()

# ─── Simulated new wines (cold-start items) ───────────────────────────────────
NEW_WINES = [
    {"title":"Château Nouveau Rouge 2024","country":"France","province":"Bordeaux",
     "variety":"Cabernet Sauvignon","price":45.0,"points":None,
     "description":"A brand new 2024 vintage. Deep ruby color, notes of black currant.",
     "has_embedding":False},
    {"title":"Vino Nuevo Tinto 2024","country":"Spain","province":"Rioja",
     "variety":"Tempranillo","price":22.0,"points":None,
     "description":"Fresh 2024 release. Cherry and plum notes with mild tannins.",
     "has_embedding":False},
    {"title":"Nuova Etichetta Bianco 2024","country":"Italy","province":"Tuscany",
     "variety":"Chardonnay","price":18.0,"points":None,
     "description":"New-to-catalog 2024 white. Crisp citrus and floral notes.",
     "has_embedding":False},
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_catalog(path, size):
    df = pd.read_csv(path).dropna(subset=["country","variety","description","title"])
    df["has_embedding"] = True   # existing wines have embeddings
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

def keyword_match_score(query, wine_dict):
    """Simple keyword-based scoring for cold-start wines."""
    q = query.lower()
    d = str(wine_dict.get("description","")).lower()
    v = str(wine_dict.get("variety","")).lower()
    c = str(wine_dict.get("country","")).lower()
    score = 0.0
    for word in q.split():
        if len(word) < 3: continue
        if word in v: score += 0.4
        if word in c: score += 0.3
        if word in d: score += 0.1
    # price check
    m = re.search(r'\$\s*(\d+)', q)
    try: wp = float(wine_dict.get("price", 0) or 0)
    except: wp = 0
    if m and wp:
        qp = float(m.group(1))
        if wp <= qp: score += 0.2
    return round(score, 3)

def tfidf_score(query, df, k=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    v   = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), sublinear_tf=True)
    mat = v.fit_transform(df["doc_text"])
    q   = v.transform([query])
    sc  = cosine_similarity(q, mat)[0]
    idx = np.argsort(sc)[::-1][:k]
    return df.iloc[idx], sc[idx]

def print_wine(wine_dict, score, tier_label, is_cold=False):
    cold_tag = f" {C.YELLOW}[NEW — Cold Start]{C.RESET}" if is_cold else ""
    print(f"\n     {C.WHITE}{str(wine_dict.get('title','?'))[:60]}{C.RESET}{cold_tag}")
    print(f"       Variety  : {wine_dict.get('variety','?')}  |  {wine_dict.get('country','?')}")
    price = wine_dict.get("price")
    price_s = f"${price:.0f}" if price else "N/A"
    print(f"       Price    : {C.GREEN}{price_s}{C.RESET}")
    print(f"       Score    : {C.MAGENTA}{score:.4f}{C.RESET}  [{tier_label}]")
    print(f"       Notes    : {C.DIM}{str(wine_dict.get('description',''))[:100]}...{C.RESET}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{C.BOLD}{'='*64}{C.RESET}")
    print(f"{C.BOLD}  🍷  Cold-Start & Fallback Demo  (Tuần 7){C.RESET}")
    print(f"{'='*64}")
    print(f"  Chiến lược 3 tầng (Tiered Fallback):")
    print(f"  {C.GREEN}Tier 1{C.RESET}: Vector Embedding Search  (có embedding → chính xác nhất)")
    print(f"  {C.YELLOW}Tier 2{C.RESET}: TF-IDF Content-Based CF (fallback khi thiếu embedding)")
    print(f"  {C.RED}Tier 3{C.RESET}: Keyword Matching         (cold-start sản phẩm mới nhất)")

    info(f"Loading catalog ({args.catalog_size:,} wines)...")
    df = load_catalog(str(cfg.WINE_CSV), args.catalog_size)
    ok(f"Loaded {len(df):,} existing wines (all have embeddings)")

    # Add new cold-start wines if demo_new flag
    new_wines_added = []
    if args.demo_new:
        hdr("Giả lập thêm sản phẩm MỚI (cold-start items)")
        for w in NEW_WINES:
            print(f"  + Adding: {w['title']}  [{w['country']}  {w['variety']}  ${w['price']}]")
            new_wines_added.append(w)
        warn(f"{len(new_wines_added)} sản phẩm mới CHƯA có embedding!")
        info("→ Sẽ dùng Keyword Matching (Tier 3) cho sản phẩm này")

    query = args.query
    print(f"\n{C.BOLD}  Query: \"{query}\"{C.RESET}")

    # ── Tier 1: Simulated Vector Search (existing wines with embedding) ────────
    hdr("TIER 1 — Vector Embedding Search (ChromaDB)")
    info("Existing wines with embeddings → TF-IDF as proxy for demo")
    t0 = time.time()
    rows_t1, scores_t1 = tfidf_score(query, df, k=3)
    lat1 = (time.time()-t0)*1000
    ok(f"Tier 1 results ({lat1:.1f}ms)  ← sản phẩm có embedding:")
    for i, (_, row) in enumerate(rows_t1.iterrows()):
        print_wine(row.to_dict(), scores_t1[i], "Tier1 Vector Search", is_cold=False)

    # ── Tier 2: Content-Based Fallback ────────────────────────────────────────
    hdr("TIER 2 — TF-IDF Content-Based Fallback")
    info("Dùng khi một subset wine thiếu embedding (ví dụ: wine chưa có review)")
    df_no_emb = df.sample(min(200, len(df)), random_state=99).copy()
    df_no_emb["has_embedding"] = False   # simulate missing embeddings
    t0 = time.time()
    rows_t2, scores_t2 = tfidf_score(query, df_no_emb, k=3)
    lat2 = (time.time()-t0)*1000
    ok(f"Tier 2 fallback results ({lat2:.1f}ms)  ← content-based CF:")
    for i, (_, row) in enumerate(rows_t2.iterrows()):
        print_wine(row.to_dict(), scores_t2[i], "Tier2 TF-IDF Fallback", is_cold=False)

    # ── Tier 3: Cold-Start New Products ───────────────────────────────────────
    hdr("TIER 3 — Keyword Matching (Cold-Start sản phẩm mới)")
    if not new_wines_added:
        warn("Chạy với --demo_new để xem cold-start demo cho sản phẩm mới nhất")
        # Still demo with a simulated new wine
        new_wines_added = [NEW_WINES[0]]
        info("Dùng 1 sản phẩm mẫu để minh họa...")

    t0 = time.time()
    cold_scored = []
    for w in new_wines_added:
        w["doc_text"] = f"{w['variety']} {w['country']} {w.get('province','')} {w['description']}"
        sc = keyword_match_score(query, w)
        cold_scored.append((w, sc))
    cold_scored.sort(key=lambda x: x[1], reverse=True)
    lat3 = (time.time()-t0)*1000
    ok(f"Tier 3 cold-start results ({lat3:.3f}ms)  ← sản phẩm MỚI:")
    for w, sc in cold_scored:
        print_wine(w, sc, "Tier3 Keyword Match", is_cold=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep()
    print(f"\n{C.BOLD}  TỔNG KẾT — Cold-Start & Fallback Strategy{C.RESET}")
    print(f"  {'Tier':<8} {'Method':<30} {'Latency':>10} {'Coverage'}")
    print(f"  {'─'*65}")
    print(f"  {C.GREEN}Tier 1{C.RESET}  {'Vector Embedding (ChromaDB)':<30} {lat1:>8.1f}ms  Wines with embedding")
    print(f"  {C.YELLOW}Tier 2{C.RESET}  {'TF-IDF Content-Based CF':<30} {lat2:>8.1f}ms  Wines without embedding")
    print(f"  {C.RED}Tier 3{C.RESET}  {'Keyword Matching (Regex)':<30} {lat3:>8.3f}ms  Brand-new cold-start wines")
    print(f"\n  {C.DIM}Tuần 7: Cơ chế Fallback đảm bảo recommendation cho mọi sản phẩm,")
    print(f"  kể cả sản phẩm hoàn toàn mới chưa có dữ liệu (Zero-Shot Cold-Start).{C.RESET}\n")

if __name__ == "__main__":
    main()
