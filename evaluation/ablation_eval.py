"""
ablation_eval.py
================
E4: Ablation Study
Compares different variants of the Hybrid Pipeline to show
contribution of each component.

Variants:
  A1 - LLM Greedy (baseline LLM alone)
  A2 - Cluster Filter only (random within cluster)
  A3 - Cluster + Price Rerank  (= full Hybrid, our best)
  A4 - Cluster + TF-IDF Rerank (flavor-based)
  A5 - Global Price Rerank (no cluster, just price proximity)
"""
import sys, os, json, math, re
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))
import config as cfg
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_price(inst):
    m = re.search(r'\$\s*(\d+)', inst)
    return float(m.group(1)) if m else None

def recall_at_k(target, ranked, k):
    return 1.0 if target in ranked[:k] else 0.0

def ndcg_at_10(target, ranked):
    for rank, pid in enumerate(ranked[:10]):
        if pid == target:
            return 1.0 / math.log2(rank + 2)
    return 0.0

def mrr(target, ranked):
    for rank, pid in enumerate(ranked[:10]):
        if pid == target:
            return 1.0 / (rank + 1)
    return 0.0

def evaluate_variant(name, get_ranked_fn, test, cat):
    records = []
    for i, item in enumerate(test):
        target_id = item['target_id']
        ranked = get_ranked_fn(i, item, cat)
        records.append({
            'r1':    recall_at_k(target_id, ranked, 1),
            'r5':    recall_at_k(target_id, ranked, 5),
            'r10':   recall_at_k(target_id, ranked, 10),
            'ndcg10':ndcg_at_10(target_id, ranked),
            'mrr':   mrr(target_id, ranked),
        })
    df = pd.DataFrame(records)
    return {
        "Method":     name,
        "Recall@1":   df['r1'].mean()   * 100,
        "Recall@5":   df['r5'].mean()   * 100,
        "Recall@10":  df['r10'].mean()  * 100,
        "NDCG@10":    df['ndcg10'].mean()* 100,
        "MRR":        df['mrr'].mean()  * 100,
    }

def main():
    print("="*60)
    print("  E4: Ablation Study")
    print("="*60)

    cat = pd.read_csv(cfg.WINE_SEMANTIC_CSV)
    cat['_price'] = pd.to_numeric(cat['price'], errors='coerce').fillna(cat['price'].median())
    cat['_text']  = (cat.get('variety','').fillna('') + ' ' +
                     cat.get('description','').fillna('') + ' ' +
                     cat.get('country','').fillna('')).str.strip()

    df_pred = pd.read_csv(cfg.RESULTS / "constrained_eval_results.csv").iloc[:500]

    with open(cfg.TEST_JSONL, encoding='utf-8') as f:
        test = [json.loads(line) for line in f][:500]

    # Pre-build TF-IDF index
    print("Building TF-IDF index...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(cat['_text'])

    def get_pred_cluster(i):
        pred_id = str(df_pred.iloc[i]['pred_id'])
        parts = pred_id.split('-')
        if pred_id == 'INVALID_ID' or len(parts) < 3:
            return ''
        return '-'.join(parts[:3])

    # A1: LLM Greedy (single prediction)
    def a1_llm_greedy(i, item, cat):
        pred_id = str(df_pred.iloc[i]['pred_id'])
        return [pred_id] if pred_id != 'INVALID_ID' else []

    # A2: Cluster Filter + Random
    def a2_cluster_random(i, item, cat):
        cluster = get_pred_cluster(i)
        if not cluster:
            return cat['Semantic_ID'].sample(10, random_state=i).tolist()
        subset = cat[cat['Semantic_ID'].str.startswith(cluster)]
        if len(subset) == 0:
            return cat['Semantic_ID'].sample(10, random_state=i).tolist()
        return subset['Semantic_ID'].sample(min(10, len(subset)), random_state=i).tolist()

    # A3: Cluster + Price Rerank (our best)
    def a3_cluster_price(i, item, cat):
        cluster = get_pred_cluster(i)
        req_price = extract_price(item['instruction'])
        subset = cat[cat['Semantic_ID'].str.startswith(cluster)].copy() if cluster else cat.copy()
        if len(subset) == 0:
            subset = cat.copy()
        if req_price is not None:
            subset['price_diff'] = np.abs(subset['_price'] - req_price)
            subset = subset.sort_values('price_diff')
        else:
            subset = subset.sample(frac=1, random_state=i)
        return subset['Semantic_ID'].tolist()[:10]

    # A4: Cluster + TF-IDF Rerank
    def a4_cluster_tfidf(i, item, cat):
        cluster = get_pred_cluster(i)
        query = item['instruction']
        subset = cat[cat['Semantic_ID'].str.startswith(cluster)].copy() if cluster else cat.copy()
        if len(subset) == 0:
            subset = cat.copy()
        idx = subset.index.tolist()
        if not idx:
            return []
        q_vec = tfidf.transform([query])
        sims  = cosine_similarity(q_vec, tfidf_matrix[idx]).flatten()
        order = np.argsort(-sims)
        ranked_ids = [subset.iloc[j]['Semantic_ID'] for j in order[:10]]
        return ranked_ids

    # A5: Global Price Rerank (no cluster)
    def a5_global_price(i, item, cat):
        req_price = extract_price(item['instruction'])
        sub = cat.copy()
        if req_price is not None:
            sub['price_diff'] = np.abs(sub['_price'] - req_price)
            sub = sub.sort_values('price_diff')
        else:
            sub = sub.sample(frac=1, random_state=i)
        return sub['Semantic_ID'].tolist()[:10]

    variants = [
        ("A1: LLM Greedy (no rerank)",       a1_llm_greedy),
        ("A2: Cluster Filter + Random",       a2_cluster_random),
        ("A3: Cluster + Price Rerank (Ours)", a3_cluster_price),
        ("A4: Cluster + TF-IDF Rerank",       a4_cluster_tfidf),
        ("A5: Global Price Rerank",           a5_global_price),
    ]

    results = []
    for name, fn in variants:
        print(f"  Evaluating: {name}...")
        r = evaluate_variant(name, fn, test, cat)
        results.append(r)
        print(f"    Recall@1={r['Recall@1']:.2f}%  Recall@10={r['Recall@10']:.2f}%  NDCG@10={r['NDCG@10']:.2f}%")

    df_out = pd.DataFrame(results)
    out_path = cfg.RESULTS / "ablation_results.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n=== ABLATION STUDY RESULTS (N=500) ===")
    print(df_out.to_string(index=False))
    print(f"\nSaved: {out_path}")
    return df_out

if __name__ == "__main__":
    main()
