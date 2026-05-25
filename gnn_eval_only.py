"""gnn_eval_only.py — Chỉ chạy GNN-Filter eval và patch vào baseline_comparison.csv"""
import sys, os, json, re, time
sys.path.insert(0, '.')
import config as cfg
import numpy as np
import pandas as pd
from tqdm import tqdm

print('Running GNN-Filter evaluation only (12,044 samples)...')

GNN_EMB = 'results/gnn_wine_embeddings.npy'
GNN_IDS = 'results/gnn_semantic_ids.csv'

gnn_emb = np.load(GNN_EMB)
gnn_emb = gnn_emb / (np.linalg.norm(gnn_emb, axis=1, keepdims=True) + 1e-9)

catalog = pd.read_csv(str(cfg.WINE_CSV))
catalog = catalog.dropna(subset=['country','variety','description','title'])

def extract_year(t):
    m = re.search(r'(19|20)\d{2}', str(t))
    return m.group(0) if m else 'NV'
def clean_text(t):
    return re.sub(r'[^A-Za-z0-9]','',str(t)).upper()[:4]

catalog['vintage'] = catalog['title'].apply(extract_year)
catalog['Semantic_ID'] = catalog.apply(
    lambda r: f"{clean_text(r['country'])}-{clean_text(r.get('province',''))}-{clean_text(r['variety'])}-{r['vintage']}", axis=1)
catalog = catalog.reset_index(drop=True)

import pickle
from sklearn.preprocessing import normalize

tfidf = pickle.load(open('results/gnn_tfidf.pkl','rb'))
svd   = pickle.load(open('results/gnn_svd.pkl','rb'))

test = [json.loads(l) for l in open(str(cfg.TEST_JSONL))][:12044]

K = [1, 5, 10]
metrics = {f'Recall@{k}': [] for k in K}
metrics.update({f'IntentMatch@{k}': [] for k in K})
metrics['MRR'] = []
latencies = []

for q in tqdm(test, desc='GNN-Filter'):
    query_text = q.get('query','')
    target_id  = q.get('semantic_id', q.get('target_id',''))
    t0 = time.perf_counter()
    q_vec = normalize(svd.transform(tfidf.transform([query_text])), norm='l2')
    sims  = gnn_emb @ q_vec.T
    ranked_idx = np.argsort(-sims.flatten())
    ranked_ids = catalog.iloc[ranked_idx]['Semantic_ID'].tolist()
    latencies.append((time.perf_counter()-t0)*1000)
    for k in K:
        metrics[f'Recall@{k}'].append(1.0 if target_id in ranked_ids[:k] else 0.0)
        t_parts = target_id.split('-')
        matches = [r for r in ranked_ids[:k] if r.split('-')[:2]==t_parts[:2]]
        metrics[f'IntentMatch@{k}'].append(1.0 if matches else 0.0)
    for i, r in enumerate(ranked_ids[:10]):
        if r == target_id:
            metrics['MRR'].append(1.0/(i+1))
            break
    else:
        metrics['MRR'].append(0.0)

res = {k: np.mean(v) for k, v in metrics.items()}
res['Latency_ms'] = np.mean(latencies)

print('\nGNN-Filter Results (N=12,044):')
for k, v in res.items():
    print(f'  {k}: {v:.4f}')

# Patch into baseline_comparison.csv
df = pd.read_csv(str(cfg.BASELINE_CSV))
for col, val in res.items():
    if col in df.columns:
        df.loc[df['Method']=='GNN-Filter', col] = val
df.to_csv(str(cfg.BASELINE_CSV), index=False)
print('\nPatched GNN-Filter into baseline_comparison.csv')
print('Next: run merge_results.py and plot_results.py --dpi 300')
