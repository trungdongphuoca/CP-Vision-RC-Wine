"""
evaluation/statistical_analysis.py
====================================
Comprehensive statistical analysis for CW-EASE^R paper:
  1. Wilcoxon signed-rank test (per-user Recall@10)
  2. Lambda sensitivity: λ vs Recall@K curves (multiple schemes)
  3. Cold-start analysis: cold (<5 pos) vs warm (≥5 pos) users
  4. Confidence scheme analysis: why soft_binary > linear (signal stats)

Outputs:
  results/stat_test_results.csv
  results/lambda_sensitivity.csv
  results/coldstart_analysis.csv
  results/confidence_scheme_stats.csv
  results/figures/lambda_sensitivity.png
  results/figures/coldstart_comparison.png
  results/figures/scheme_comparison.png
  results/figures/confidence_distribution.png
"""

import os, sys, time, ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.linalg import inv
from scipy.stats import wilcoxon
from sklearn.preprocessing import MultiLabelBinarizer, normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def ease_B(Xc, lam):
    """Closed-form EASE^R weight matrix. Returns float32 to save memory.
    Key: compute Gram matrix (n_items × n_items) in float32, then cast to
    float64 only the small result before inversion — never casts the large
    (n_items × n_users) matrix to float64.
    """
    G = (Xc.T @ Xc).astype(np.float64) + lam * np.eye(Xc.shape[1])
    P = inv(G)
    B = np.eye(Xc.shape[1]) - P / np.diag(P)
    np.fill_diagonal(B, 0.)
    return B.astype(np.float32)



MAX_MK = 150
LOG_FACTORS = [1.0 / np.log2(i + 2) for i in range(MAX_MK + 2)]
SUM_LOG_FACTORS = [0.0] + list(np.cumsum(LOG_FACTORS))

def dcg(r, k):
    r = np.asarray(r, float)[:k]
    return sum(v / np.log2(i + 2) for i, v in enumerate(r)) if len(r) else 0.

def ndcg(r, k):
    d = dcg(r, k); i = dcg(sorted(r, reverse=True), k)
    return d / i if i > 0 else 0.

def per_user_metrics_batched(X, B, gt, mask, k=10, batch=2000):
    """
    Memory-efficient per-user (Recall@k, nDCG@k).
    Scores computed batch-by-batch; full score matrix never materialised.
    X: (n_users, n_items) float32
    B: (n_items, n_items) float32
    """
    user_ids = np.array(sorted(gt.keys()), dtype=np.int32)
    out = {}
    assert k <= MAX_MK, f"k={k} exceeds MAX_MK={MAX_MK}"
    for start in range(0, len(user_ids), batch):
        uids = user_ids[start:start + batch]
        S_b  = X[uids].astype(np.float32) @ B     # (batch, n_items)
        S_b[mask[uids] != 0] = -1e9
        
        # Vectorized top-k
        top_idx = np.argpartition(-S_b, k, axis=1)[:, :k]
        S_b_top = np.take_along_axis(S_b, top_idx, axis=1)
        sort_sort = np.argsort(-S_b_top, axis=1)
        top_sorted = np.take_along_axis(top_idx, sort_sort, axis=1)
        
        for j, uid in enumerate(uids):
            rel = gt[uid]
            top = top_sorted[j]
            n_hits = 0
            dcg_val = 0.0
            for idx, x in enumerate(top):
                if x in rel:
                    n_hits += 1
                    dcg_val += LOG_FACTORS[idx]
            rec = n_hits / len(rel)
            ndcg_val = dcg_val / SUM_LOG_FACTORS[n_hits] if n_hits > 0 else 0.0
            out[uid] = (rec, ndcg_val)
        del S_b
    return out

def evaluate_scores_batched(X, B, gt, mask, K_VALUES=(5,10,50,100), batch=2000):
    """
    Full ranking metrics (Precision, Recall, F1, nDCG) computed in batches.
    Never materialises the full score matrix.
    """
    met = {k: {'P':0.,'R':0.,'F1':0.,'nDCG':0.} for k in K_VALUES}
    MK  = max(K_VALUES)
    user_ids = np.array(sorted(gt.keys()), dtype=np.int32)
    n = len(user_ids)
    assert MK <= MAX_MK, f"MK={MK} exceeds MAX_MK={MAX_MK}"
    
    for start in range(0, n, batch):
        uids = user_ids[start:start + batch]
        S_b  = X[uids].astype(np.float32) @ B
        S_b[mask[uids] != 0] = -1e9
        
        # Vectorized top-K
        top_idx = np.argpartition(-S_b, MK, axis=1)[:, :MK]
        S_b_top = np.take_along_axis(S_b, top_idx, axis=1)
        sort_sort = np.argsort(-S_b_top, axis=1)
        top_sorted = np.take_along_axis(top_idx, sort_sort, axis=1)
        
        for j, uid in enumerate(uids):
            rel = gt[uid]
            nr = len(rel)
            top = top_sorted[j]
            
            h_cumsum = [0] * (MK + 1)
            dcg_cumsum = [0.0] * (MK + 1)
            s_val = 0
            d_val = 0.0
            for idx, x in enumerate(top):
                val = 1 if x in rel else 0
                s_val += val
                h_cumsum[idx+1] = s_val
                if val:
                    d_val += LOG_FACTORS[idx]
                dcg_cumsum[idx+1] = d_val
                
            for k in K_VALUES:
                n_hits = h_cumsum[k]
                p  = n_hits / k
                r  = n_hits / nr
                f1 = 2*p*r/(p+r) if p+r > 0 else 0.
                ndcg_val = dcg_cumsum[k] / SUM_LOG_FACTORS[n_hits] if n_hits > 0 else 0.
                met[k]['P'] += p
                met[k]['R'] += r
                met[k]['F1'] += f1
                met[k]['nDCG'] += ndcg_val
        del S_b
    for k in K_VALUES:
        for m in met[k]: met[k][m] /= n
    return met, n

# Keep legacy signature for callers that pass S directly (lambda sensitivity uses small matrices)
def evaluate_scores(S, gt, mask, K_VALUES=(5,10,50,100)):
    met = {k: {'P':0.,'R':0.,'F1':0.,'nDCG':0.} for k in K_VALUES}
    MK  = max(K_VALUES); n = 0
    for uid, rel in gt.items():
        s  = S[uid].copy(); s[mask[uid]] = -1e9
        top = np.argsort(-s)[:MK].tolist(); nr = len(rel)
        for k in K_VALUES:
            h  = [1 if x in rel else 0 for x in top[:k]]
            p  = sum(h)/k; r = sum(h)/nr
            f1 = 2*p*r/(p+r) if p+r > 0 else 0.
            met[k]['P'] += p; met[k]['R'] += r
            met[k]['F1'] += f1; met[k]['nDCG'] += ndcg(h, k)
        n += 1
    for k in K_VALUES:
        for m in met[k]: met[k][m] /= n
    return met, n

def macro_avg_batched(X, B, gt, mask, K_VALUES=(5,10,50,100)):
    return evaluate_scores_batched(X, B, gt, mask, K_VALUES)



def build_Xsb(df_tr, n_u, n_i):
    """soft_binary confidence matrix — vectorised."""
    rat = df_tr['Rating'].values.astype(np.float32)
    uid = df_tr['uid'].values.astype(np.int32)
    iid = df_tr['iid'].values.astype(np.int32)
    c   = np.where(rat >= 4.0, 1.0,
           np.where(rat >= 2.5, (rat - 2.5) * 0.2, -0.5)).astype(np.float32)
    nonzero = c != 0
    X = np.zeros((n_u, n_i), dtype=np.float32)
    X[uid[nonzero], iid[nonzero]] = c[nonzero]
    return X

def build_Xlin(df_tr, n_u, n_i):
    """linear confidence matrix: c = r - 3.5 — vectorised."""
    rat = df_tr['Rating'].values.astype(np.float32)
    uid = df_tr['uid'].values.astype(np.int32)
    iid = df_tr['iid'].values.astype(np.int32)
    X   = np.zeros((n_u, n_i), dtype=np.float32)
    X[uid, iid] = (rat - 3.5)
    return X

def build_Xbin(df_pos, n_u, n_i):
    """standard binary matrix — vectorised."""
    uid = df_pos['uid'].values.astype(np.int32)
    iid = df_pos['iid'].values.astype(np.int32)
    X   = np.zeros((n_u, n_i), dtype=np.float32)
    X[uid, iid] = 1.
    return X

# ─── Data loading (shared) ────────────────────────────────────────────────────

def load_data():
    """
    Load X-Wines dataset with automatic item/user filtering for memory budget.
    Full 21M dataset: 1M users × 100K items = 396 GB dense → must filter.
    Strategy: keep items with ≥ MIN_ITEM_RATINGS and users with ≥ MIN_USER_RATINGS.
    Auto-tune MIN_ITEM_RATINGS so matrix fits within MAX_MATRIX_GB.
    """
    full_path = os.path.join(ROOT, "data", "xwines", "XWines_Full_21M_ratings.csv")
    slim_path = os.path.join(ROOT, "data", "xwines", "XWines_Slim_150K_ratings.csv")
    MAX_MATRIX_GB   = 20.0   # target RAM budget for user×item float32 matrix
    MIN_USER_RATINGS = 5

    if os.path.exists(full_path):
        print(f"  Using FULL dataset: {full_path}")
        print("  Reading 21M ratings in chunks (1M rows) ...")
        chunks = []
        for i, chunk in enumerate(pd.read_csv(
                full_path, low_memory=False, chunksize=1_000_000,
                usecols=['UserID', 'WineID', 'Rating'])):
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                print(f"    {(i+1)*1_000_000:,} rows ...", flush=True)
        df_raw = pd.concat(chunks, ignore_index=True)
        print(f"  Loaded {len(df_raw):,} ratings.")

        # Auto-filter to fit in memory
        item_counts = df_raw['WineID'].value_counts()
        for min_ir in [50, 100, 200, 500, 1000, 2000, 5000]:
            pop_items  = set(item_counts[item_counts >= min_ir].index)
            df_f       = df_raw[df_raw['WineID'].isin(pop_items)]
            user_cnts  = df_f['UserID'].value_counts()
            act_users  = set(user_cnts[user_cnts >= MIN_USER_RATINGS].index)
            df_f       = df_f[df_f['UserID'].isin(act_users)]
            n_u_f      = df_f['UserID'].nunique()
            n_i_f      = df_f['WineID'].nunique()
            mem_gb     = n_u_f * n_i_f * 4 / 1e9
            print(f"  min_item={min_ir:5d}: {n_u_f:,} users × {n_i_f:,} items "
                  f"= {mem_gb:.1f} GB", flush=True)
            if mem_gb <= MAX_MATRIX_GB:
                df = df_f.copy()
                print(f"  → Selected: min_item_ratings={min_ir}, "
                      f"{n_u_f:,} users, {n_i_f:,} items, {mem_gb:.1f} GB")
                break
        else:
            raise RuntimeError("Cannot fit matrix in memory even at min_item=5000")
    else:
        print(f"  Using slim dataset: {slim_path}")
        df = pd.read_csv(slim_path, low_memory=False,
                         usecols=['UserID', 'WineID', 'Rating'])
        print(f"  Loaded {len(df):,} ratings.")

    THR = 4.0
    users  = sorted(df['UserID'].unique()); items = sorted(df['WineID'].unique())
    u2i    = {u: i for i, u in enumerate(users)}
    it2i   = {w: i for i, w in enumerate(items)}
    n_u, n_i = len(users), len(items)
    df['uid'] = df['UserID'].map(u2i)
    df['iid'] = df['WineID'].map(it2i)

    # Vectorised split for 80/20 train/test
    np.random.seed(42)
    df['rand'] = np.random.rand(len(df))
    # Sort by uid and rand so indices are randomly shuffled within each user group
    df_sorted = df.sort_values(['uid', 'rand']).copy()
    # Rank within each user group
    df_sorted['rank'] = df_sorted.groupby('uid').cumcount() + 1
    # User group size mapping
    sizes = df_sorted.groupby('uid').size().to_dict()
    df_sorted['user_size'] = df_sorted['uid'].map(sizes)
    df_sorted['split_thresh'] = np.maximum(1, (0.8 * df_sorted['user_size']).astype(np.int32))

    is_train = df_sorted['rank'] <= df_sorted['split_thresh']
    tr_df = df_sorted[is_train].copy()
    te_df = df_sorted[~is_train].copy()

    # Drop temporary columns
    for d in [df, tr_df, te_df]:
        if 'rand' in d.columns: d.drop(columns=['rand'], inplace=True)
        if 'rank' in d.columns: d.drop(columns=['rank'], inplace=True)
        if 'user_size' in d.columns: d.drop(columns=['user_size'], inplace=True)
        if 'split_thresh' in d.columns: d.drop(columns=['split_thresh'], inplace=True)

    tr_pos = tr_df[tr_df['Rating'] >= THR]
    te_pos = te_df[te_df['Rating'] >= THR]
    gt = {uid: s for uid, s in te_pos.groupby('uid')['iid'].apply(set).items() if len(s) > 0}
    return df, tr_df, te_df, tr_pos, te_pos, gt, n_u, n_i, u2i, it2i

# ─── 1. Wilcoxon Signed-Rank Test ─────────────────────────────────────────────

def wilcoxon_analysis(tr_df, tr_pos, gt, n_u, n_i):
    print("\n" + "="*60)
    print("  [1] WILCOXON SIGNED-RANK TEST")
    print("="*60)

    import gc

    print("  [Wilcoxon] Evaluating EASE^R (baseline) ...", flush=True)
    Xbin = build_Xbin(tr_pos, n_u, n_i)
    B_base = ease_B(Xbin, 500)
    base_pu = per_user_metrics_batched(Xbin, B_base, gt, Xbin, k=10)
    
    # Delete baseline models to free memory
    del Xbin, B_base
    gc.collect()

    print("  [Wilcoxon] Evaluating CW-EASE^R+IPS (proposed) ...", flush=True)
    Xsb  = build_Xsb(tr_df, n_u, n_i)
    freq = np.bincount(np.where(Xsb > 0)[1], minlength=n_i).astype(np.float32)
    p_i  = np.clip((freq / (freq.max()+1e-9)) ** 0.3, 1e-3, 1.)
    Xips = Xsb / np.sqrt(p_i)[np.newaxis, :]
    B_prop = ease_B(Xips, 750)
    del Xips
    gc.collect()

    prop_pu = per_user_metrics_batched(Xsb,  B_prop, gt, Xsb,  k=10)
    del Xsb, B_prop
    gc.collect()

    common  = sorted(set(base_pu) & set(prop_pu))
    r_base  = np.array([base_pu[u][0] for u in common])
    r_prop  = np.array([prop_pu[u][0] for u in common])
    nd_base = np.array([base_pu[u][1] for u in common])
    nd_prop = np.array([prop_pu[u][1] for u in common])

    stat_r, p_r   = wilcoxon(r_prop,  r_base,  alternative='greater', zero_method='wilcox')
    stat_nd, p_nd = wilcoxon(nd_prop, nd_base, alternative='greater', zero_method='wilcox')

    diff_r  = r_prop  - r_base
    diff_nd = nd_prop - nd_base

    print(f"  Users in test: {len(common):,}")
    print(f"  Mean Recall@10:  BASE={r_base.mean()*100:.4f}%  PROP={r_prop.mean()*100:.4f}%")
    print(f"  Mean nDCG@10:    BASE={nd_base.mean()*100:.4f}%  PROP={nd_prop.mean()*100:.4f}%")
    print(f"\n  Wilcoxon Recall@10:  statistic={stat_r:.1f}  p-value={p_r:.6f}")
    print(f"  Wilcoxon nDCG@10:    statistic={stat_nd:.1f}  p-value={p_nd:.6f}")
    print(f"  Users improved (R@10): {(diff_r>0).sum():,} / {len(common):,} "
          f"({(diff_r>0).mean()*100:.1f}%)")
    print(f"  Users hurt    (R@10): {(diff_r<0).sum():,} / {len(common):,} "
          f"({(diff_r<0).mean()*100:.1f}%)")
    sig_r  = "*** (p<0.001)" if p_r  < 0.001 else ("** (p<0.01)" if p_r  < 0.01 else
              ("* (p<0.05)"  if p_r  < 0.05  else "n.s."))
    sig_nd = "*** (p<0.001)" if p_nd < 0.001 else ("** (p<0.01)" if p_nd < 0.01 else
              ("* (p<0.05)"  if p_nd < 0.05  else "n.s."))
    print(f"  Significance Recall@10: {sig_r}")
    print(f"  Significance nDCG@10:   {sig_nd}")

    results = pd.DataFrame({
        "Metric":              ["Recall@10", "nDCG@10"],
        "Baseline_mean":       [r_base.mean(), nd_base.mean()],
        "Proposed_mean":       [r_prop.mean(), nd_prop.mean()],
        "Delta_mean":          [(r_prop-r_base).mean(), (nd_prop-nd_base).mean()],
        "Wilcoxon_statistic":  [stat_r, stat_nd],
        "p_value":             [p_r, p_nd],
        "Significance":        [sig_r, sig_nd],
        "Users_improved":      [(diff_r>0).sum(), (diff_nd>0).sum()],
        "Users_hurt":          [(diff_r<0).sum(), (diff_nd<0).sum()],
        "N_users":             [len(common), len(common)],
    })
    return results, diff_r, diff_nd, common


# ─── 2. Lambda Sensitivity ────────────────────────────────────────────────────

def lambda_sensitivity(tr_df, tr_pos, gt, n_u, n_i):
    print("\n" + "="*60)
    print("  [2] LAMBDA SENSITIVITY ANALYSIS")
    print("="*60)

    import gc
    LAMBDAS = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
    records = []

    # 1. binary (EASE^R)
    print("  Evaluating binary (EASE^R) scheme ...", flush=True)
    Xbin = build_Xbin(tr_pos, n_u, n_i)
    for lam in LAMBDAS:
        B = ease_B(Xbin, lam)
        m, _ = evaluate_scores_batched(Xbin, B, gt, Xbin, (5,10,50,100))
        del B
        r10 = m[10]["R"]; nd10 = m[10]["nDCG"]
        records.append({"Scheme": "binary (EASE^R)", "lambda": lam,
                         "Recall@5":   m[5]["R"],
                         "Recall@10":  r10,
                         "nDCG@10":    nd10,
                         "Recall@100": m[100]["R"]})
        print(f"  binary (EASE^R)               λ={lam:5d}  R@10={r10*100:.3f}%  nDCG@10={nd10*100:.3f}%",
              flush=True)
    del Xbin
    gc.collect()

    # 2. soft_binary (CW-EASE^R)
    print("  Evaluating soft_binary (CW-EASE^R) scheme ...", flush=True)
    Xsb  = build_Xsb(tr_df, n_u, n_i)
    for lam in LAMBDAS:
        B = ease_B(Xsb, lam)
        m, _ = evaluate_scores_batched(Xsb, B, gt, Xsb, (5,10,50,100))
        del B
        r10 = m[10]["R"]; nd10 = m[10]["nDCG"]
        records.append({"Scheme": "soft_binary (CW-EASE^R)", "lambda": lam,
                         "Recall@5":   m[5]["R"],
                         "Recall@10":  r10,
                         "nDCG@10":    nd10,
                         "Recall@100": m[100]["R"]})
        print(f"  soft_binary (CW-EASE^R)       λ={lam:5d}  R@10={r10*100:.3f}%  nDCG@10={nd10*100:.3f}%",
              flush=True)
    del Xsb
    gc.collect()

    # 3. linear
    print("  Evaluating linear scheme ...", flush=True)
    Xlin = build_Xlin(tr_df, n_u, n_i)
    for lam in LAMBDAS:
        B = ease_B(Xlin, lam)
        m, _ = evaluate_scores_batched(Xlin, B, gt, Xlin, (5,10,50,100))
        del B
        r10 = m[10]["R"]; nd10 = m[10]["nDCG"]
        records.append({"Scheme": "linear", "lambda": lam,
                         "Recall@5":   m[5]["R"],
                         "Recall@10":  r10,
                         "nDCG@10":    nd10,
                         "Recall@100": m[100]["R"]})
        print(f"  linear                        λ={lam:5d}  R@10={r10*100:.3f}%  nDCG@10={nd10*100:.3f}%",
              flush=True)
    del Xlin
    gc.collect()

    df_lam = pd.DataFrame(records)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for scheme, color in [("binary (EASE^R)", COLORS[0]),
                           ("soft_binary (CW-EASE^R)", COLORS[1]),
                           ("linear", COLORS[2])]:
        sub = df_lam[df_lam["Scheme"] == scheme]
        axes[0].plot(sub["lambda"], sub["Recall@10"]*100,
                     marker="o", markersize=5, color=color, label=scheme, linewidth=2)
        axes[1].plot(sub["lambda"], sub["nDCG@10"]*100,
                     marker="o", markersize=5, color=color, label=scheme, linewidth=2)

    for ax, ylabel, title in zip(axes, ["Recall@10 (%)", "nDCG@10 (%)"],
                                  ["Recall@10 vs λ", "nDCG@10 vs λ"]):
        ax.set_xlabel("Regularisation λ", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([50, 100, 200, 500, 1000, 2000])

    fig.suptitle("Hyperparameter Sensitivity: λ vs Ranking Metrics\n"
                 "X-Wines Full 21M  |  Threshold ≥ 4.0",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lambda_sensitivity.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Figure saved → results/figures/lambda_sensitivity.png")
    return df_lam


# ─── 3. Cold-Start Analysis ───────────────────────────────────────────────────

def coldstart_analysis(tr_df, tr_pos, te_pos, gt, n_u, n_i):
    print("\n" + "="*60)
    print("  [3] COLD-START ANALYSIS (<5 vs ≥5 positive interactions)")
    print("="*60)

    import gc
    user_n_pos = tr_pos.groupby("uid").size().to_dict()
    THRESHOLDS = [3, 5, 10]
    records = []

    # 1. Evaluate EASE^R (baseline)
    print("  Evaluating EASE^R (baseline) ...", flush=True)
    Xbin = build_Xbin(tr_pos, n_u, n_i)
    B_base = ease_B(Xbin, 500)
    for cold_thresh in THRESHOLDS:
        cold_users = {uid for uid in gt if user_n_pos.get(uid, 0) < cold_thresh}
        warm_users = {uid for uid in gt if user_n_pos.get(uid, 0) >= cold_thresh}
        for group_name, group in [("cold", cold_users), ("warm", warm_users)]:
            gt_sub = {uid: gt[uid] for uid in group if uid in gt}
            if not gt_sub: continue
            m, _ = evaluate_scores_batched(Xbin, B_base, gt_sub, Xbin, [5, 10, 50, 100])
            records.append({
                "cold_threshold": cold_thresh, "user_group": group_name,
                "n_users": len(gt_sub), "model": "EASE^R (baseline)",
                "Recall@5": m[5]["R"],   "nDCG@5":   m[5]["nDCG"],
                "Recall@10": m[10]["R"],  "nDCG@10":  m[10]["nDCG"],
                "Recall@100": m[100]["R"], "nDCG@100": m[100]["nDCG"],
            })
            print(f"  thresh={cold_thresh}  {group_name:4} ({len(gt_sub):6,} users)  "
                  f"EASE^R (baseline)          "
                  f"R@10={m[10]['R']*100:.3f}%  nDCG@10={m[10]['nDCG']*100:.3f}%",
                  flush=True)
    del Xbin, B_base
    gc.collect()

    # 2. Evaluate CW-EASE^R+IPS (proposed)
    print("  Evaluating CW-EASE^R+IPS (proposed) ...", flush=True)
    Xsb  = build_Xsb(tr_df, n_u, n_i)
    freq = np.bincount(np.where(Xsb > 0)[1], minlength=n_i).astype(np.float32)
    p_i  = np.clip((freq / (freq.max()+1e-9)) ** 0.3, 1e-3, 1.)
    Xips = Xsb / np.sqrt(p_i)[np.newaxis, :]
    B_prop = ease_B(Xips, 750)
    del Xips
    gc.collect()

    for cold_thresh in THRESHOLDS:
        cold_users = {uid for uid in gt if user_n_pos.get(uid, 0) < cold_thresh}
        warm_users = {uid for uid in gt if user_n_pos.get(uid, 0) >= cold_thresh}
        for group_name, group in [("cold", cold_users), ("warm", warm_users)]:
            gt_sub = {uid: gt[uid] for uid in group if uid in gt}
            if not gt_sub: continue
            m, _ = evaluate_scores_batched(Xsb, B_prop, gt_sub, Xsb, [5, 10, 50, 100])
            records.append({
                "cold_threshold": cold_thresh, "user_group": group_name,
                "n_users": len(gt_sub), "model": "CW-EASE^R+IPS (proposed)",
                "Recall@5": m[5]["R"],   "nDCG@5":   m[5]["nDCG"],
                "Recall@10": m[10]["R"],  "nDCG@10":  m[10]["nDCG"],
                "Recall@100": m[100]["R"], "nDCG@100": m[100]["nDCG"],
            })
            print(f"  thresh={cold_thresh}  {group_name:4} ({len(gt_sub):6,} users)  "
                  f"CW-EASE^R+IPS (proposed)   "
                  f"R@10={m[10]['R']*100:.3f}%  nDCG@10={m[10]['nDCG']*100:.3f}%",
                  flush=True)
    del Xsb, B_prop
    gc.collect()

    df_cs = pd.DataFrame(records)


    # Plot: cold_thresh=5, cold vs warm, R@10 and nDCG@10
    sub5 = df_cs[df_cs["cold_threshold"] == 5].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = ["cold", "warm"]
    models = ["EASE^R (baseline)", "CW-EASE^R+IPS (proposed)"]
    x = np.arange(len(groups)); width = 0.32

    for ax, metric, ylabel in zip(axes, ["Recall@10", "nDCG@10"],
                                   ["Recall@10 (%)", "nDCG@10 (%)"]):
        for mi, (model, color) in enumerate(zip(models, [COLORS[0], COLORS[1]])):
            vals = [sub5[(sub5["user_group"]==g) & (sub5["model"]==model)][metric].values[0]*100
                    for g in groups]
            bars = ax.bar(x + mi*width - width/2, vals, width, label=model,
                          color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        n_cold = sub5[sub5["user_group"]=="cold"]["n_users"].values[0]
        n_warm = sub5[sub5["user_group"]=="warm"]["n_users"].values[0]
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cold users\n(<5 positives)\nn={n_cold:,}",
                             f"Warm users\n(≥5 positives)\nn={n_warm:,}"])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(metric.replace("@", " @"), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Cold-Start Analysis: Cold (<5 pos.) vs Warm (≥5 pos.) Users\n"
                 "X-Wines 150K  |  EASE^R vs CW-EASE^R+IPS",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "coldstart_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Figure saved → results/figures/coldstart_comparison.png")
    return df_cs

# ─── 4. Confidence Scheme Analysis ───────────────────────────────────────────

def confidence_scheme_analysis(tr_df, n_u, n_i):
    """
    Formal analysis: WHY does soft_binary beat linear?
    Key metrics:
      a) Signal-to-noise: mean(|c|) for positive vs negative items
      b) Gram matrix conditioning: condition number of G = X^T X + λI
      c) Effective item coverage: # items with |X[:,i]|_1 > 0
      d) Rating → confidence mapping visualisation
    """
    print("\n" + "="*60)
    print("  [4] CONFIDENCE SCHEME ANALYSIS")
    print("="*60)

    ratings = np.arange(1.0, 5.5, 0.5)
    schemes = {
        "binary_pos":  lambda r: 1.0 if r >= 4 else 0.0,
        "soft_binary": lambda r: 1.0 if r >= 4 else ((r-2.5)*0.2 if r >= 2.5 else -0.5),
        "linear":      lambda r: r - 3.5,
        "shifted":     lambda r: max(0, r - 3.0),
        "signed":      lambda r: (r - 3.0) / 2.0,
        "log_pos":     lambda r: (np.log2(r-2) if r >= 4 else (0.0 if r >= 3 else -0.3)),
    }
    scheme_colors = {s: c for s, c in zip(schemes.keys(), COLORS)}

    # A) Rating → confidence mapping plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, fn in schemes.items():
        c_vals = [fn(r) for r in ratings]
        lw = 2.5 if name in ("soft_binary", "binary_pos") else 1.5
        ls = "-" if name in ("soft_binary", "binary_pos") else "--"
        ax.plot(ratings, c_vals, marker="o", markersize=5,
                label=name, color=scheme_colors[name], linewidth=lw, linestyle=ls)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(4.0, color="gray", linewidth=1.0, linestyle=":", alpha=0.7, label="threshold=4.0")
    ax.set_xlabel("Rating r", fontsize=12)
    ax.set_ylabel("Confidence c(r)", fontsize=12)
    ax.set_title("Rating → Confidence Mapping: 6 Schemes\n"
                 "soft_binary (bold) is the proposed mapping", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(ratings)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confidence_mapping.png"), bbox_inches="tight")
    plt.close(fig)

    # B) Quantitative signal analysis
    rating_counts = tr_df["Rating"].value_counts().sort_index()
    total = len(tr_df)
    records = []
    for name, fn in schemes.items():
        c_vals_all = np.array([fn(r) for r in tr_df["Rating"].values])
        pos_mask = c_vals_all > 0; neg_mask = c_vals_all < 0; neu_mask = c_vals_all == 0
        records.append({
            "scheme": name,
            "n_positive_signal": pos_mask.sum(),
            "n_negative_signal": neg_mask.sum(),
            "n_neutral":         neu_mask.sum(),
            "mean_pos_c":        c_vals_all[pos_mask].mean() if pos_mask.any() else 0,
            "mean_neg_c":        c_vals_all[neg_mask].mean() if neg_mask.any() else 0,
            "snr_proxy":         (c_vals_all[pos_mask].mean() /
                                  (abs(c_vals_all[neg_mask]).mean() + 1e-9)
                                  if neg_mask.any() and pos_mask.any() else np.inf),
            "variance_c":        c_vals_all.var(),
            "l2_norm_c":         np.linalg.norm(c_vals_all),
        })
        print(f"  {name:<16}  pos={pos_mask.sum():,} neg={neg_mask.sum():,} "
              f"neu={neu_mask.sum():,}  "
              f"μ_pos={c_vals_all[pos_mask].mean():.3f}"
              f"  var={c_vals_all.var():.4f}")

    df_scheme = pd.DataFrame(records)

    # C) Bar chart: positive/negative/neutral signal distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scheme_names = [r["scheme"] for r in records]
    x = np.arange(len(scheme_names)); w = 0.28

    pos_counts = [r["n_positive_signal"] for r in records]
    neg_counts = [r["n_negative_signal"] for r in records]
    neu_counts = [r["n_neutral"]         for r in records]

    axes[0].bar(x - w, pos_counts, w, label="Positive c>0", color="#16A34A", alpha=0.85)
    axes[0].bar(x,     neu_counts, w, label="Neutral  c=0", color="#9CA3AF", alpha=0.85)
    axes[0].bar(x + w, neg_counts, w, label="Negative c<0", color="#DC2626", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(scheme_names, rotation=30, ha="right")
    axes[0].set_ylabel("Number of interactions", fontsize=11)
    axes[0].set_title("Signal Distribution per Confidence Scheme\n(train set)",
                       fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # SNR proxy
    snr = [r["snr_proxy"] for r in records]
    variance = [r["variance_c"] for r in records]
    bars = axes[1].bar(x, snr, color=[scheme_colors[s] for s in scheme_names],
                       alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, snr):
        label = f"{val:.1f}" if val < 100 else "∞"
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                     label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(scheme_names, rotation=30, ha="right")
    axes[1].set_ylabel("SNR proxy (μ_pos / |μ_neg|)", fontsize=11)
    axes[1].set_title("Signal-to-Noise Ratio per Confidence Scheme\n"
                       "Higher = cleaner positive signal",
                       fontsize=11, fontweight="bold")

    fig.suptitle("Why soft_binary outperforms linear:\n"
                 "Better Signal-to-Noise Ratio with meaningful negative coverage",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scheme_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Figures saved → results/figures/")
    return df_scheme

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Statistical Analysis Suite — CW-EASE^R Paper")
    print("  X-Wines 150K  |  4 Analyses")
    print("=" * 70)
    t0 = time.time()

    df, tr_df, te_df, tr_pos, te_pos, gt, n_u, n_i, _, _ = load_data()
    print(f"Loaded: {len(df):,} ratings | {n_u:,} users | {n_i:,} items | {len(gt):,} eval users")

    # Run all 4 analyses
    df_wil, diff_r, diff_nd, common = wilcoxon_analysis(tr_df, tr_pos, gt, n_u, n_i)
    df_lam = lambda_sensitivity(tr_df, tr_pos, gt, n_u, n_i)
    df_cs  = coldstart_analysis(tr_df, tr_pos, te_pos, gt, n_u, n_i)
    df_sch = confidence_scheme_analysis(tr_df, n_u, n_i)

    # Save all CSVs
    out_dir = os.path.join(ROOT, "results")
    df_wil.to_csv(os.path.join(out_dir, "stat_test_results.csv"),       index=False)
    df_lam.to_csv(os.path.join(out_dir, "lambda_sensitivity.csv"),       index=False)
    df_cs.to_csv(os.path.join(out_dir,  "coldstart_analysis.csv"),       index=False)
    df_sch.to_csv(os.path.join(out_dir, "confidence_scheme_stats.csv"),  index=False)

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"  Wall-clock: {time.time()-t0:.1f}s")
    print(f"  CSVs  → results/stat_test_results.csv, lambda_sensitivity.csv,")
    print(f"          coldstart_analysis.csv, confidence_scheme_stats.csv")
    print(f"  Plots → results/figures/lambda_sensitivity.png")
    print(f"          results/figures/coldstart_comparison.png")
    print(f"          results/figures/scheme_comparison.png")
    print(f"          results/figures/confidence_mapping.png")

if __name__ == "__main__":
    main()
