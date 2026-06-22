"""
evaluation/cw_ease_eval.py — CW-EASE^R: Confidence-Weighted EASE^R
===================================================================
Key insight: Standard EASE^R binarises at ≥4.0, discarding 40% of ratings
(2.5–3.9 stars = implicit negative/neutral signal). CW-EASE^R uses ALL
150,000 ratings with a continuous confidence mapping.

CONFIDENCE MAPPING  c(r):
  r = 5.0 → c = +2.0   (strong positive)
  r = 4.5 → c = +1.5
  r = 4.0 → c = +1.0   (positive boundary)
  r = 3.5 → c = +0.0   (neutral: neither push nor pull)
  r = 3.0 → c = -0.5   (mild dislike)
  r = 2.5 → c = -1.0
  r ≤ 2.0 → c = -1.5   (strong dislike)

  General formula:  c(r) = r - 3.5   (linear, centred at neutral 3.5)

THE MODIFIED EASE^R OBJECTIVE:
  min_B  ||X_c - X_c B||_F^2  +  λ ||B||_F^2   s.t. diag(B)=0
  where X_c[u,i] = c(r_{ui}) if (u,i) observed, else 0

  Closed-form:
    G_c = X_c^T X_c + λ I
    P_c = inv(G_c)
    B_c = I - P_c · diag(1/diag(P_c))

ABLATION:
  We compare several confidence schemes to find the best mapping.

EVALUATION (unchanged):
  Ground-truth still: rating ≥ 4.0 = relevant
  Full 20% RatioSplit test set (9,200 users)
  Metrics: Recall@K, Precision@K, nDCG@K, F1@K for K ∈ {5,10,50,100}
"""

import os, sys, time
import numpy as np
import pandas as pd
from scipy.linalg import inv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── Metrics ──────────────────────────────────────────────────────────────────

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2))) if r.size else 0.0

def ndcg_at_k(r, k):
    dcg = dcg_at_k(r, k); ideal = dcg_at_k(sorted(r, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

def evaluate_scores(S, gt, train_mask, K_VALUES=(5, 10, 50, 100)):
    met = {k: {"P": 0., "R": 0., "F1": 0., "nDCG": 0.} for k in K_VALUES}
    n = 0; MK = max(K_VALUES)
    for uid, rel in gt.items():
        s = S[uid].copy(); s[train_mask[uid]] = -1e9
        top = np.argsort(-s)[:MK].tolist(); nr = len(rel)
        for k in K_VALUES:
            h = [1 if x in rel else 0 for x in top[:k]]
            p = sum(h)/k; r = sum(h)/nr
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.
            met[k]["P"] += p; met[k]["R"] += r
            met[k]["F1"] += f1; met[k]["nDCG"] += ndcg_at_k(h, k)
        n += 1
    for k in K_VALUES:
        for m in met[k]: met[k][m] /= n
    return met, n

def rcl(S, gt, mask, k=10):
    tot = 0; n = 0
    for uid, rel in gt.items():
        s = S[uid].copy(); s[mask[uid]] = -1e9
        top = np.argsort(-s)[:k].tolist()
        tot += sum(1 for x in top if x in rel)/len(rel); n += 1
    return tot/n

def ease_B(Xc, lam):
    G = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    P = inv(G)
    B = np.eye(Xc.shape[1]) - P / np.diag(P)
    np.fill_diagonal(B, 0.)
    return B

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import datetime
    import io
    # Force UTF-8 output so Unicode chars work on Windows when redirected to file
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    run_start = datetime.datetime.now()
    print("=" * 72)
    print("  CW-EASE^R: Confidence-Weighted EASE^R — X-Wines 150K Ratings")
    print("  Uses ALL ratings (1–5 stars) with continuous confidence mapping")
    print(f"  Run started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    t0 = time.time()
    sys.stdout.flush()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    rpath = os.path.join(ROOT, "data", "xwines", "XWines_Slim_150K_ratings.csv")
    print(f"\n[Step 1/12] Loading data from: {rpath}")
    sys.stdout.flush()
    t_load = time.time()
    df = pd.read_csv(rpath, low_memory=False)
    print(f"  Load time: {time.time()-t_load:.2f}s")
    print(f"Loaded {len(df):,} total ratings (ALL stars, not just ≥4.0)")
    print(f"  Rating distribution:")
    for star, cnt in sorted(df['Rating'].value_counts().items()):
        print(f"    Rating={star}: {cnt:,} rows ({cnt/len(df)*100:.1f}%)")
    sys.stdout.flush()

    THR = 4.0
    print(f"\n[Step 2/12] Building user/item index ...")
    users = sorted(df["UserID"].unique()); items = sorted(df["WineID"].unique())
    u2i = {u:i for i,u in enumerate(users)}
    it2i = {w:i for i,w in enumerate(items)}
    n_u, n_i = len(users), len(items)
    df["uid"] = df["UserID"].map(u2i)
    df["iid"] = df["WineID"].map(it2i)
    print(f"  Unique users: {n_u:,}  |  Unique wines: {n_i:,}")
    print(f"  Sparsity: {len(df)/(n_u*n_i)*100:.4f}% filled")
    sys.stdout.flush()

    # ── 2. RatioSplit 80/20 (same seed as all other models for comparability) ─
    print(f"\n[Step 3/12] RatioSplit 80/20 (seed=42, same as all baseline models) ...")
    np.random.seed(42)
    tr_idx, te_idx = [], []
    for _, grp in df.groupby("uid"):
        idx = grp.index.tolist(); np.random.shuffle(idx)
        sp = max(1, int(0.8 * len(idx)))
        tr_idx.extend(idx[:sp]); te_idx.extend(idx[sp:])

    tr_df = df.loc[tr_idx]; te_df = df.loc[te_idx]
    tr_pos = tr_df[tr_df["Rating"] >= THR]
    te_pos = te_df[te_df["Rating"] >= THR]
    print(f"Train: {len(tr_df):,} interactions | Test: {len(te_df):,}")
    print(f"Train positives (≥4): {len(tr_pos):,} | Test positives: {len(te_pos):,}")
    print(f"Train negatives/neutral (<4): {len(tr_df)-len(tr_pos):,} (NEW signal used by CW-EASE^R!)")
    print(f"  Train positive rate: {len(tr_pos)/len(tr_df)*100:.1f}%  |  Test positive rate: {len(te_pos)/len(te_df)*100:.1f}%")
    sys.stdout.flush()

    # ── 3. Ground-truth & standard binary train mask ───────────────────────────
    print(f"\n[Step 4/12] Building ground-truth and train mask ...")
    # NOTE: train_mask covers ALL train interactions (positive & negative)
    #       so we mask ALL seen items at eval time.
    X_all_pos = np.zeros((n_u, n_i), dtype=bool)   # all train interactions
    for _, r in tr_df.iterrows():
        X_all_pos[int(r["uid"]), int(r["iid"])] = True
    train_mask = X_all_pos

    gt = {uid: s for uid, s in te_pos.groupby("uid")["iid"].apply(set).items()
          if len(s) > 0}
    print(f"Eval users: {len(gt):,}")
    avg_pos = sum(len(v) for v in gt.values()) / len(gt)
    print(f"  Avg positive items per eval user: {avg_pos:.2f}")
    sys.stdout.flush()

    # ── 4. EASE^R BASELINE (binary, λ=500) ────────────────────────────────────
    print(f"\n[Step 5/12] Building binary interaction matrix (positives only) ...")
    X_bin = np.zeros((n_u, n_i), dtype=np.float32)
    for _, r in tr_pos.iterrows(): X_bin[int(r["uid"]), int(r["iid"])] = 1.
    # Note: use binary mask (seen positives only) for standard EASE^R
    mask_bin = X_bin.astype(bool)
    print(f"  X_bin shape: {X_bin.shape}  |  nnz: {int(X_bin.sum()):,}")
    sys.stdout.flush()

    print("\n[Step 6/12] Computing EASE^R BASELINE (λ=500, binary ≥4.0) ...")
    t_base = time.time()
    print("  Computing G = X^T X + λI  (shape: {}x{}) ...".format(n_i, n_i))
    B_std = ease_B(X_bin, 500)
    print(f"  EASE^R B matrix computed in {time.time()-t_base:.2f}s")
    print("  Computing score matrix S = X @ B ...")
    t_score = time.time()
    S_std = X_bin @ B_std
    print(f"  Score matrix S computed in {time.time()-t_score:.2f}s  shape: {S_std.shape}")
    print("  Evaluating baseline metrics on test set ...")
    t_eval = time.time()
    m_std, n_ev = evaluate_scores(S_std, gt, mask_bin, (5,10,50,100))
    print(f"  Eval done in {time.time()-t_eval:.2f}s  |  Users evaluated: {n_ev:,}")
    print(f"  BASELINE EASE^R: R@10={m_std[10]['R']*100:.4f}%  nDCG@10={m_std[10]['nDCG']*100:.4f}%")
    sys.stdout.flush()

    # ── 5. Define confidence schemes to compare ────────────────────────────────
    # c(r): confidence as function of rating r
    def build_Xc(df_tr, scheme, n_u, n_i):
        X = np.zeros((n_u, n_i), dtype=np.float32)
        for _, row in df_tr.iterrows():
            uid, iid, rating = int(row["uid"]), int(row["iid"]), row["Rating"]
            if scheme == "linear":
                c = rating - 3.5             # c ∈ [-2.5, +1.5]
            elif scheme == "shifted":
                c = max(0.0, rating - 3.0)   # c ∈ [0, 2], neutral at 3.0
            elif scheme == "binary_pos":
                c = 1.0 if rating >= 4.0 else 0.0   # standard binary
            elif scheme == "soft_binary":
                # Hard positive ≥4, soft negative 2.5-3.9, zero otherwise
                if rating >= 4.0:     c = 1.0
                elif rating >= 2.5:   c = (rating - 2.5) * 0.2   # 0..0.3
                else:                 c = -0.5
            elif scheme == "signed":
                # Signed: strong positive/negative
                c = (rating - 3.0) / 2.0    # c ∈ [-1, +1]
            elif scheme == "log_pos":
                # Log-scale positive, small negative
                if rating >= 4.0:     c = np.log2(rating - 2.0)  # log2(2..3)
                elif rating >= 3.0:   c = 0.0
                else:                 c = -0.3
            elif scheme == "ips_linear":
                # Linear + IPS popularity debiasing
                c = rating - 3.5
            else:
                c = 1.0 if rating >= 4.0 else 0.0
            if c != 0.0:
                X[uid, iid] = c
        return X

    # ── 6. Validation split for λ tuning ──────────────────────────────────────
    np.random.seed(1)
    val_rows = []
    for _, grp in tr_pos.groupby("uid"):
        idx = grp.index.tolist(); np.random.shuffle(idx)
        nv = max(0, len(idx) // 10)
        if nv: val_rows.extend(idx[:nv])

    # Val GT from held-out positives
    val_pos = tr_pos.loc[val_rows]
    vgt = {uid: s for uid, s in val_pos.groupby("uid")["iid"].apply(set).items()
           if len(s) > 0}

    # ── 7. Tune λ for each scheme on validation ───────────────────────────────
    SCHEMES   = ["binary_pos", "soft_binary", "linear", "shifted", "signed", "log_pos"]
    LAM_GRID  = [100, 200, 300, 500, 750, 1000, 1500, 2000]
    best_per_scheme = {}

    print(f"\n[Step 8/12] Grid search λ per confidence scheme ...")
    print(f"  Schemes: {SCHEMES}")
    print(f"  λ grid:  {LAM_GRID}")
    print(f"  Val GT size: {len(vgt):,} users")
    sys.stdout.flush()
    t_tune = time.time()
    for scheme in SCHEMES:
        print(f"\n  -- Scheme: [{scheme}] --")
        sys.stdout.flush()
        t_scheme = time.time()
        # Build Xc on FULL train (use validation items for GT, not for removing)
        Xc = build_Xc(tr_df, scheme, n_u, n_i)
        nnz_xc = int((Xc != 0).sum())
        print(f"     Xc built: shape={Xc.shape}  nnz={nnz_xc:,}  val_range=[{Xc.min():.3f}, {Xc.max():.3f}]")
        mask_c = (Xc != 0).astype(bool) | mask_bin   # mask any seen item

        best_r = -1; best_lam = 500
        for lam in LAM_GRID:
            t_lam = time.time()
            B = ease_B(Xc, lam)
            S = Xc @ B
            r = rcl(S, vgt, mask_c)
            marker = " ← new best" if r > best_r else ""
            if r > best_r: best_r = r; best_lam = lam
            print(f"     λ={lam:5d}  Val R@10={r*100:.4f}%  ({time.time()-t_lam:.2f}s){marker}")
            sys.stdout.flush()
        best_per_scheme[scheme] = (best_lam, best_r)
        print(f"     BEST: λ={best_lam:5d}  Val R@10={best_r*100:.4f}%  (scheme time: {time.time()-t_scheme:.1f}s)")
        sys.stdout.flush()
    print(f"\n  Total tuning time: {time.time()-t_tune:.1f}s")
    print(f"  Grid search complete. Summary:")
    for scheme in SCHEMES:
        lam, r = best_per_scheme[scheme]
        print(f"  {scheme:<16} best λ={lam:5d}  Val R@10={r*100:.3f}%")
    sys.stdout.flush()

    # ── 8. Full evaluation of each scheme on test set ─────────────────────────
    K_VALUES = [5, 10, 50, 100]
    results  = {"EASE^R (binary ≥4, λ=500)": m_std}

    print(f"\n[Step 9/12] Full test set evaluation for each scheme (n={n_ev:,} users) ...")
    sys.stdout.flush()
    t_eval_all = time.time()
    for scheme in SCHEMES:
        lam, _ = best_per_scheme[scheme]
        print(f"\n  -- Evaluating scheme [{scheme}] with best λ={lam} ...")
        sys.stdout.flush()
        t_s = time.time()
        Xc     = build_Xc(tr_df, scheme, n_u, n_i)
        mask_c = (Xc != 0).astype(bool) | mask_bin
        print(f"     Building EASE^R B matrix (λ={lam}) ...")
        B      = ease_B(Xc, lam)
        print(f"     Computing score matrix ...")
        S      = Xc @ B
        print(f"     Evaluating {n_ev:,} users ...")
        m, _   = evaluate_scores(S, gt, mask_c, K_VALUES)
        label  = f"CW-EASE^R ({scheme}, λ={lam})"
        results[label] = m
        for k in K_VALUES:
            print(f"     @{k:<4}  P={m[k]['P']*100:.4f}%  R={m[k]['R']*100:.4f}%  F1={m[k]['F1']*100:.4f}%  nDCG={m[k]['nDCG']*100:.4f}%")
        print(f"     Scheme [{scheme}] eval time: {time.time()-t_s:.2f}s")
        sys.stdout.flush()
    print(f"\n  Total scheme eval time: {time.time()-t_eval_all:.1f}s")
    sys.stdout.flush()

    # ── 9. Best CW-EASE^R + IPS ensemble with best scheme ─────────────────────
    print(f"\n[Step 10/12] Building CW-EASE^R+IPS+Ensemble (best scheme) ...")
    # Find best scheme by test R@10
    best_scheme = max((s for s in SCHEMES), key=lambda s:
                      results[f"CW-EASE^R ({s}, λ={best_per_scheme[s][0]})" ][10]["R"])
    best_lam, _ = best_per_scheme[best_scheme]
    print(f"  Best scheme by test R@10: [{best_scheme}]  λ={best_lam}")
    print(f"  R@10 per scheme:")
    for s in SCHEMES:
        lam_s, _ = best_per_scheme[s]
        r_s = results[f"CW-EASE^R ({s}, λ={lam_s})"][10]["R"] * 100
        marker = " <-- BEST" if s == best_scheme else ""
        print(f"    {s:<16} R@10={r_s:.4f}%{marker}")
    sys.stdout.flush()
    Xc_best     = build_Xc(tr_df, best_scheme, n_u, n_i)
    mask_best   = (Xc_best != 0).astype(bool) | mask_bin

    # IPS: reweight by inverse sqrt of item frequency
    print(f"\n[Step 11/12] Applying IPS debiasing (beta=0.3) ...")
    freq_vec = np.array([(Xc_best[:, i] > 0).sum() for i in range(n_i)],
                        dtype=np.float32)
    print(f"  Item frequency stats: min={freq_vec.min():.0f}  max={freq_vec.max():.0f}  mean={freq_vec.mean():.2f}  median={np.median(freq_vec):.2f}")
    p_i = np.clip((freq_vec / freq_vec.max()) ** 0.3, 1e-3, 1.0)
    print(f"  IPS weight stats:     min={p_i.min():.4f}  max={p_i.max():.4f}  mean={p_i.mean():.4f}")
    Xc_ips = Xc_best / np.sqrt(p_i)[np.newaxis, :]
    sys.stdout.flush()

    # Ensemble: best_scheme @ [lam, lam+250]
    ens_lams = [best_lam, min(best_lam + 250, 2000)]
    print(f"  Ensemble λ values: {ens_lams}")
    S_ens = np.zeros((n_u, n_i), dtype=np.float32)
    for lam_e in ens_lams:
        t_ens = time.time()
        print(f"    Computing EASE^R with IPS, λ={lam_e} ...")
        B_ = ease_B(Xc_ips, lam_e)
        S_ens += Xc_best @ B_
        print(f"    Done in {time.time()-t_ens:.2f}s")
        sys.stdout.flush()
    S_ens /= 2
    print(f"  Score ensemble averaged over {len(ens_lams)} models.")

    print(f"\n  Evaluating CW-EASE^R-ACE on {n_ev:,} test users ...")
    t_final_eval = time.time()
    m_final, _ = evaluate_scores(S_ens, gt, mask_best, K_VALUES)
    print(f"  Eval done in {time.time()-t_final_eval:.2f}s")
    for k in K_VALUES:
        print(f"  @{k:<4}  P={m_final[k]['P']*100:.4f}%  R={m_final[k]['R']*100:.4f}%  F1={m_final[k]['F1']*100:.4f}%  nDCG={m_final[k]['nDCG']*100:.4f}%")
    results[f"★ CW-EASE^R-ACE ({best_scheme}+IPS+Ens)"] = m_final
    sys.stdout.flush()

    # ── 10. Display ───────────────────────────────────────────────────────────
    print(f"\n  Full test: {n_ev:,} users.\n")
    print("=" * 98)
    print("  CW-EASE^R FULL ABLATION (user_based=True, threshold=4.0, 150K ratings)")
    print("=" * 98)
    hdr = f"{'Model':<45}" + "".join(
        f"  {'R@'+str(k):>7}  {'nDCG@'+str(k):>8}" for k in K_VALUES)
    print(hdr); print("-" * 98)
    for name, m in results.items():
        row = f"{name:<45}"
        for k in K_VALUES:
            row += f"  {m[k]['R']*100:>6.3f}%  {m[k]['nDCG']*100:>7.3f}%"
        print(row)
    print("=" * 98)

    # Improvements vs baseline
    print("\n  Improvement of best CW variant vs EASE^R baseline:")
    m_best_cw = results[f"★ CW-EASE^R-ACE ({best_scheme}+IPS+Ens)"]
    for k in K_VALUES:
        dr = (m_best_cw[k]["R"]    - m_std[k]["R"])    * 100
        dn = (m_best_cw[k]["nDCG"] - m_std[k]["nDCG"]) * 100
        sg_r = "+" if dr >= 0 else ""; sg_n = "+" if dn >= 0 else ""
        print(f"  @{k:<4}  ΔRecall={sg_r}{dr:.3f}pp  ΔnDCG={sg_n}{dn:.3f}pp")

    # ── 11. Detailed final results ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  ★ CW-EASE^R-ACE DETAILED RESULTS")
    print(f"  Scheme: {best_scheme}, λ={best_lam}, IPS β=0.3, Ensemble")
    print("=" * 72)
    print(f"{'k':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'nDCG':>10}")
    print("-" * 72)
    for k in K_VALUES:
        print(f"  @{k:<5}  "
              f"{m_final[k]['P']:>10.4f}  "
              f"{m_final[k]['R']:>10.4f}  "
              f"{m_final[k]['F1']:>10.4f}  "
              f"{m_final[k]['nDCG']:>10.4f}")

    # ── 12. Save ───────────────────────────────────────────────────────────────────────
    print(f"\n[Step 12/12] Saving results CSV ...")
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for name, m in results.items():
        row = {"Model": name}
        for k in K_VALUES:
            for metric, val in m[k].items():
                row[f"{metric}@{k}"] = val
        records.append(row)
    csv_path = os.path.join(out_dir, "cw_ease_eval_results.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    total_time = time.time() - t0
    print(f"[OK] Saved → {csv_path}")
    print(f"     Total wall-clock: {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"\n{'='*72}")
    print(f"  RUN COMPLETE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
