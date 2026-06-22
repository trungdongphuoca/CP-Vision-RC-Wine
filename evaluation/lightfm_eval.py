"""
evaluation/lightfm_eval.py — Hybrid Content + CF (Pure-Python LightFM replacement)
====================================================================================
Implements Matrix Factorization with item content features using PyTorch (WARP loss).
Since the C-extension lightfm fails to compile on this system (no MSVC),
we implement the Hybrid MF model in pure PyTorch + GPU.

Architecture:
  score(u, i) = q_u @ p_i  +  q_u @ F_i @ W  +  b_i
where:
  q_u  ∈ R^d    — user latent factor
  p_i  ∈ R^d    — item CF latent factor
  F_i  ∈ R^f    — item content feature vector (one-hot: Type, Country, Body, Acidity + multi-hot: Grapes)
  W    ∈ R^(f×d) — content embedding projection

Loss: BPR (Bayesian Personalised Ranking) via pairwise sampling on GPU.
"""

import os, sys, time, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Ranking Metric Helpers ───────────────────────────────────────────────────

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    dcg = dcg_at_k(r, k)
    ideal = dcg_at_k(sorted(r, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

def ranking_metrics(recommended, relevant_set, k):
    hits = [1 if item in relevant_set else 0 for item in recommended[:k]]
    n_rel = len(relevant_set)
    precision = sum(hits) / k if k > 0 else 0.0
    recall    = sum(hits) / n_rel if n_rel > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ndcg      = ndcg_at_k(hits, k)
    return precision, recall, f1, ndcg

# ─── Hybrid MF Model ─────────────────────────────────────────────────────────

class HybridMF(nn.Module):
    def __init__(self, n_users, n_items, n_features, dim=64):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, dim)
        self.item_emb  = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_bias = nn.Embedding(n_users, 1)
        self.content_proj = nn.Linear(n_features, dim, bias=False)

        nn.init.normal_(self.user_emb.weight,  std=0.01)
        nn.init.normal_(self.item_emb.weight,  std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.user_bias.weight)

    def forward(self, user_ids, item_ids, item_features):
        qu = self.user_emb(user_ids)              # (B, d)
        pi = self.item_emb(item_ids)              # (B, d)
        fi = self.content_proj(item_features)     # (B, d)
        bi = self.item_bias(item_ids).squeeze(1)  # (B,)
        bu = self.user_bias(user_ids).squeeze(1)  # (B,)
        score = (qu * (pi + fi)).sum(dim=1) + bi + bu
        return score

    def score_all(self, user_ids, item_emb_all, item_feat_all, item_bias_all, user_bias):
        """Score user(s) against all items efficiently."""
        qu = self.user_emb(user_ids)              # (B, d)
        fi_all = self.content_proj(item_feat_all) # (n_items, d)
        combined = item_emb_all + fi_all          # (n_items, d)
        scores = qu @ combined.T                  # (B, n_items)
        scores += item_bias_all.T                 # broadcast (1, n_items)
        scores += user_bias                       # (B, 1)
        return scores

# ─── BPR Dataset ─────────────────────────────────────────────────────────────

class BPRDataset(Dataset):
    def __init__(self, pos_pairs, n_items, item_feat_tensor, neg_sample=1):
        self.pos = pos_pairs           # list of (uid, iid)
        self.n_items = n_items
        self.neg_sample = neg_sample
        self.item_feat = item_feat_tensor
        # Build set per user for fast negative sampling
        from collections import defaultdict
        self.user_pos_set = defaultdict(set)
        for u, i in pos_pairs:
            self.user_pos_set[u].add(i)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        u, i = self.pos[idx]
        # sample negative
        j = np.random.randint(self.n_items)
        while j in self.user_pos_set[u]:
            j = np.random.randint(self.n_items)
        return (torch.tensor(u, dtype=torch.long),
                torch.tensor(i, dtype=torch.long),
                torch.tensor(j, dtype=torch.long))

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Hybrid MF (Content + CF) Evaluation — X-Wines 150K Ratings")
    print("  PyTorch BPR Loss | Features: Type, Country, Body, Acidity, Grapes")
    print("=" * 70)

    # ── 1. Load ratings ──────────────────────────────────────────────────────
    ratings_path = os.path.join(ROOT, "data", "xwines", "XWines_Slim_150K_ratings.csv")
    wines_path   = os.path.join(ROOT, "data", "xwines", "XWines_Slim_1K_wines.csv")

    df_r = pd.read_csv(ratings_path, low_memory=False)
    df_w = pd.read_csv(wines_path)
    print(f"Ratings: {len(df_r):,} | Users: {df_r['UserID'].nunique()} | Wines: {df_r['WineID'].nunique()}")

    # ── 2. Encode users & items ───────────────────────────────────────────────
    users  = sorted(df_r["UserID"].unique())
    items  = sorted(df_r["WineID"].unique())
    u2i    = {u: i for i, u in enumerate(users)}
    it2i   = {w: i for i, w in enumerate(items)}
    i2it   = {i: w for w, i in it2i.items()}
    n_users = len(users)
    n_items = len(items)

    df_r["uid"] = df_r["UserID"].map(u2i)
    df_r["iid"] = df_r["WineID"].map(it2i)

    # ── 3. Build item feature matrix ─────────────────────────────────────────
    # Map wine metadata only for items present in ratings
    wine_lookup = df_w.set_index("WineID")
    feat_rows = []
    for iid_enc in range(n_items):
        wid = i2it[iid_enc]
        if wid in wine_lookup.index:
            row = wine_lookup.loc[wid]
        else:
            row = pd.Series({"Type": "Unknown", "Country": "Unknown",
                             "Body": "Unknown", "Acidity": "Unknown", "Grapes": "[]"})
        feat_rows.append(row)
    feat_df = pd.DataFrame(feat_rows).reset_index(drop=True)

    # OneHot: Type, Country, Body, Acidity
    cat_cols = ["Type", "Country", "Body", "Acidity"]
    cat_dummies = pd.get_dummies(feat_df[cat_cols], dtype=float)

    # MultiHot: Grapes (parse list strings)
    def parse_list(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else []
        except Exception:
            return []

    grapes_lists = feat_df["Grapes"].apply(parse_list).tolist()
    mlb = MultiLabelBinarizer(sparse_output=False)
    grapes_mat = mlb.fit_transform(grapes_lists).astype(float)
    grapes_df  = pd.DataFrame(grapes_mat, columns=[f"grape_{g}" for g in mlb.classes_])

    item_feat_np = np.hstack([cat_dummies.values, grapes_df.values]).astype(np.float32)
    n_features   = item_feat_np.shape[1]
    print(f"Item features: {n_features} dims (Type/Country/Body/Acidity + {grapes_mat.shape[1]} grape varieties)")

    item_feat_tensor = torch.tensor(item_feat_np, dtype=torch.float32).to(DEVICE)

    # ── 4. RatioSplit 80/20 ───────────────────────────────────────────────────
    THRESHOLD = 4.0
    np.random.seed(42)
    train_rows, test_rows = [], []
    for uid, grp in df_r.groupby("uid"):
        idx = grp.index.tolist()
        np.random.shuffle(idx)
        split = max(1, int(0.8 * len(idx)))
        train_rows.extend(idx[:split])
        test_rows.extend(idx[split:])

    train_df = df_r.loc[train_rows]
    test_df  = df_r.loc[test_rows]
    train_pos_df = train_df[train_df["Rating"] >= THRESHOLD]
    test_pos_df  = test_df[test_df["Rating"]  >= THRESHOLD]
    print(f"Train positives: {len(train_pos_df):,} | Test positives: {len(test_pos_df):,}")

    pos_pairs = list(zip(train_pos_df["uid"].values, train_pos_df["iid"].values))

    # ── 5. Build & Train model ───────────────────────────────────────────────
    DIM      = 64
    N_EPOCHS = 20
    LR       = 1e-3
    BATCH    = 2048

    model     = HybridMF(n_users, n_items, n_features, dim=DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    dataset    = BPRDataset(pos_pairs, n_items, item_feat_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                            num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    print(f"\n[HybridMF] Training {N_EPOCHS} epochs on {DEVICE}  (dim={DIM}, bs={BATCH}) ...")
    t_train = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for u_ids, pos_ids, neg_ids in dataloader:
            u_ids   = u_ids.to(DEVICE)
            pos_ids = pos_ids.to(DEVICE)
            neg_ids = neg_ids.to(DEVICE)
            pos_feat = item_feat_tensor[pos_ids]
            neg_feat = item_feat_tensor[neg_ids]

            pos_score = model(u_ids, pos_ids, pos_feat)
            neg_score = model(u_ids, neg_ids, neg_feat)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{N_EPOCHS}  loss={total_loss/len(dataloader):.4f}")

    print(f"  Training done in {time.time()-t_train:.1f}s")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    model.eval()
    K_VALUES = [5, 10, 50, 100]
    metrics_sum = {k: {"P": 0, "R": 0, "F1": 0, "nDCG": 0} for k in K_VALUES}
    n_eval_users = 0

    gt_per_user = test_pos_df.groupby("uid")["iid"].apply(set).to_dict()

    # Build seen-in-train set per user
    from collections import defaultdict
    train_seen = defaultdict(set)
    for _, row in train_df.iterrows():
        train_seen[row["uid"]].add(row["iid"])

    with torch.no_grad():
        # Pre-compute item embeddings + biases
        item_emb_all  = model.item_emb.weight             # (n_items, d)
        item_feat_all = item_feat_tensor                   # (n_items, f)
        item_bias_all = model.item_bias.weight             # (n_items, 1)

        BATCH_EVAL = 512
        uid_list = list(gt_per_user.keys())
        for start in range(0, len(uid_list), BATCH_EVAL):
            batch_uids = uid_list[start:start + BATCH_EVAL]
            uid_t = torch.tensor(batch_uids, dtype=torch.long).to(DEVICE)
            user_bias = model.user_bias(uid_t)             # (B, 1)

            scores = model.score_all(uid_t, item_emb_all,
                                     item_feat_all, item_bias_all, user_bias)
            scores = scores.cpu().numpy()                  # (B, n_items)

            for bi, uid in enumerate(batch_uids):
                relevant = gt_per_user[uid]
                s = scores[bi].copy()
                # Mask training items
                for seen_iid in train_seen[uid]:
                    s[seen_iid] = -1e9
                top_items = np.argsort(-s)[:max(K_VALUES)].tolist()
                for k in K_VALUES:
                    p, r, f1, nd = ranking_metrics(top_items, relevant, k)
                    metrics_sum[k]["P"]    += p
                    metrics_sum[k]["R"]    += r
                    metrics_sum[k]["F1"]   += f1
                    metrics_sum[k]["nDCG"] += nd
                n_eval_users += 1

    print(f"\n  Evaluated on {n_eval_users:,} users.")

    # ── 7. Print & Save ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  HybridMF (Content + BPR) RANKING RESULTS")
    print("=" * 75)
    print(f"{'k':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'nDCG':>10}")
    print("-" * 75)
    flat_row = {"Model": "HybridMF (Content+BPR)"}
    for k in K_VALUES:
        p  = metrics_sum[k]["P"]    / n_eval_users
        r  = metrics_sum[k]["R"]    / n_eval_users
        f1 = metrics_sum[k]["F1"]   / n_eval_users
        nd = metrics_sum[k]["nDCG"] / n_eval_users
        print(f"  @{k:<5}  {p:>10.4f}  {r:>10.4f}  {f1:>10.4f}  {nd:>10.4f}")
        flat_row[f"Precision@{k}"] = p
        flat_row[f"Recall@{k}"]    = r
        flat_row[f"F1@{k}"]        = f1
        flat_row[f"nDCG@{k}"]      = nd
    print("=" * 75)

    out_dir = os.path.join(ROOT, "results")
    pd.DataFrame([flat_row]).to_csv(
        os.path.join(out_dir, "lightfm_eval_results.csv"), index=False
    )
    print(f"\n[OK] Results saved → results/lightfm_eval_results.csv")

if __name__ == "__main__":
    main()
