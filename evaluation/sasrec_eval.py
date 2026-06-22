"""
evaluation/sasrec_eval.py — SASRec: Self-Attentive Sequential Recommendation
==============================================================================
Kang & McAuley. "Self-Attentive Sequential Recommendation." ICDM 2018.

Fix log:
  v2 — Fix NaN: float causal mask, normal_ init, no mixed bool/float masks,
       handle all-padding rows with nan_to_num, remove key_padding_mask
       (incompatible type with float attn_mask in this PyTorch version).
"""

import os, sys, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Ranking Metrics ─────────────────────────────────────────────────────────

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2))) if r.size else 0.0

def ndcg_at_k(r, k):
    dcg   = dcg_at_k(r, k)
    ideal = dcg_at_k(sorted(r, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

def ranking_metrics(recommended, relevant_set, k):
    hits = [1 if item in relevant_set else 0 for item in recommended[:k]]
    n_rel = len(relevant_set)
    p  = sum(hits) / k        if k     > 0 else 0.0
    r  = sum(hits) / n_rel    if n_rel > 0 else 0.0
    f1 = 2*p*r/(p+r)          if (p+r) > 0 else 0.0
    nd = ndcg_at_k(hits, k)
    return p, r, f1, nd

# ─── Building Blocks ─────────────────────────────────────────────────────────

class PFFN(nn.Module):
    """Point-wise Feed-Forward Network with residual connection."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

# ─── SASRec Model ─────────────────────────────────────────────────────────────

class SASRec(nn.Module):
    def __init__(self, n_items, max_len, dim=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.dim      = dim
        self.max_len  = max_len
        self.scale    = math.sqrt(dim)

        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.emb_drop = nn.Dropout(dropout)

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(n_layers)])
        self.attn_norms  = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.ff_layers   = nn.ModuleList([PFFN(dim, dropout) for _ in range(n_layers)])

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.dim)
        nn.init.normal_(self.item_emb.weight.data, std=std)
        nn.init.normal_(self.pos_emb.weight.data,  std=std)
        with torch.no_grad():
            self.item_emb.weight.data[0] = 0   # zero-out padding

    def forward(self, seq):
        """
        seq: (B, T) long — item ids, 0 = padding on the LEFT.
        Returns: (B, T, dim)
        """
        B, T = seq.shape
        # Positional indices 0..T-1
        pos  = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, T)
        x    = self.emb_drop(self.emb_norm(self.item_emb(seq) + self.pos_emb(pos)))

        # Float causal mask: upper-triangular = -inf
        causal = torch.full((T, T), float('-inf'), device=seq.device)
        causal = torch.triu(causal, diagonal=1)   # (T, T) float

        for attn, anorm, ff in zip(self.attn_layers, self.attn_norms, self.ff_layers):
            res   = x
            out, _ = attn(x, x, x, attn_mask=causal)   # float mask only
            out   = torch.nan_to_num(out)               # safety against edge-case NaN
            x     = anorm(res + out)
            x     = ff(x)

        return x  # (B, T, dim)

    @torch.no_grad()
    def score_all_items(self, seq, item_emb_all):
        """Score every item for each user in batch."""
        h     = self.forward(seq)            # (B, T, dim)
        h_t   = h[:, -1, :]                 # (B, dim)
        scores = (h_t @ item_emb_all.T) / self.scale  # (B, n_items)
        return scores

# ─── Dataset ─────────────────────────────────────────────────────────────────

class SASRecTrainDataset(Dataset):
    def __init__(self, sequences, n_items, max_len):
        self.seqs    = sequences   # list of item-id lists (1-indexed, no padding)
        self.n_items = n_items
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def _pad(self, seq):
        seq = seq[-self.max_len:]
        return [0] * (self.max_len - len(seq)) + seq

    def __getitem__(self, idx):
        items = self.seqs[idx]
        if len(items) < 2:
            inp = self._pad(items)
            pos = items[-1]
            neg = self._sample_neg(pos)
            return (torch.tensor(inp, dtype=torch.long),
                    torch.tensor(pos, dtype=torch.long),
                    torch.tensor(neg, dtype=torch.long))

        # Random training position
        t   = np.random.randint(1, len(items))
        inp = self._pad(items[:t])
        pos = items[t]
        neg = self._sample_neg(pos)
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long))

    def _sample_neg(self, pos):
        neg = np.random.randint(1, self.n_items + 1)
        while neg == pos:
            neg = np.random.randint(1, self.n_items + 1)
        return neg

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SASRec v2 — Self-Attentive Sequential Recommendation")
    print("  Kang & McAuley, ICDM 2018 | PyTorch GPU | X-Wines 150K")
    print("=" * 70)

    # 1. Load & sort
    path = os.path.join(ROOT, "data", "xwines", "XWines_Slim_150K_ratings.csv")
    df   = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df   = df.sort_values(["UserID", "Date"])
    print(f"Loaded {len(df):,} | {df['UserID'].nunique()} users | {df['WineID'].nunique()} wines")

    # 2. Keep positives only & encode items
    THRESHOLD = 4.0
    pos_df = df[df["Rating"] >= THRESHOLD].copy()
    items  = sorted(pos_df["WineID"].unique())
    it2i   = {w: i + 1 for i, w in enumerate(items)}  # 1-indexed (0 = pad)
    n_items = len(items)
    pos_df["iid"] = pos_df["WineID"].map(it2i)

    # 3. Build user sequences
    user_seqs = {}   # uid → sorted list of iids
    for uid, grp in pos_df.groupby("UserID"):
        seq = grp["iid"].tolist()
        if len(seq) >= 3:    # need train (≥2) + test (1)
            user_seqs[uid] = seq
    all_uids = list(user_seqs.keys())
    u2enc    = {u: i for i, u in enumerate(all_uids)}
    print(f"Users with ≥3 positive interactions: {len(user_seqs):,}")

    # 4. Leave-one-out split
    train_seqs = []
    test_gt    = {}
    for uid, seq in user_seqs.items():
        enc = u2enc[uid]
        train_seqs.append(seq[:-1])
        test_gt[enc] = {seq[-1]}

    # 5. Hyperparameters
    MAX_LEN  = 50
    DIM      = 64
    N_HEADS  = 2
    N_LAYERS = 2
    DROPOUT  = 0.2
    N_EPOCHS = 30
    LR       = 1e-3
    BATCH    = 512
    WD       = 1e-5

    # 6. Train
    model   = SASRec(n_items, MAX_LEN, dim=DIM,
                     n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    ds      = SASRecTrainDataset(train_seqs, n_items, MAX_LEN)
    loader  = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                         pin_memory=(DEVICE.type == "cuda"))

    print(f"\n[SASRec] Training {N_EPOCHS} epochs | {DEVICE} | "
          f"dim={DIM} heads={N_HEADS} layers={N_LAYERS} bs={BATCH}")
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for seq_t, pos_t, neg_t in loader:
            seq_t = seq_t.to(DEVICE)
            pos_t = pos_t.to(DEVICE)
            neg_t = neg_t.to(DEVICE)

            h     = model(seq_t)          # (B, T, dim)
            h_t   = h[:, -1, :]          # (B, dim)

            pos_e = model.item_emb(pos_t)   # (B, dim)
            neg_e = model.item_emb(neg_t)   # (B, dim)

            ps = (h_t * pos_e).sum(1) / model.scale
            ns = (h_t * neg_e).sum(1) / model.scale
            loss = -torch.log(torch.sigmoid(ps - ns) + 1e-8).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches  += 1

        sched.step()
        avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{N_EPOCHS}  loss={avg_loss:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.5f}")

    print(f"  Training done: {time.time()-t0:.1f}s")

    # 7. Evaluate (leave-one-out)
    model.eval()
    K_VALUES    = [5, 10, 50, 100]
    metrics_sum = {k: {"P": 0, "R": 0, "F1": 0, "nDCG": 0} for k in K_VALUES}
    n_eval      = 0

    MAX_K = max(K_VALUES)

    def pad_seq(seq):
        seq = seq[-MAX_LEN:]
        return [0] * (MAX_LEN - len(seq)) + seq

    # Pre-fetch all item embeddings (without padding emb at idx 0)
    with torch.no_grad():
        all_iids     = torch.arange(1, n_items + 1, dtype=torch.long, device=DEVICE)
        item_emb_all = model.item_emb(all_iids)    # (n_items, dim)

    BATCH_EVAL = 256
    uid_list   = list(test_gt.keys())

    with torch.no_grad():
        for start in range(0, len(uid_list), BATCH_EVAL):
            batch_enc = uid_list[start:start + BATCH_EVAL]
            seqs_padded = [pad_seq(train_seqs[enc]) for enc in batch_enc]
            seq_t = torch.tensor(seqs_padded, dtype=torch.long, device=DEVICE)

            scores = model.score_all_items(seq_t, item_emb_all)  # (B, n_items)
            scores = scores.cpu().numpy()

            for bi, enc in enumerate(batch_enc):
                relevant = test_gt[enc]   # set of 1-indexed iids
                s = scores[bi].copy()
                # Mask train-seen items from this user's train sequence
                for seen_iid in train_seqs[enc]:
                    s[seen_iid - 1] = -1e9  # iid is 1-indexed → index = iid-1
                top_k = (np.argsort(-s)[:MAX_K] + 1).tolist()  # back to 1-indexed
                for k in K_VALUES:
                    p, r, f1, nd = ranking_metrics(top_k, relevant, k)
                    metrics_sum[k]["P"]    += p
                    metrics_sum[k]["R"]    += r
                    metrics_sum[k]["F1"]   += f1
                    metrics_sum[k]["nDCG"] += nd
                n_eval += 1

    print(f"\n  Evaluated on {n_eval:,} users (leave-one-out).")

    # 8. Print & Save
    print("\n" + "=" * 75)
    print("  SASRec v2 RANKING RESULTS (leave-one-out, threshold=4.0)")
    print("=" * 75)
    print(f"{'k':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'nDCG':>10}")
    print("-" * 75)
    flat_row = {"Model": "SASRec (DL-Sequential)"}
    for k in K_VALUES:
        p  = metrics_sum[k]["P"]    / n_eval
        r  = metrics_sum[k]["R"]    / n_eval
        f1 = metrics_sum[k]["F1"]   / n_eval
        nd = metrics_sum[k]["nDCG"] / n_eval
        print(f"  @{k:<5}  {p:>10.4f}  {r:>10.4f}  {f1:>10.4f}  {nd:>10.4f}")
        flat_row[f"Precision@{k}"] = p
        flat_row[f"Recall@{k}"]    = r
        flat_row[f"F1@{k}"]        = f1
        flat_row[f"nDCG@{k}"]      = nd
    print("=" * 75)

    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([flat_row]).to_csv(
        os.path.join(out_dir, "sasrec_eval_results.csv"), index=False)
    print(f"\n[OK] Results saved → results/sasrec_eval_results.csv")
    print(f"     Total wall-clock: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
