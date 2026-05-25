"""
gnn_indexer.py — GNN-guided Semantic Indexing
=============================================
Builds an Item-Attribute heterogeneous graph, trains a PyTorch LightGCN
via self-supervised link prediction (BPR Loss), and performs Hierarchical
K-Means clustering on the resulting wine embeddings to assign GNN Semantic IDs.
"""

import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))
import config as cfg

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import time

def extract_year(title):
    if pd.isna(title): return "NV"
    import re
    m = re.search(r"(19|20)\d{2}", str(title))
    return m.group(0) if m else "NV"

def clean_text(t):
    if pd.isna(t): return "UNKN"
    import re
    return re.sub(r"[^A-Za-z0-9]", "", str(t)).upper()[:4]

def make_semantic_id(row):
    return (f"{clean_text(row['country'])}-"
            f"{clean_text(row.get('province',''))}-"
            f"{clean_text(row['variety'])}-"
            f"{row['vintage']}")

def main():
    print("="*60)
    print("  GNN-Guided Semantic Indexing Pipeline")
    print("="*60)

    # 1. Load catalog
    csv_path = str(cfg.WINE_CSV)
    print(f"Loading raw catalog from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["country", "variety", "description", "title"])
    df["vintage"] = df["title"].apply(extract_year)
    df = df.reset_index(drop=True)
    num_wines = len(df)
    print(f"Loaded {num_wines:,} valid wine entries.")

    # 2. Map categorical attribute nodes
    print("Mapping category nodes...")
    df["winery_clean"] = df["winery"].fillna("UnknownWinery")
    df["province_clean"] = df["province"].fillna("UnknownProvince")
    df["variety_clean"] = df["variety"].fillna("UnknownVariety")

    winery_cats = df["winery_clean"].astype("category").cat
    province_cats = df["province_clean"].astype("category").cat
    variety_cats = df["variety_clean"].astype("category").cat

    winery_ids = torch.tensor(winery_cats.codes.values, dtype=torch.long)
    province_ids = torch.tensor(province_cats.codes.values, dtype=torch.long)
    variety_ids = torch.tensor(variety_cats.codes.values, dtype=torch.long)

    num_wineries = len(winery_cats.categories)
    num_provinces = len(province_cats.categories)
    num_varieties = len(variety_cats.categories)

    print(f"  Wines     : {num_wines:,}")
    print(f"  Wineries  : {num_wineries:,}")
    print(f"  Provinces : {num_provinces:,}")
    print(f"  Varieties : {num_varieties:,}")

    # 3. Generate initial text embeddings using TF-IDF + SVD
    print("Generating initial TF-IDF + SVD features for wine nodes...")
    t0 = time.time()
    df["doc_text"] = df.apply(
        lambda r: f"{r['variety']} {r['country']} {r['winery']} {r['description']}", axis=1
    )
    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    tfidf_mat = vectorizer.fit_transform(df["doc_text"])
    
    embedding_dim = 128
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    wine_init_features = svd.fit_transform(tfidf_mat)
    wine_init_features = torch.tensor(wine_init_features, dtype=torch.float32)
    print(f"  Initialized features in {time.time()-t0:.1f}s | Shape: {wine_init_features.shape}")

    # Save TF-IDF and SVD to align evaluation queries
    import pickle
    with open(cfg.RESULTS / "gnn_tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(cfg.RESULTS / "gnn_svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    print("  Saved TF-IDF and SVD objects to results/")

    # 4. Define PyTorch LightGCN model for attribute-link prediction
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running GNN on device: {device}")

    # Embeddings for attributes
    emb_wine = nn.Embedding(num_wines, embedding_dim).to(device)
    emb_winery = nn.Embedding(num_wineries, embedding_dim).to(device)
    emb_province = nn.Embedding(num_provinces, embedding_dim).to(device)
    emb_variety = nn.Embedding(num_varieties, embedding_dim).to(device)

    # Initialize with SVD / average SVD
    with torch.no_grad():
        emb_wine.weight.copy_(wine_init_features)
        
        # Attribute embeddings initialized as average of connected wine embeddings
        winery_avg = torch.zeros(num_wineries, embedding_dim)
        winery_cnt = torch.zeros(num_wineries, 1)
        winery_avg.index_add_(0, winery_ids, wine_init_features)
        winery_cnt.index_add_(0, winery_ids, torch.ones(num_wines, 1))
        emb_winery.weight.copy_(winery_avg / (winery_cnt + 1e-9))

        prov_avg = torch.zeros(num_provinces, embedding_dim)
        prov_cnt = torch.zeros(num_provinces, 1)
        prov_avg.index_add_(0, province_ids, wine_init_features)
        prov_cnt.index_add_(0, province_ids, torch.ones(num_wines, 1))
        emb_province.weight.copy_(prov_avg / (prov_cnt + 1e-9))

        var_avg = torch.zeros(num_varieties, embedding_dim)
        var_cnt = torch.zeros(num_varieties, 1)
        var_avg.index_add_(0, variety_ids, wine_init_features)
        var_cnt.index_add_(0, variety_ids, torch.ones(num_wines, 1))
        emb_variety.weight.copy_(var_avg / (var_cnt + 1e-9))

    # Adjacency tensors on device
    w_ids = winery_ids.to(device)
    p_ids = province_ids.to(device)
    v_ids = variety_ids.to(device)

    def run_lightgcn():
        # Layer 0 embeddings
        e_w = emb_wine.weight
        e_winery = emb_winery.weight
        e_prov = emb_province.weight
        e_var = emb_variety.weight

        # Message Passing Layer 1: Attribute node aggregates wine nodes
        # (This is implicitly done by our initialization and regular training step)
        
        # Message Passing Layer 1: Wine node aggregates attributes
        e_w_1 = (e_winery[w_ids] + e_prov[p_ids] + e_var[v_ids]) / 3.0

        # Layer combination (residual link)
        final_wine_emb = (e_w + e_w_1) / 2.0
        return final_wine_emb

    # Optimizer
    optimizer = optim.Adam(
        list(emb_wine.parameters()) + 
        list(emb_winery.parameters()) + 
        list(emb_province.parameters()) + 
        list(emb_variety.parameters()), 
        lr=0.005
    )

    # 5. Training Loop using self-supervised BPR link prediction
    epochs = 12
    batch_size = 4096
    print("Training LightGCN model via self-supervised Link Prediction...")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_wines)
        epoch_loss = 0.0
        
        # Training batches
        for i in range(0, num_wines, batch_size):
            optimizer.zero_grad()
            batch_w_idx = perm[i : i + batch_size].to(device)
            
            # Forward: get wine embeddings after convolution
            final_wine = run_lightgcn()
            w_emb = final_wine[batch_w_idx]

            # Positive attributes
            pos_winery = emb_winery(w_ids[batch_w_idx])
            pos_prov = emb_province(p_ids[batch_w_idx])
            pos_var = emb_variety(v_ids[batch_w_idx])

            # Negative attributes (random sampling)
            neg_winery_idx = torch.randint(0, num_wineries, (len(batch_w_idx),), device=device)
            neg_prov_idx = torch.randint(0, num_provinces, (len(batch_w_idx),), device=device)
            neg_var_idx = torch.randint(0, num_varieties, (len(batch_w_idx),), device=device)

            neg_winery = emb_winery(neg_winery_idx)
            neg_prov = emb_province(neg_prov_idx)
            neg_var = emb_variety(neg_var_idx)

            # Compute BPR Loss
            loss_winery = -torch.log(torch.sigmoid(torch.sum(w_emb * pos_winery, dim=-1) - torch.sum(w_emb * neg_winery, dim=-1)) + 1e-9).mean()
            loss_prov = -torch.log(torch.sigmoid(torch.sum(w_emb * pos_prov, dim=-1) - torch.sum(w_emb * neg_prov, dim=-1)) + 1e-9).mean()
            loss_var = -torch.log(torch.sigmoid(torch.sum(w_emb * pos_var, dim=-1) - torch.sum(w_emb * neg_var, dim=-1)) + 1e-9).mean()

            loss = loss_winery + loss_prov + loss_var
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_w_idx)

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss/num_wines:.4f}")

    # Generate final embeddings
    with torch.no_grad():
        final_wine_embeddings = run_lightgcn().cpu().numpy()
    
    # Save GNN embeddings for evaluation baseline use
    embeddings_file = cfg.RESULTS / "gnn_wine_embeddings.npy"
    np.save(str(embeddings_file), final_wine_embeddings)
    print(f"  Saved trained GNN embeddings to: {embeddings_file}")

    # 6. Hierarchical K-Means clustering
    print("Performing Hierarchical K-Means clustering (10 x 10 x 10)...")
    t0 = time.time()
    
    # Level 1 clustering
    kmeans_l1 = KMeans(n_clusters=10, random_state=42, n_init=10)
    l1_labels = kmeans_l1.fit_predict(final_wine_embeddings)

    l2_labels = np.zeros(num_wines, dtype=int)
    l3_labels = np.zeros(num_wines, dtype=int)

    for i1 in range(10):
        mask1 = (l1_labels == i1)
        sub_emb1 = final_wine_embeddings[mask1]
        if len(sub_emb1) < 10:
            l2_labels[mask1] = np.arange(len(sub_emb1))
            continue
        
        # Level 2 clustering inside cluster i1
        kmeans_l2 = KMeans(n_clusters=10, random_state=42, n_init=5)
        l2_lbls = kmeans_l2.fit_predict(sub_emb1)
        l2_labels[mask1] = l2_lbls

        for i2 in range(10):
            mask2 = mask1 & (l2_labels == i2)
            sub_emb2 = final_wine_embeddings[mask2]
            if len(sub_emb2) < 10:
                l3_labels[mask2] = np.arange(len(sub_emb2))
                continue
            
            # Level 3 clustering inside cluster i1-i2
            kmeans_l3 = KMeans(n_clusters=10, random_state=42, n_init=5)
            l3_lbls = kmeans_l3.fit_predict(sub_emb2)
            l3_labels[mask2] = l3_lbls

    # Construct IDs
    gnn_ids = [f"{l1}-{l2}-{l3}" for l1, l2, l3 in zip(l1_labels, l2_labels, l3_labels)]
    df["GNN_Semantic_ID"] = gnn_ids
    df["Semantic_ID"] = df.apply(make_semantic_id, axis=1) # standard id for backup

    # Output catalog mapping
    mapping_df = df[["Semantic_ID", "GNN_Semantic_ID", "title", "country", "variety"]]
    mapping_file = cfg.RESULTS / "gnn_semantic_ids.csv"
    mapping_df.to_csv(str(mapping_file), index=False)
    print(f"  Clustering completed in {time.time()-t0:.1f}s")
    print(f"  Saved GNN Semantic ID mapping file to: {mapping_file}")

    print("\nSample GNN Semantic IDs generated:")
    print(mapping_df.head(10).to_string())
    print("\n✅ GNN-guided indexing complete!")

if __name__ == "__main__":
    main()
