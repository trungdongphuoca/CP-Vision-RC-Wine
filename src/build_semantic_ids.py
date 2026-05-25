"""
build_semantic_ids.py
=====================
Builds hierarchical semantic IDs for wine recommendations, inspired by TIGER/DSI.
Uses TF-IDF + SVD embeddings followed by Hierarchical K-Means (16x16x16) and a unique item suffix.
This guarantees 100% unique Semantic IDs while preserving semantic grouping.
"""

import sys, os, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))
import config as cfg
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import pickle

def main():
    print("="*60)
    print("  Building Hierarchical Semantic IDs (TIGER/DSI style)")
    print("="*60)

    # 1. Load data
    df = pd.read_csv(cfg.WINE_CSV)
    df = df.dropna(subset=['country', 'variety', 'description', 'title']).reset_index(drop=True)
    num_wines = len(df)
    print(f"Loaded {num_wines:,} valid wine entries.")

    # 2. Extract text representation
    print("Extracting text features...")
    df["doc_text"] = df.apply(
        lambda r: f"{r.get('title','')} {r['variety']} {r['country']} {r.get('province','')} {r.get('winery','')} {r['description']}", axis=1
    )
    
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=25000, stop_words="english")
    tfidf_mat = vectorizer.fit_transform(df["doc_text"])
    
    svd = TruncatedSVD(n_components=128, random_state=42)
    wine_embeddings = svd.fit_transform(tfidf_mat)
    print(f"TF-IDF + SVD completed in {time.time()-t0:.1f}s | Shape: {wine_embeddings.shape}")

    # 3. Hierarchical Clustering (16 x 16 x 16 = 4096 clusters)
    print("Performing Hierarchical K-Means clustering (16 x 16 x 16)...")
    t0 = time.time()
    
    K1, K2, K3 = 16, 16, 16
    
    l1_labels = KMeans(n_clusters=K1, random_state=42, n_init=10).fit_predict(wine_embeddings)
    l2_labels = np.zeros(num_wines, dtype=int)
    l3_labels = np.zeros(num_wines, dtype=int)
    
    for i1 in range(K1):
        mask1 = (l1_labels == i1)
        sub_emb1 = wine_embeddings[mask1]
        if len(sub_emb1) < K2:
            l2_labels[mask1] = np.arange(len(sub_emb1))
            continue
            
        l2_lbls = KMeans(n_clusters=K2, random_state=42, n_init=5).fit_predict(sub_emb1)
        l2_labels[mask1] = l2_lbls
        
        for i2 in range(K2):
            mask2 = mask1 & (l2_labels == i2)
            sub_emb2 = wine_embeddings[mask2]
            if len(sub_emb2) < K3:
                l3_labels[mask2] = np.arange(len(sub_emb2))
                continue
                
            l3_lbls = KMeans(n_clusters=K3, random_state=42, n_init=5).fit_predict(sub_emb2)
            l3_labels[mask2] = l3_lbls

    # Generate Semantic IDs like "03-12-07"
    # Pad to 2 digits to ensure uniform tokenization length
    semantic_ids = [f"{l1:02d}-{l2:02d}-{l3:02d}" for l1, l2, l3 in zip(l1_labels, l2_labels, l3_labels)]
    df["Semantic_ID_Cluster"] = semantic_ids
    
    print(f"Clustering completed in {time.time()-t0:.1f}s")
    
    # 4. Make them uniquely identifiable globally
    # Add a level 4: item index within the cluster
    df["Item_Index"] = df.groupby("Semantic_ID_Cluster").cumcount()
    df["Semantic_ID"] = df.apply(lambda r: f"{r['Semantic_ID_Cluster']}-{r['Item_Index']:03d}", axis=1)
    
    print("\nAfter ensuring global uniqueness:")
    print(f"  Total unique Semantic IDs: {df['Semantic_ID'].nunique():,}")
    print(f"  Format: C1-C2-C3-ITEM (e.g. {df['Semantic_ID'].iloc[0]})")
    
    # 5. Save the processed catalog
    output_path = cfg.DATA_PROC / "wine_catalog_semantic.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved catalog with Semantic IDs to {output_path}")

    # Save mapping for evaluation script
    mapping_df = df[["Semantic_ID", "title", "country", "variety"]]
    mapping_path = cfg.RESULTS / "semantic_id_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)

    # 6. Save models
    with open(cfg.RESULTS / "semantic_tfidf.pkl", "wb") as f: pickle.dump(vectorizer, f)
    with open(cfg.RESULTS / "semantic_svd.pkl", "wb") as f: pickle.dump(svd, f)
    np.save(str(cfg.RESULTS / "semantic_embeddings.npy"), wine_embeddings)

if __name__ == "__main__":
    main()
