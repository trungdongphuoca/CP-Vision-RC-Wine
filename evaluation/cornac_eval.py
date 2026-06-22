"""
evaluation/cornac_eval.py — Collaborative Filtering Evaluation Suite
====================================================================
Evaluates 5 classical CF models and 3 Deep Learning CF models on the
X-Wines dataset using the Cornac recommendation framework.

Inputs:
  - Test Split: 20% random split from data/xwines/XWines_Test_1K_ratings.csv
  - rating_threshold = 4.0 (for binarizing ranking metrics)
  - user_based = True (averages metrics across users)

Metrics computed:
  - Rating Error: MAE, RMSE
  - Ranking Metrics: Precision@k, Recall@k, F1-Score@k, NDCG@k for k in [5, 10, 50, 100]

Usage:
  python evaluation/cornac_eval.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Path setup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cornac
from cornac.models import UserKNN, ItemKNN, SVD, NMF, MostPop, VAECF, BiVAECF, RecVAE
from cornac.metrics import MAE, RMSE, Precision, Recall, FMeasure, NDCG

def main():
    print("=" * 70)
    print("  Cornac Collaborative Filtering Evaluation (X-Wines Dataset)")
    print("  Config: user_based=True | rating_threshold=4.0")
    print("=" * 70)

    # 1. Load data
    ratings_path = os.path.join(ROOT, "data", "xwines", "XWines_Slim_150K_ratings.csv")
    if not os.path.exists(ratings_path):
        print(f"ERROR: {ratings_path} not found. Ensure the dataset is downloaded first.")
        sys.exit(1)

    df = pd.read_csv(ratings_path, low_memory=False)
    print(f"Loaded {len(df):,} ratings from {ratings_path}")
    print(f"Unique Users : {df['UserID'].nunique()}")
    print(f"Unique Items : {df['WineID'].nunique()}")

    # Convert to Cornac format (user_id, item_id, rating)
    data = list(df[['UserID', 'WineID', 'Rating']].itertuples(index=False, name=None))

    # 2. Setup Train/Test Split (80/20)
    ratio_split = cornac.eval_methods.RatioSplit(
        data=data,
        test_size=0.2,
        rating_threshold=4.0,
        exclude_unknowns=True,
        seed=42,
        verbose=False
    )

    # 3. Instantiate Models
    # 5 Classical Models
    models_classical = [
        UserKNN(k=20, similarity="cosine", name="User-KNN"),
        ItemKNN(k=20, similarity="cosine", name="Item-KNN"),
        SVD(k=10, max_iter=30, learning_rate=0.01, seed=42, name="MF-SVD"),
        NMF(k=10, max_iter=30, seed=42, name="MF-NMF"),
        MostPop(name="Popularity-CF")
    ]

    # 3 Deep Learning Models (PyTorch-based)
    models_dl = [
        VAECF(k=10, autoencoder_structure=[20], n_epochs=30, seed=42, use_gpu=True, name="VAECF (DL)"),
        BiVAECF(k=10, encoder_structure=[20], n_epochs=30, seed=42, use_gpu=True, name="BiVAECF (DL)"),
        RecVAE(hidden_dim=32, latent_dim=10, n_epochs=30, seed=42, use_gpu=True, name="RecVAE (DL)")
    ]

    all_models = models_classical + models_dl

    # 4. Instantiate Metrics
    k_values = [5, 10, 50, 100]
    metrics = [MAE(), RMSE()]

    for k in k_values:
        metrics.extend([
            Precision(k=k),
            Recall(k=k),
            FMeasure(k=k),
            NDCG(k=k)
        ])

    # 5. Run Experiment
    print(f"\nRunning experiment on {len(all_models)} models...")
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=all_models,
        metrics=metrics,
        user_based=True
    )
    exp.run()

    # 6. Parse and Save Results
    results_records = []
    for model_res in exp.result:
        row = {"Model": model_res.model_name}
        for metric in metrics:
            row[metric.name] = model_res.metric_avg_results.get(metric.name)
        results_records.append(row)

    results_df = pd.DataFrame(results_records)
    
    # Save to CSV
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "cornac_eval_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Evaluation results saved to: {out_csv}")

    # Generate Markdown Table
    print("\n" + "=" * 80)
    print("  COLLABORATIVE FILTERING METRICS SUMMARY (user_based=True)")
    print("=" * 80)
    
    # Let's print a subset of metrics for readability
    print_cols = ["Model", "MAE", "RMSE"]
    for k in [5, 10]:
        print_cols.extend([f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"])

    # Rename columns in df to match standard format
    col_mapping = {
        "MAE": "MAE",
        "RMSE": "RMSE"
    }
    for k in k_values:
        col_mapping[f"Precision@{k}"] = f"Precision@{k}"
        col_mapping[f"Recall@{k}"] = f"Recall@{k}"
        col_mapping[f"F1@{k}"] = f"F1-Score@{k}"
        col_mapping[f"NDCG@{k}"] = f"nDCG@{k}"

    results_df = results_df.rename(columns=col_mapping)
    
    # Filter columns that exist
    actual_print_cols = [c for c in print_cols if c in results_df.columns]
    print(results_df[actual_print_cols].to_markdown(index=False))
    print("=" * 80)

    # Save complete markdown report
    report_path = os.path.join(ROOT, "results", "cornac_eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Đánh giá Collaborative Filtering trên X-Wines (Cornac)\n\n")
        f.write(f"*Thời gian thực hiện: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("## 1. Phương pháp Đánh giá\n")
        f.write("- **Tập dữ liệu**: `XWines_Slim_150K_ratings.csv` (150,000 ratings từ 10,561 người dùng trên 1,007 sản phẩm).\n")
        f.write("- **Phân chia dữ liệu**: RatioSplit (80% Train, 20% Test).\n")
        f.write("- **Ngưỡng nhị phân (Binarization Threshold)**: >= 4.0 sao.\n")
        f.write("- **Chế độ tính toán**: `user_based = True` (tính toán sai số và chỉ số xếp hạng của từng người dùng trước, sau đó lấy trung bình cộng).\n\n")
        f.write("## 2. Bảng kết quả Đánh giá đầy đủ\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n*Chú thích: Các cột chỉ số học sâu (DL) đều được thực hiện tăng tốc phần cứng bằng GPU.*")
    
    print(f"[OK] Markdown report saved to: {report_path}")

if __name__ == "__main__":
    main()
