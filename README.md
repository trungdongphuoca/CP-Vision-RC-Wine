# CP-Vision-RC-Wine

Ứng dụng Thị giác máy tính vào Hệ thống Gợi ý Rượu vang Đa phương thức  
**Computer Vision Applications in Multimodal Wine Recommender Systems**

Chuyên đề 2 — Trường Đại học Tôn Đức Thắng, Khoa CNTT  
Học viên: Trần Thành Trung | GVHD: TS. Bùi Quy Anh

---

## Tổng quan

Hệ thống cho phép người dùng chụp ảnh nhãn chai rượu vang bằng điện thoại để nhận được thông tin sản phẩm và danh sách gợi ý phù hợp theo hành vi cá nhân. Pipeline gồm 5 giai đoạn liên tiếp:

1. **YOLOv11** — Phát hiện và cắt vùng nhãn chai trong ảnh thực tế
2. **Cylindrical Unwarping** — Phẳng hóa hình học nhãn cong do thân chai hình trụ
3. **Florence-2** — OCR trích xuất thông tin và mô tả phong cách nhãn
4. **CLIP + FAISS** — So khớp đa phương thức, gợi ý thay thế khi hết hàng
5. **CW-EASE^R+IPS** — Bộ gợi ý backend cá nhân hóa với khử thiên kiến phổ biến

---

## Kết quả chính

**OCR & Nhận diện nhãn chai (Florence-2 + Cylindrical Unwarping)**

| Cấu hình | F1 Variety | F1 Country |
|:---|:---:|:---:|
| Ảnh cong gốc (Warped) | 0.7333 | 0.6667 |
| Ảnh sau phẳng hóa (Unwarped) | **1.0000** | **1.0000** |

Thời gian xử lý end-to-end trên CPU: **3.85 giây** (giảm từ 20 giây ban đầu)

**Gợi ý backend — Tập X-Wines (509,757 users test)**

| Mô hình | Recall@10 | nDCG@10 | MRR |
|:---|:---:|:---:|:---:|
| User-KNN | 0.92% | 0.46% | — |
| MF-SVD | 2.10% | 1.05% | 0.0402 |
| Popularity-CF | 11.20% | 6.50% | 0.1542 |
| EASE^R (Steck, 2019) | 8.823% | 9.523% | — |
| **CW-EASE^R+IPS (Proposed)** | **8.928%** | **9.621%** | **0.2285** |

> Popularity-CF có Recall@10 cao hơn nhờ khai thác thiên lệch phân phối (power-law), nhưng thua trên nDCG@10 (−47.9%) và MRR (−32.5%). Xem phân tích trong [bao_cao_toan_dien.md](thesis/bao_cao_toan_dien.md).

**Cold-start (người dùng < 5 tương tác, n=238,895)**

| Mô hình | Recall@10 | nDCG@10 |
|:---|:---:|:---:|
| EASE^R baseline | 8.36% | 6.00% |
| CW-EASE^R+IPS | **8.48%** | **6.05%** |

Kiểm định Wilcoxon signed-rank: p ≪ 0.001

---

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Yêu cầu: Python 3.10+, PyTorch 2.x, CUDA (tùy chọn — pipeline chạy được trên CPU)

---

## Chạy Demo

```bash
# Khởi động demo server (Flask, port 5050)
python demo/app.py

# Truy cập: http://localhost:5050
# Upload ảnh nhãn chai → nhận thông tin + gợi ý
```

Demo đi kèm 15 nhãn chai curated trong `demo/curated_demo_labels/` để test nhanh.

---

## Đánh giá

### OCR & Computer Vision
```bash
# Đánh giá trên tập curated (n=15)
python evaluation/cv_eval.py

# Đánh giá thực tế quy mô lớn (n=1007 ảnh)
python evaluation/cv_real_eval.py

# Ablation: so sánh warped vs unwarped
python evaluation/cv_ablation.py
```

### Bộ gợi ý backend
```bash
# CW-EASE^R+IPS (mô hình đề xuất)
python evaluation/cw_ease_eval.py

# Baseline: Cornac (User-KNN, SVD, VAECF, ...)
python evaluation/cornac_eval.py

# Tổng hợp kết quả
python evaluation/merge_results.py
```

### Phân tích thống kê
```bash
python evaluation/statistical_analysis.py   # Wilcoxon signed-rank test
python evaluation/plot_results.py           # Sinh biểu đồ vào results/figures/
```

Log đầy đủ quá trình chạy lưu tại `results/logs/`.

---

## Cấu trúc dự án

```
CP-Vision-RC-Wine/
│
├── src/                        # Source code chính
│   ├── cv_wine.py              #   Pipeline thị giác: YOLO + Unwarping + Florence-2
│   ├── inference_rag.py        #   FastAPI server + RAG Sommelier
│   ├── data_prep.py            #   Chuẩn bị dữ liệu, Semantic ID
│   ├── build_semantic_ids.py   #   Xây dựng Semantic ID phân cấp
│   ├── fine_tune.py            #   LoRA fine-tuning (Llama-3-8B)
│   ├── gnn_retrieval.py        #   GNN-based retrieval
│   ├── visual_semantic_fusion.py #  CLIP + FAISS fusion
│   ├── vlm_parser.py           #   Phân tích kết quả Florence-2
│   ├── xai_shap.py             #   SHAP feature attribution
│   └── attention_xai.py        #   Attention-based XAI
│
├── evaluation/                 # Scripts đánh giá
│   ├── cw_ease_eval.py         #   CW-EASE^R+IPS (mô hình đề xuất)
│   ├── cornac_eval.py          #   Baseline CF (KNN, SVD, VAECF...)
│   ├── cv_eval.py              #   OCR evaluation (curated set)
│   ├── cv_real_eval.py         #   OCR evaluation (large-scale)
│   ├── cv_ablation.py          #   Ablation warped vs unwarped
│   ├── baseline_eval.py        #   BM25, TF-IDF, content-based
│   ├── statistical_analysis.py #   Kiểm định thống kê Wilcoxon
│   ├── plot_results.py         #   Sinh biểu đồ
│   ├── eval_logger.py          #   Structured JSON evaluation logger
│   └── merge_results.py        #   Tổng hợp kết quả
│
├── demo/                       # Demo application
│   ├── app.py                  #   Flask server
│   ├── index.html              #   Giao diện chính
│   └── curated_demo_labels/    #   15 nhãn chai curated để demo
│
├── api/static/                 # Web UI (chat interface)
├── models/                     # Model weights (.gitignored phần lớn)
│   ├── yolo11_wine.pt          #   YOLOv11 fine-tuned (wine labels)
│   └── catalog_clip_embeddings.pt # CLIP embeddings catalog
│
├── results/
│   ├── figures/                #   Biểu đồ kết quả (28 PNG)
│   ├── logs/                   #   Evaluation logs đầy đủ
│   ├── cw_ease_eval_results.csv
│   └── cornac_eval_results.csv
│
├── data/
│   ├── processed/              #   Catalog, Semantic ID mapping
│   └── cv_ground_truth/        #   Ground truth cho OCR evaluation
│
├── scripts/                    # Utility scripts
├── notebooks/                  # Colab notebooks
├── synthetic_dataset/          # Dataset tổng hợp cho YOLO training
├── tests/                      # Unit tests
└── config.py                   # Cấu hình đường dẫn tập trung
```

---

## Dữ liệu

| Dataset | Mô tả | Nguồn |
|:---|:---|:---|
| **X-Wines** | 21M ratings, 56K wines, 1M+ users | [Kaggle](https://www.kaggle.com/datasets/cbddo/xwines) |
| **WineReviews 130K** | 130K mô tả hương vị (Winemag) | [Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews) |
| **Sapo** | Dữ liệu bán hàng thực tế tại Việt Nam | Nội bộ (không công khai) |
| **XWines-1K Labels** | 1,007 ảnh nhãn chai thực tế | Trích từ X-Wines |

Dữ liệu thô không được đưa lên repository. Xem `scripts/import_xwines.py` để tải về.

---

## Các mô hình bên ngoài (cần tải thủ công)

| Mô hình | Mục đích | Nguồn |
|:---|:---|:---|
| `microsoft/Florence-2-large` | OCR + captioning | HuggingFace |
| `openai/clip-vit-large-patch14` | Visual embeddings | HuggingFace |
| `YOLOv11n` (pretrained) | Object detection backbone | Ultralytics |

Fine-tuned weights (`yolo11_wine.pt`, `catalog_clip_embeddings.pt`) có trong `models/`.

---

## Tài liệu tham khảo chính

- Steck, H. (2019). Embarrassingly Shallow Autoencoders for Sparse Data. *WWW 2019*
- Schnabel et al. (2016). Recommendations as Treatments: Debiasing Learning and Evaluation. *ICML 2016*
- Xiao et al. (2024). Florence-2: Advancing a Unified Representation for Vision Tasks. *arXiv 2311.06242*
- Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML 2021*
- Johnson et al. (2019). Billion-scale Similarity Search with GPUs. *IEEE Trans. Big Data*

---

## Ghi chú

- Toàn bộ pipeline có thể chạy trên **CPU** (không cần GPU). Dùng GPU để tăng tốc Florence-2.
- Dữ liệu Sapo và luận văn không được đưa lên repository do tính bảo mật.
- Evaluation logs đầy đủ (không tóm tắt) lưu tại `results/logs/`.
