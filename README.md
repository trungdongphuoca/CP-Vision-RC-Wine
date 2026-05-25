# Generative Retrieval for Explainable Wine Recommendation using LLMs

> **Đề tài:** Generative Retrieval for Explainable Wine Recommendation using Large Language Models  
> **Dataset:** [Kaggle Wine Reviews 130k](https://www.kaggle.com/datasets/zynicide/wine-reviews) — Aeberhard, S. (2017)

---

## Kiến trúc hệ thống

```
[A] Data Preparation
    winemag-data-130k-v2.csv → Semantic ID → group split JSONL
    └── data_prep.py

[B] LLM Fine-Tuning (Llama-3-8B + LoRA)
    wine_train_130k.jsonl → lora_wine_model/
    └── fine_tune.py  |  Wine_Finetune_Colab.ipynb

[C] Baselines
    BM25 + TF-IDF CF + Base RAG (no LoRA)
    └── baseline_eval.py  |  base_rag_eval.py

[D] RAG + XAI Pipeline
    ChromaDB + FastAPI + SHAP Explanation
    └── inference_rag.py  |  xai_shap.py

[E] Evaluation
    EM, ROUGE-L, BERTScore → final_comparison.csv
    └── Wine_Evaluate_Colab.ipynb  |  merge_results.py
```

---

## Trạng thái hiện tại (Tuần 17/18)

| Module | File | Status |
|---|---|---|
| A. Data Prep | `data_prep.py` | ✅ Semantic_ID group split — no train/test ID overlap |
| B. Fine-Tune | `src/fine_tune.py` + `models/lora_wine_model/` | ✅ LoRA r=16 script; rerun after new group split |
| C1. BM25 | `evaluation/baseline_eval.py` | ✅ Group split Recall@10=0.187, MRR=0.075 |
| C2. TF-IDF CF | `evaluation/baseline_eval.py` | ✅ Group split Recall@10=0.141, MRR=0.056 |
| C3. Base RAG | `evaluation/base_rag_eval.py` | ✅ EM=0.00% (mock smoke test; GPU needed for real) |
| D. RAG+XAI | `src/inference_rag.py` + `src/xai_shap.py` | ✅ SHAP integrated |
| E. LLM Eval | `notebooks/Wine_Evaluate_Colab.ipynb` | 🔴 **TODO — chạy real eval trên Colab GPU** |

---

## Kết quả so sánh hiện tại (1000 samples)

| Method | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | MRR | ROUGE-L | BERTScore |
|---|---|---|---|---|---|---|---|
| BM25 | 0.0400 | 0.1140 | **0.1870** | 0.0774 | 0.0751 | — | — |
| TF-IDF CF | 0.0290 | 0.0950 | 0.1410 | 0.0606 | 0.0557 | — | — |
| Base RAG (no LoRA) | 0.0000 | — | — | — | — | 0.1245 | — |
| **LLM-LoRA (Proposed)** | **Pending real Colab eval** | — | — | — | — | **Pending** | **Pending** |

> **Luận điểm:** Nếu `LLM-LoRA Recall@1 > BM25 Recall@10` (0.187 trên group split) → single-shot GR vượt top-10 keyword retrieval

---

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Chạy hệ thống

### 1. Chuẩn bị dữ liệu
```bash
python3 src/data_prep.py
# Output: data/processed/wine_{train,val,test}_130k.jsonl
```

### 2. Fine-tuning (cần GPU — chạy trên Colab)
```bash
# Upload notebooks/Wine_Finetune_Colab.ipynb lên Google Colab
# Tải về: lora_wine_model.zip → giải nén vào models/lora_wine_model/
```

### 3. Baseline evaluation
```bash
python3 evaluation/baseline_eval.py          # BM25 + TF-IDF CF
python3 evaluation/base_rag_eval.py --mock   # Base RAG mock smoke test
# Output: results/baseline_comparison.csv
```

### 4. Demo API
```bash
cd src && uvicorn inference_rag:app --reload --port 8080
# UI:     http://localhost:8080
# Health: http://localhost:8080/health
# SHAP:   POST http://localhost:8080/explain
```

### 5. Đánh giá LLM (Colab GPU)
```
Upload lên Colab:
  - notebooks/Wine_Evaluate_Colab.ipynb
  - data/processed/wine_test_130k.jsonl
  - results/baseline_comparison.csv
  - models/lora_wine_model.zip

Tải về → đặt vào results/:
  - llm_eval_results.csv
```

### 6. Tổng hợp kết quả cuối
```bash
python3 evaluation/merge_results.py --llm results/llm_eval_results.csv
# → In bảng so sánh 4 methods + LaTeX table
# → Lưu: results/final_comparison.csv
```

> Không dùng `--add_mock_llm` cho bảng chính. Tuỳ chọn này chỉ tạo dòng
> `LLM-LoRA (Estimated - not main)` để minh hoạ khi chưa có kết quả Colab thật.

### 7. Test heuristic SHAP attribution standalone
```bash
python3 src/xai_shap.py
# → Benchmark 3 queries, in heuristic SHAP values + latency
```

### 8. Chạy test
```bash
pytest
```

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/` | UI demo (chat interface) |
| POST | `/recommend` | Wine recommendation + SHAP XAI |
| POST | `/explain` | Standalone SHAP explanation |
| GET | `/health` | System status (LLM, ChromaDB, SHAP) |

### Ví dụ response `/recommend`
```json
{
  "type": "recommendation",
  "message": "I recommend this bold Cabernet...",
  "retrieved_wine": {
    "title": "Jordan 2016 Cabernet Sauvignon",
    "country": "US", "variety": "Cabernet Sauvignon", "price": 47.0
  },
  "xai_explanation": {
    "attribution_type": "heuristic_feature_attribution",
    "score_model": "weighted_features_v1",
    "feature_names": ["price_match", "style_match", "pairing_match", "region_match", "semantic_sim"],
    "shap_values": [0.045, 0.082, 0.031, 0.150, 0.0],
    "base_value": 0.325,
    "explanation_text": "↑ region_match: +0.150\n↑ style_match: +0.082\n↑ price_match: +0.045"
  }
}
```

---

## XAI — Heuristic SHAP Feature Attribution

SHAP (SHapley Additive exPlanations) được dùng để giải thích đóng góp của 5 feature
trong một hàm điểm heuristic minh bạch. Đây là **heuristic feature attribution**,
không phải giải thích trực tiếp trạng thái nội bộ của LLM hoặc ChromaDB.

**5 features được phân tích:**

| Feature | Ý nghĩa | Công thức |
|---|---|---|
| `price_match` | Budget compatibility | min(q_price, w_price) / max(...) |
| `style_match` | Style keyword overlap | Jaccard(query words, variety words) |
| `pairing_match` | Food pairing score | # food keywords matched / total |
| `region_match` | Country mentioned | 1.0 if country in query else 0 |
| `semantic_sim` | Embedding similarity | cosine(query_emb, wine_emb) |

```bash
# Test heuristic SHAP standalone
python3 src/xai_shap.py
```

---

## Semantic ID Design

```
FORMAT: COUNTRY(4) - PROVINCE(4) - VARIETY(4) - YEAR(4)

US-NAPA-CABE-2015  → Napa Valley Cabernet Sauvignon 2015
FRAN-BORD-REDB-2018 → Bordeaux Red Blend 2018
ITAL-TUSC-SANG-2016 → Tuscany Sangiovese 2016
ARGE-MEND-MALB-2015 → Mendoza Malbec 2015
```

**Thống kê:**
- Total wines: 129,907
- Unique Semantic IDs: ~13,245
- Avg wines per ID: ~9.8 (multiple vintages/producers per style cluster)

---

## BibTeX

```bibtex
@misc{wine_kaggle_2017,
  author    = {Aeberhard, Stefan and Forina, M.},
  title     = {{Wine Reviews}},
  year      = {2017},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/zynicide/wine-reviews}
}
```

---

## Cấu trúc thư mục

```
Code LLM Recommend System/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                        ← centralized path constants
├── reorganize.py                    ← script đã dùng để sắp xếp
│
├── src/                             ← source code chính
│   ├── data_prep.py                   Tạo Semantic ID, split train/val/test
│   ├── fine_tune.py                   LoRA fine-tuning Llama-3-8B
│   ├── inference_rag.py               FastAPI server + SHAP integration
│   └── xai_shap.py                    SHAP XAI module (5 features)
│
├── evaluation/                      ← scripts đánh giá
│   ├── baseline_eval.py               BM25 + TF-IDF CF evaluation
│   ├── base_rag_eval.py               Ablation: Base RAG không LoRA
│   └── merge_results.py               Gộp results → LaTeX table
│
├── notebooks/                       ← Colab notebooks
│   ├── Wine_Finetune_Colab.ipynb      Fine-tuning trên GPU
│   ├── Wine_Evaluate_Colab.ipynb      Evaluation (1000 samples)
│   ├── generate_eval_notebook.py      Generator script cho eval notebook
│   └── generate_notebook.py           Generator script cho fine-tune notebook
│
├── data/
│   ├── raw/winemag-data-130k-v2.csv   Kaggle Wine Reviews (129,907 records)
│   └── processed/
│       ├── wine_train_130k.jsonl       103,925 training samples (80%)
│       ├── wine_val_130k.jsonl          12,991 validation samples (10%)
│       └── wine_test_130k.jsonl         12,991 test samples (10%)
│
├── results/                         ← evaluation outputs
│   ├── baseline_comparison.csv        BM25 + TF-IDF CF + Base RAG metrics
│   ├── final_comparison.csv           Bảng so sánh tổng hợp
│   ├── bm25_per_query.csv             BM25 per-query results
│   └── tfidf_per_query.csv            TF-IDF per-query results
│
├── models/                          ← model artifacts (.gitignored)
│   └── lora_wine_model/               LoRA adapters (Llama-3-8B)
│
├── api/static/                      ← web interface (chat UI)
└── chroma_db/                       ← ChromaDB vector store (.gitignored)
```
