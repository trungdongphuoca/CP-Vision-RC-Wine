# CV Pipeline — Ablation Study Report

**Date**: 2026-06-11 23:30:54

## Configuration Descriptions

| Config | YOLO | OCR Engine | LLM Parser | Description |
|--------|------|-----------|------------|-------------|
| A | ✓ | EasyOCR | ✗ | Baseline with traditional OCR |
| B | ✓ | Florence-2 | ✗ | VLM-based OCR with YOLO |
| C | ✗ | Florence-2 | ✗ | VLM-based OCR on full image |
| D | ✓ | Florence-2 | ✓ | Full pipeline with LLM |
| E | ✗ | Florence-2 | ✓ | Full image VLM + LLM |

## Results Comparison

| Metric | Config A | Config B | Config C | Config D | Config E |
|--------|---------|---------|---------|---------|---------|
| **Macro F1** | 0.7257 | 0.8156 | 0.8156 | **1.0000** | **1.0000** |
| **Title F1** | 0.2952 | 0.4111 | 0.4111 | **1.0000** | **1.0000** |
| **Variety F1** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Country F1** | 0.6667 | 0.6667 | 0.6667 | **1.0000** | **1.0000** |
| **Year EM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Price Match** | 0.6667 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **CER ↓** | 0.6523 | **0.5819** | **0.5819** | **0.5819** | **0.5819** |
| **Detection Rate** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Avg Latency (ms)** | 1885.0 | 1143.7 | **980.9** | 1048.0 | 1039.5 |
| **P95 Latency (ms)** | 2582.0 | 1362.2 | **1021.1** | 1111.2 | 1050.5 |

## Component Contributions (Macro F1 Δ)

| Component | Comparison | Δ Macro F1 | Interpretation |
|-----------|-----------|-----------|----------------|
| Yolo Contribution | Config B (YOLO+Florence) vs Config C (Florence only) | +0.0000 | Improvement from adding YOLO region detection |
| Florence Vs Easyocr | Config B (YOLO+Florence) vs Config A (YOLO+EasyOCR) | +0.0899 | Improvement from Florence-2 over EasyOCR |
| Llm Parser Contribution | Config D (YOLO+Florence+LLM) vs Config B (YOLO+Florence) | +0.1844 | Improvement from adding LLM parsing |
| Full Pipeline Vs Minimal | Config D (full) vs Config A (minimal) | +0.2743 | Total improvement: full pipeline vs minimal |
| Yolo With Llm | Config D (YOLO+Florence+LLM) vs Config E (Florence+LLM) | +0.0000 | YOLO contribution when LLM parser is present |

## Summary

**Best Configuration**: Config D (YOLO + Florence-2 + LLM) — Macro F1 = 1.0000
