# CV Pipeline — Real Image Evaluation Report

**Date**: 2026-06-18 13:28:30
**Samples**: 3 (Real: 0, Synthetic: 3)

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.5933 |
| Title Token F1 | 0.3000 |
| Variety Token F1 | 1.0000 |
| Country Token F1 | 0.6667 |
| Year Exact Match | 1.0000 |
| Price Match (±15%) | 0.0000 |
| CER (lower=better) | 0.4122 |
| Detection Rate | 1.0000 |

## Latency

| Stat | ms |
|------|-----|
| Average | 1776.6 |
| P50 (Median) | 1491.1 |
| P95 | 2423.9 |
| Max | 2527.5 |

## Pipeline Info

- YOLO detection used: 0.0%
- Average confidence: 0.8167

## Per-Sample Results

| Image | Title F1 | Variety F1 | Country F1 | Year EM | Price | CER | Latency (ms) |
|-------|----------|-----------|------------|---------|-------|-----|---------------|
| image_001 (S) | 0.50 | 1.00 | 0.00 | 1 | 0 | 0.461 | 2528 |
| image_002 (S) | 0.00 | 1.00 | 1.00 | 1 | 0 | 0.436 | 1311 |
| image_003 (S) | 0.40 | 1.00 | 1.00 | 1 | 0 | 0.340 | 1491 |

*(S) = Synthetic image (no real photo available)*
