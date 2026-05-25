#!/usr/bin/env bash
# =============================================================================
# run_report.sh — Chạy toàn bộ pipeline báo cáo (Tuần 3–9)
# =============================================================================
# Usage:
#   bash run_report.sh           # full pipeline
#   bash run_report.sh --quick   # skip baseline_eval (dùng kết quả cũ)
#   bash run_report.sh --demo    # chỉ chạy demo scripts
# =============================================================================

set -euo pipefail

# ─── Detect python ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    echo "ERROR: Python not found"; exit 1
fi

# ─── Args ─────────────────────────────────────────────────────────────────────
QUICK=false
DEMO_ONLY=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
        --demo)  DEMO_ONLY=true ;;
    esac
done

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
info() { echo -e "  ${CYAN}▶${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
hdr()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"; \
         echo -e "${BOLD}  $1${RESET}"; \
         echo -e "${CYAN}══════════════════════════════════════════════════${RESET}"; }
timing(){ local end=$SECONDS; echo -e "  ${YELLOW}⏱  Thời gian: $((end-$1))s${RESET}"; }

# ─── HEADER ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   🍷  LLM Wine Recommendation System — Báo Cáo Môn Học  ║${RESET}"
echo -e "${BOLD}║       Chuyên Đề 03 | Tuần 3–9 Pipeline                  ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Python     : $($PY --version)"
echo -e "  Directory  : $SCRIPT_DIR"
echo -e "  Mode       : $([ "$QUICK" = true ] && echo 'Quick (skip baseline)' || echo 'Full')"
echo ""

TOTAL_START=$SECONDS

# ─── STEP 0: Kiểm tra dataset ────────────────────────────────────────────────
hdr "BƯỚC 0 — Kiểm tra dữ liệu"
if [ -f "data/raw/winemag-data-130k-v2.csv" ]; then
    ok "Dataset tìm thấy: data/raw/winemag-data-130k-v2.csv"
else
    warn "Không tìm thấy dataset!"
    echo "  Tải tại: https://www.kaggle.com/datasets/zynicide/wine-reviews"
    echo "  Đặt file vào: data/raw/winemag-data-130k-v2.csv"
    exit 1
fi

if [ "$DEMO_ONLY" = true ]; then
    # ─── DEMO ONLY MODE ───────────────────────────────────────────────────────
    hdr "TUẦN 5 — Demo Pipeline Cơ Bản"
    t=$SECONDS
    info "Chạy demo_pipeline.py --batch..."
    $PY demo_pipeline.py --batch --catalog_size 3000 --top_k 3
    timing $t

    hdr "TUẦN 7 — Cold-Start & Fallback Demo"
    t=$SECONDS
    info "Chạy demo_coldstart.py --demo_new..."
    $PY demo_coldstart.py --catalog_size 2000 --demo_new
    timing $t
else
    # ─── FULL PIPELINE ────────────────────────────────────────────────────────

    # Step 1: Data prep check
    hdr "BƯỚC 1 — Kiểm tra dữ liệu tiền xử lý (Tuần 4)"
    if [ -f "data/processed/wine_test_130k.jsonl" ]; then
        ok "Test data đã sẵn sàng: data/processed/wine_test_130k.jsonl"
    else
        info "Chạy data_prep.py để tạo training/test splits..."
        $PY src/data_prep.py || warn "data_prep.py thất bại — bỏ qua"
    fi

    # Step 2: Baseline evaluation
    if [ "$QUICK" = false ]; then
        hdr "BƯỚC 2 — Đánh giá Baseline: BM25 + TF-IDF (Tuần 3, 8)"
        t=$SECONDS
        info "Chạy baseline_eval.py (1000 mẫu)..."
        $PY evaluation/baseline_eval.py
        ok "Baseline evaluation hoàn thành!"
        timing $t
    else
        hdr "BƯỚC 2 — Bỏ qua Baseline (--quick mode)"
        if [ -f "results/baseline_comparison.csv" ]; then
            ok "Dùng kết quả cũ: results/baseline_comparison.csv"
        else
            warn "Không có kết quả cũ — chạy baseline_eval.py..."
            $PY evaluation/baseline_eval.py
        fi
    fi

    # Step 3: Base RAG ablation (mock — không cần GPU)
    hdr "BƯỚC 3 — Ablation Study: Base RAG without LoRA (Tuần 4, 8)"
    t=$SECONDS
    info "Chạy base_rag_eval.py --mock (không cần GPU)..."
    $PY evaluation/base_rag_eval.py --mock --eval_size 500
    ok "Base RAG ablation hoàn thành!"
    timing $t

    # Step 4: Demo pipeline (Tuần 5)
    hdr "BƯỚC 4 — Demo Pipeline Cơ Bản (Tuần 5)"
    t=$SECONDS
    info "Chạy demo_pipeline.py --batch..."
    $PY demo_pipeline.py --batch --catalog_size 5000 --top_k 3
    ok "Demo pipeline hoàn thành!"
    timing $t

    # Step 5: XAI benchmark (Tuần 6)
    hdr "BƯỚC 5 — XAI / SHAP Explainability Benchmark (Tuần 6)"
    t=$SECONDS
    info "Chạy xai_shap.py benchmark (3 queries)..."
    $PY src/xai_shap.py || warn "xai_shap.py: cần pip install shap"
    timing $t

    # Step 6: Cold-start demo (Tuần 7)
    hdr "BƯỚC 6 — Cold-Start & Fallback Demo (Tuần 7)"
    t=$SECONDS
    info "Chạy demo_coldstart.py..."
    $PY demo_coldstart.py --catalog_size 3000 --demo_new
    ok "Cold-start demo hoàn thành!"
    timing $t

    # Step 7: Merge results (Tuần 8–9)
    hdr "BƯỚC 7 — Tổng hợp bảng so sánh (Tuần 8–9)"
    t=$SECONDS
    info "Chạy merge_results.py (không tự thêm số LLM ước tính)..."
    $PY evaluation/merge_results.py
    ok "Bảng so sánh hoàn thành!"
    timing $t

    # Step 8: Plot results (Tuần 9)
    hdr "BƯỚC 8 — Vẽ biểu đồ so sánh (Tuần 9)"
    t=$SECONDS
    info "Chạy plot_results.py..."
    $PY evaluation/plot_results.py --dpi 150 || \
        warn "plot_results.py thất bại — cần: pip install matplotlib"
    timing $t
fi

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   ✓  Pipeline báo cáo hoàn thành!                       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Kết quả:${RESET}"
[ -f "results/baseline_comparison.csv" ] && \
    echo -e "  ${GREEN}✓${RESET} results/baseline_comparison.csv   (BM25, TF-IDF)"
[ -f "results/base_rag_results.csv" ] && \
    echo -e "  ${GREEN}✓${RESET} results/base_rag_results.csv      (Base RAG ablation)"
[ -f "results/final_comparison.csv" ] && \
    echo -e "  ${GREEN}✓${RESET} results/final_comparison.csv      (Bảng so sánh đầy đủ)"
[ -f "results/demo_pipeline_results.json" ] && \
    echo -e "  ${GREEN}✓${RESET} results/demo_pipeline_results.json"
[ -d "results/figures" ] && \
    echo -e "  ${GREEN}✓${RESET} results/figures/                  (Biểu đồ PNG)"

echo ""
echo -e "  ${BOLD}Thời gian tổng:${RESET} $((SECONDS-TOTAL_START))s"
echo ""
echo -e "  ${CYAN}Tuần 9 — Tiếp theo:${RESET}"
echo -e "  - Mở results/figures/ để xem biểu đồ"
echo -e "  - Copy LaTeX từ merge_results.py vào báo cáo"
echo -e "  - Nếu có GPU: chạy notebooks/Wine_Evaluate_Colab.ipynb trên Colab"
echo -e "    rồi chạy: python3 evaluation/merge_results.py --llm llm_eval_results.csv"
echo ""
