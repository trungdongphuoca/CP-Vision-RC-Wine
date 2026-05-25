"""
run_full_eval.py
================
Pipeline chính thức - chạy đánh giá toàn bộ 12,044 mẫu test trên GPU.
Tối ưu hoàn chỉnh cho RTX 3060 12GB:
  - TF32 Tensor Core acceleration (Ampere GPU)
  - cuDNN auto-tune benchmark
  - Auto-detect best batch_size (thử 16 -> OOM fallback 8 -> 4)
  - model.eval() disabled dropout

Các bước:
  Step 1: Base RAG evaluation (Llama-3-8B không có LoRA)
  Step 2: LLM-LoRA evaluation (Llama-3-8B + LoRA fine-tuned)
  Step 3: Merge results vào final_comparison.csv
  Step 4: Vẽ biểu đồ so sánh DPI=300 cho báo cáo

Usage:
    python run_full_eval.py
"""
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

ROOT   = Path(__file__).parent.resolve()
PYTHON = str(Path(sys.executable))

# ── Thông số: bắt đầu ở mức cao nhất, tự giảm nếu OOM ───────────────────────
EVAL_SIZE      = 12044   # Toàn bộ tập test (bản chính thức)
BATCH_SIZE     = 128     # MAX! auto-fallback: 128->64->32->16 neu OOM
MAX_NEW_TOKENS = 150     # Đủ để sinh đầy đủ <thought> + Semantic ID

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def fmt_eta(seconds):
    return str(timedelta(seconds=int(seconds)))

def run_step(name, cmd, cwd=ROOT):
    log("=" * 65)
    log(f"  STARTING: {name}")
    log(f"  CMD: {' '.join(cmd)}")
    log("=" * 65)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f"  DONE: {name} in {fmt_eta(elapsed)}", level="OK ")
    else:
        log(f"  FAILED: {name} (exit {result.returncode})", level="ERR")
        sys.exit(result.returncode)
    log("=" * 65 + "\n")
    return elapsed

# ─────────────────────────────────────────────────────────────────────────────
log("=" * 65)
log("  WINE RECOMMENDATION — OFFICIAL FULL EVALUATION")
log("  GPU    : NVIDIA GeForce RTX 3060 12GB (Ampere)")
log(f"  Samples: {EVAL_SIZE:,} (full test set - ban chinh thuc)")
log(f"  Batch  : {BATCH_SIZE} (auto-fallback: 16->8->4 if OOM)")
log(f"  Tokens : {MAX_NEW_TOKENS}")
log("  OPT    : TF32 Tensor Cores + cuDNN benchmark + model.eval()")
log("=" * 65)

start_total = time.time()

# ── Step 0: Re-evaluate ALL traditional baselines (full dataset) ──────────────
step0_time = run_step(
    "Step 0/4 - Traditional Baselines (BM25/TF-IDF/Struct-Filter/GNN) - FULL DATA",
    [
        PYTHON, "evaluation/baseline_eval.py",
        "--eval_size", str(EVAL_SIZE),
    ]
)

# ── Step 1: Base RAG (no LoRA) ────────────────────────────────────────────────
step1_time = run_step(
    "Step 1/4 - Base RAG Evaluation (Llama-3-8B, NO LoRA)",
    [
        PYTHON, "evaluation/base_rag_eval.py",
        "--eval_size",      str(EVAL_SIZE),
        "--batch_size",     str(BATCH_SIZE),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--update_baseline",
    ]
)

# ── Step 2: LLM-LoRA Proposed ─────────────────────────────────────────────────
step2_time = run_step(
    "Step 2/4 - LLM-LoRA Fine-tuned Evaluation (Proposed Model)",
    [
        PYTHON, "evaluation/llm_eval.py",
        "--eval_size",      str(EVAL_SIZE),
        "--batch_size",     str(BATCH_SIZE),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
    ]
)

# ── Step 3: Merge results ─────────────────────────────────────────────────────
step3_time = run_step(
    "Step 3/4 - Merging all results -> final_comparison.csv",
    [PYTHON, "evaluation/merge_results.py"]
)

# ── Step 4: Plot high-res figures for report ──────────────────────────────────
step4_time = run_step(
    "Step 4/4 - Generating report figures (DPI=300)",
    [PYTHON, "evaluation/plot_results.py", "--dpi", "300"]
)

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.time() - start_total
log("=" * 65)
log("  FULL EVALUATION COMPLETE - BAN CHINH THUC")
log(f"  Step 0 Baselines : {fmt_eta(step0_time)}")
log(f"  Step 1 Base RAG  : {fmt_eta(step1_time)}")
log(f"  Step 2 LLM-LoRA  : {fmt_eta(step2_time)}")
log(f"  Step 3 Merge     : {step3_time:.1f}s")
log(f"  Step 4 Plot      : {step4_time:.1f}s")
log(f"  TOTAL TIME       : {fmt_eta(total_time)}")
log(f"  Results -> results/final_comparison.csv")
log(f"  Figures -> results/figures/ (DPI=300, ready for report)")
log("=" * 65)
