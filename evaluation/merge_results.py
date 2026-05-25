"""
merge_results.py
================
Merge baseline results with LLM results from Colab
and print the final comparison table (also exports LaTeX).

Usage:
    python3 merge_results.py
    python3 merge_results.py --llm results/llm_eval_results.csv   # real Colab results
"""

import sys, os; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1])); import config as cfg


import argparse
import os
import sys
import numpy as np
import pandas as pd
import re
import json

# ─── Parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--baseline",      default=str(cfg.BASELINE_CSV))
parser.add_argument("--llm",           default=str(cfg.RESULTS / "llm_eval_results.csv"),
                    help="Per-query LLM results CSV from Colab")
parser.add_argument("--out",           default=str(cfg.FINAL_CSV))
parser.add_argument("--add_mock_llm",  action="store_true",
                    help="Append a clearly labelled estimated LLM row; do not use for the main result table")
args, _ = parser.parse_known_args()

# ─── Load baseline results ────────────────────────────────────────────────────
print("="*65)
print("  Wine Recommendation - Final Results Aggregation")
print("  Dataset: Kaggle Wine Reviews")
print("  Cite  : https://www.kaggle.com/datasets/zynicide/wine-reviews")
print("="*65)

summaries = {}

if os.path.exists(args.baseline):
    base_df = pd.read_csv(args.baseline, index_col=0)
    for method in base_df.index:
        summaries[method] = base_df.loc[method].to_dict()
    print(f"\n[OK] Loaded baseline results: {list(base_df.index)}")
else:
    print(f"\n[WARN] {args.baseline} not found - run baseline_eval.py first.")

# ─── Load LLM results ─────────────────────────────────────────────────────────
if os.path.exists(args.llm):
    llm_df = pd.read_csv(args.llm)
    K = [1, 5, 10]
    llm_summary = {}

    # ── Recall@K / NDCG@K ──────────────────────────────────────────────────
    # If beam search was used (num_beams > 1), the CSV will have proper Recall@5,
    # Recall@10 columns computed from the ranked list of candidates.
    # If only greedy (num_beams=1), only Recall@1 = ExactMatch is meaningful.
    has_beam_recall = any(f"Recall@{k}" in llm_df.columns for k in K)

    if has_beam_recall:
        for k in K:
            if f"Recall@{k}" in llm_df.columns:
                llm_summary[f"Recall@{k}"] = llm_df[f"Recall@{k}"].mean()
            if f"NDCG@{k}" in llm_df.columns:
                llm_summary[f"NDCG@{k}"] = llm_df[f"NDCG@{k}"].mean()
        print(f"   [INFO] Recall@K read from beam-search results (proper multi-candidate eval).")
    else:
        # Legacy / greedy-only: only Recall@1 = EM is valid
        if "ExactMatch" in llm_df.columns:
            em_val = llm_df["ExactMatch"].mean()
            llm_summary["Recall@1"] = em_val
            llm_summary["NDCG@1"]   = em_val
            # Mark @5 and @10 as NaN — they are NOT meaningful without beam search
            llm_summary["Recall@5"]  = float("nan")
            llm_summary["Recall@10"] = float("nan")
            llm_summary["NDCG@5"]    = float("nan")
            llm_summary["NDCG@10"]   = float("nan")
            print(f"   [WARN] Greedy/single-shot mode - Recall@5/@10 set to NaN (not meaningful).")
            print(f"          Re-run with --num_beams 10 to get proper Recall@5/@10.")

    # ── ExactMatch ──────────────────────────────────────────────────────────
    if "ExactMatch" in llm_df.columns:
        llm_summary["ExactMatch(EM)"] = llm_df["ExactMatch"].mean()

    # ── MRR ─────────────────────────────────────────────────────────────────
    if "MRR" in llm_df.columns:
        llm_summary["MRR"] = llm_df["MRR"].mean()
    elif "Recall@1" in llm_summary and not has_beam_recall:
        llm_summary["MRR"] = llm_summary["Recall@1"]  # greedy: MRR = Recall@1

    # ── Intent / Country / Variety match ─────────────────────────────────
    for col in ["IntentMatch@1", "IntentMatch@5", "IntentMatch@10",
                "CountryMatch@1", "VarietyMatch@1", "VintageMatch@1"]:
        if col in llm_df.columns:
            llm_summary[col] = llm_df[col].mean()

    # ── Text quality ──────────────────────────────────────────────────────
    if "ROUGE_L" in llm_df.columns:
        llm_summary["ROUGE_L"] = llm_df["ROUGE_L"].mean()
    if "BERTScore_F1" in llm_df.columns:
        bs = llm_df["BERTScore_F1"].dropna()
        llm_summary["BERTScore_F1"] = float(bs.mean()) if len(bs) > 0 else float("nan")
    if "latency_ms" in llm_df.columns:
        llm_summary["Latency_ms"] = llm_df["latency_ms"].mean()

    llm_summary["ResultType"] = "real_colab"
    summaries["LLM-LoRA (Proposed)"] = llm_summary
    print(f"[OK] Loaded LLM results: {len(llm_df):,} samples")
    r1_pct  = llm_summary.get("Recall@1", 0) * 100
    r10_pct = llm_summary.get("Recall@10", float("nan"))
    im_pct  = llm_summary.get("IntentMatch@1", 0) * 100
    em_pct  = llm_summary.get("ExactMatch(EM)", 0) * 100
    print(f"   ExactMatch  : {em_pct:.2f}%")
    print(f"   Recall@1    : {r1_pct:.2f}%")
    r10_str = f"{r10_pct*100:.2f}%" if pd.notna(r10_pct) else "NaN (need --num_beams 10)"
    print(f"   Recall@10   : {r10_str}")
    print(f"   IntentMatch@1: {im_pct:.2f}%")


else:
    if args.add_mock_llm:
        # Estimated LLM-LoRA row based on published DSI/GR benchmarks.
        # It is deliberately labelled and separated from the real proposed row.
        summaries["LLM-LoRA (Estimated - not main)"] = {
            "Recall@1":  0.142, "Recall@5":  0.348, "Recall@10": 0.478,
            "NDCG@1":    0.142, "NDCG@5":    0.238, "NDCG@10":   0.279,
            "MRR":       0.228, "ExactMatch(EM)": 0.142,
            "IntentMatch@1": 0.821, "IntentMatch@5": 0.943, "IntentMatch@10": 0.967,
            "ROUGE_L":   0.312, "BERTScore_F1": None, "Latency_ms": 245.0,
            "ResultType": "estimated_not_for_main_table",
        }
        print(f"[WARN] Added labelled ESTIMATED LLM row (--add_mock_llm).")
        print(f"   Main table still needs real Colab results: python3 evaluation/merge_results.py --llm results/llm_eval_results.csv")
    else:
        # Insert placeholder for LLM (to be filled from Colab)
        summaries["LLM-LoRA (Proposed)"] = {
            "Recall@1": None, "Recall@5": None, "Recall@10": None,
            "NDCG@1": None,   "NDCG@5": None,   "NDCG@10": None,
            "MRR": None,      "ExactMatch(EM)": None,
            "IntentMatch@1": None, "IntentMatch@10": None,
            "ROUGE_L": None,  "BERTScore_F1": None, "Latency_ms": None,
            "ResultType": "pending_real_colab_eval",
        }
        print(f"[WARN] {args.llm} not found - placeholder row added.")
        print(f"   Run the Colab evaluation, save results/llm_eval_results.csv, then rerun this script.")

# ─── Build dataframe ──────────────────────────────────────────────────────────
result_df = pd.DataFrame(summaries).T
result_df.index.name = "Method"

# ─── Print comparison table ───────────────────────────────────────────────────
DISPLAY_COLS = [
    "Recall@1", "Recall@10",
    "NDCG@5",
    "MRR",
    "ExactMatch(EM)", "IntentMatch@1", "IntentMatch@10",
    "ROUGE_L", "BERTScore_F1",
    "Latency_ms",
]
display_cols = [c for c in DISPLAY_COLS if c in result_df.columns]

print("\n" + "-"*90)
print(f"{'Method':<25}" + "".join(f"{c:>13}" for c in display_cols))
print("-"*90)
for method, row in result_df[display_cols].iterrows():
    is_proposed = "Proposed" in str(method) and result_df.at[method, "ResultType"] == "real_colab" if "ResultType" in result_df.columns else "Proposed" in str(method)
    prefix = ">> " if is_proposed else "   "
    vals = "".join(
        f"{v:>13.4f}" if pd.notna(v) else f"{'  -  ':>13}"
        for v in row
    )
    print(f"{prefix}{method:<23}" + vals)
print("-"*90)

# ─── Save CSV ─────────────────────────────────────────────────────────────────
result_df.to_csv(args.out)
print(f"\n[OK] Full table saved: {args.out}")

# ─── LaTeX output ────────────────────────────────────────────────────────────
LATEX_COLS = ["Recall@1", "Recall@10", "NDCG@5", "MRR", "IntentMatch@1", "IntentMatch@10", "ROUGE_L", "BERTScore_F1"]
latex_cols  = [c for c in LATEX_COLS if c in result_df.columns]

print("\n" + "-"*65)
print("LaTeX Table:")
print("-"*65)
col_header = " & ".join(["\\textbf{Method}"] + [f"\\textbf{{{c}}}" for c in latex_cols])
print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Comparison of wine recommendation methods on the")
print("  Kaggle Wine Reviews dataset~\\cite{wine_kaggle_2017}.}")
print("\\label{tab:main_results}")
print("\\begin{tabular}{l" + "r"*len(latex_cols) + "}")
print("\\hline")
print(col_header + " \\\\")
print("\\hline")
for method, row in result_df[latex_cols].iterrows():
    is_proposed = "Proposed" in str(method) and result_df.at[method, "ResultType"] == "real_colab" if "ResultType" in result_df.columns else "Proposed" in str(method)
    m_str = f"\\textbf{{{method}}}" if is_proposed else method
    vals  = " & ".join(
        f"{v:.4f}" if pd.notna(v) else "--" for v in row
    )
    print(f"{m_str} & {vals} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

print("\n% BibTeX:")
print("@misc{wine_kaggle_2017,")
print("  author={Aeberhard, Stefan and Forina, M.},")
print("  title={{Wine Reviews}},")
print("  year={2017}, publisher={Kaggle},")
print("  url={https://www.kaggle.com/datasets/zynicide/wine-reviews}}")
