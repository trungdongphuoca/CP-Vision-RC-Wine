"""
plot_results.py — Tuần 9: Tổng hợp kết quả & vẽ biểu đồ so sánh
=================================================================
Đọc results/final_comparison.csv và vẽ bộ biểu đồ:
  1. Recall@K grouped bar chart
  2. NDCG@K grouped bar chart
  3. MRR + IntentMatch@1/@10 comparison
  4. Latency comparison (ms)
  5. Summary radar / spider chart

Xuất ảnh PNG chất lượng cao vào results/figures/

Usage:
    python3 evaluation/plot_results.py
    python3 evaluation/plot_results.py --add_mock_llm  # thêm dòng ước tính, có label rõ
    python3 evaluation/plot_results.py --dpi 300       # chất lượng ảnh cao hơn
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

mpl_config_dir = cfg.RESULTS / ".matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",          default=str(cfg.FINAL_CSV))
parser.add_argument("--add_mock_llm", action="store_true",
                    help="Insert a clearly labelled estimated LLM-LoRA row")
parser.add_argument("--dpi",          type=int, default=150)
parser.add_argument("--style",        default="dark_background",
                    help="matplotlib style (dark_background | seaborn-v0_8 | ggplot)")
args = parser.parse_args()

# ─── Matplotlib setup ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Run: pip install matplotlib")

# ─── Load / build results ─────────────────────────────────────────────────────
def load_results(csv_path, add_mock_llm=False):
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run evaluation/baseline_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col=0)

    if add_mock_llm:
        # Insert labelled estimates only when explicitly requested.
        llm_row = {
            "Recall@1":  0.142,  "Recall@5":  0.348, "Recall@10": 0.478,
            "NDCG@1":    0.142,  "NDCG@5":    0.238, "NDCG@10":   0.279,
            "IntentMatch@1": 0.821, "IntentMatch@5": 0.943, "IntentMatch@10": 0.967,
            "MRR":       0.228,
            "ExactMatch(EM)": 0.142,
            "ROUGE_L":   0.312,
            "Latency_ms": 245.0,
            "CountryMatch@1": 0.912, "VarietyMatch@1": 0.843,
            "ResultType": "estimated_not_for_main_table",
        }
        estimated_label = "LLM-LoRA (Estimated - not main)"
        if estimated_label in df.index:
            for col, val in llm_row.items():
                if col in df.columns:
                    df.at[estimated_label, col] = val
        else:
            row_df = pd.DataFrame([llm_row], index=[estimated_label])
            df = pd.concat([df, row_df])

    # Keep only main methods in order
    method_order = [
        "Random Baseline",
        "Popularity-Based",
        "BM25",
        "BM25+ Enhanced",
        "TF-IDF CF",
        "TF-IDF + LSA",
        "Hybrid BM25+TF-IDF",
        "Struct-Filter TF-IDF",
        "Struct-Filter BM25",
        "Struct-Filter+Price",
        "GNN-Filter",
        "Base RAG (no LoRA)",
        "LLM-LoRA (Proposed)",
        "LLM-LoRA (Estimated - not main)",
    ]
    df = df.reindex([m for m in method_order if m in df.index])
    return df

# ─── Color palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "Random Baseline"     : "#90A4AE",
    "Popularity-Based"    : "#CE93D8",
    "BM25"                : "#4FC3F7",
    "BM25+ Enhanced"      : "#0288D1",
    "TF-IDF CF"           : "#81C784",
    "TF-IDF + LSA"        : "#2E7D32",
    "Hybrid BM25+TF-IDF"  : "#FFF176",
    "Struct-Filter TF-IDF": "#FF8A65",   # orange — improved
    "Struct-Filter BM25"  : "#E64A19",   # deep orange — improved
    "Struct-Filter+Price" : "#FF1744",   # red — best retrieval
    "GNN-Filter"          : "#AB47BC",   # purple — new GNN representation
    "Base RAG (no LoRA)"  : "#FFB74D",
    "LLM-LoRA (Proposed)" : "#F06292",
    "LLM-LoRA (Estimated - not main)": "#B0BEC5",
}
HATCH = {
    "Random Baseline"     : "xx",
    "Popularity-Based"    : "--",
    "BM25"                : "",
    "BM25+ Enhanced"      : "",
    "TF-IDF CF"           : "",
    "TF-IDF + LSA"        : "...",
    "Hybrid BM25+TF-IDF"  : "+",
    "Struct-Filter TF-IDF": "",
    "Struct-Filter BM25"  : "",
    "Struct-Filter+Price" : "",
    "GNN-Filter"          : "\\\\",
    "Base RAG (no LoRA)"  : "///",
    "LLM-LoRA (Proposed)" : "**",
    "LLM-LoRA (Estimated - not main)": "xx",
}

def get_color(method): return PALETTE.get(method, "#90A4AE")
def get_hatch(method): return HATCH.get(method, "")

# ─── Plot helpers ─────────────────────────────────────────────────────────────
def apply_style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

def grouped_bar(ax, df, cols, title, ylabel="Score"):
    """
    Draw grouped bar chart for given columns.
    NaN → no bar drawn, shows small grey 'N/A' annotation (metric not applicable).
    Used so Base RAG @5/@10 appear blank while LLM-LoRA still shows its bars.
    """
    methods = df.index.tolist()
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    n_groups = len(valid_cols)
    n_bars   = len(methods)
    width    = 0.8 / n_bars
    x        = np.arange(n_groups)

    for i, method in enumerate(methods):
        raw_vals = [df.at[method, c] if c in df.columns else np.nan
                    for c in valid_cols]
        offset = (i - n_bars/2 + 0.5) * width

        for j, (raw, col) in enumerate(zip(raw_vals, valid_cols)):
            xpos = x[j] + offset
            if pd.isna(raw):
                # Metric N/A for this method (single-shot — no @5/@10 ranking)
                ax.text(xpos, 0.006, "N/A", ha="center", va="bottom",
                        fontsize=5.2, color="#888888", rotation=90, style="italic")
            else:
                val = float(raw)
                ax.bar(xpos, val, width,
                       color=get_color(method), hatch=get_hatch(method),
                       alpha=0.88, edgecolor="white", linewidth=0.5)
                if val > 0:
                    ax.text(xpos, val + 0.005, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("@","@\n") for c in valid_cols], fontsize=8)
    apply_style(ax, title, ylabel=ylabel)
    all_non_nan = [float(df.at[m, c]) for m in methods for c in valid_cols
                   if c in df.columns and pd.notna(df.at[m, c])]
    ax.set_ylim(0, min(max(all_non_nan)*1.35 + 0.01, 1.05) if all_non_nan else 1.05)

def horizontal_bar(ax, df, col, title):
    """Horizontal bar for single metric."""
    methods = df.index.tolist()
    vals    = [float(df.at[m, col]) if (col in df.columns and pd.notna(df.at[m, col])) else 0.0
               for m in methods]
    colors  = [get_color(m) for m in methods]
    bars    = ax.barh(methods, vals, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8)
    apply_style(ax, title, xlabel=col)
    ax.set_xlim(0, max(vals)*1.25+0.01 if max(vals) else 1)
    ax.invert_yaxis()

def radar_chart(ax, df, metrics, title):
    """Radar / spider chart."""
    valid = [m for m in metrics if m in df.columns]
    if len(valid) < 3:
        ax.text(0.5, 0.5, "Not enough metrics", ha="center", va="center", transform=ax.transAxes)
        return
    N = len(valid)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [v.replace("@","@") for v in valid], size=8)

    for method in df.index:
        vals = [float(df.at[method, m]) if pd.notna(df.at[method, m]) else 0.0
                for m in valid]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=method, color=get_color(method))
        ax.fill(angles, vals, alpha=0.1, color=get_color(method))

    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Tuần 9 — Tổng hợp kết quả thực nghiệm")
    print("  Vẽ biểu đồ so sánh các phương pháp")
    print("="*60)

    df = load_results(args.csv, args.add_mock_llm)
    print(f"\n  Methods loaded : {list(df.index)}")
    print(f"  Columns        : {list(df.columns)}\n")

    # Print text table first
    display = ["Recall@1","Recall@5","Recall@10","NDCG@5","MRR","IntentMatch@1","IntentMatch@10","Latency_ms"]
    display = [c for c in display if c in df.columns]
    print(f"  {'Method':<25}" + "".join(f"{c:>13}" for c in display))
    print("  " + "─"*65)
    for m in df.index:
        is_prop = (
            "Proposed" in m
            and "ResultType" in df.columns
            and df.at[m, "ResultType"] == "real_colab"
        )
        tag = " ◄" if is_prop else ""
        row = "  " + f"{m:<25}" + "".join(
            f"{df.at[m,c]:>13.4f}" if c in df.columns and pd.notna(df.at[m,c]) else f"{'—':>13}"
            for c in display)
        print(row + tag)
    print()

    if not HAS_MPL:
        print("  matplotlib not available — skipping plot generation")
        return

    # ── Create figures directory ───────────────────────────────────────────────
    fig_dir = cfg.RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Main comparison (Recall + NDCG + MRR) ──────────────────────
    try:
        plt.style.use(args.style)
    except Exception:
        plt.style.use("default")

    fig = plt.figure(figsize=(24, 13))
    fig.suptitle(
        "Wine Recommendation System — Evaluation Results\n"
        "Dataset: Kaggle Wine Reviews (130k) | N=1,000 test queries",
        fontsize=14, fontweight="bold", y=0.99
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.30)

    # 1a. Recall@K
    ax1 = fig.add_subplot(gs[0, :2])
    grouped_bar(ax1, df, ["Recall@1","Recall@5","Recall@10"],
                "Recall@K — Tỉ lệ truy xuất đúng trong top-K", "Recall")

    # 1b. NDCG@K
    ax2 = fig.add_subplot(gs[1, :2])
    grouped_bar(ax2, df, ["NDCG@1","NDCG@5","NDCG@10"],
                "NDCG@K — Normalized Discounted Cumulative Gain", "NDCG")

    # 1c. IntentMatch@K
    ax3 = fig.add_subplot(gs[2, :2])
    grouped_bar(ax3, df, ["IntentMatch@1","IntentMatch@5","IntentMatch@10"],
                "IntentMatch@K — Country+Variety intent alignment", "IntentMatch")

    # 1d. MRR
    ax4 = fig.add_subplot(gs[0, 2])
    horizontal_bar(ax4, df, "MRR", "MRR\n(Mean Reciprocal Rank)")

    # 1e. Latency
    ax5 = fig.add_subplot(gs[1, 2])
    horizontal_bar(ax5, df, "Latency_ms", "Avg Latency\n(ms/query)")

    # 1f. Radar
    ax6 = fig.add_subplot(gs[2, 2], polar=True)
    radar_chart(ax6, df,
                ["Recall@1","Recall@10","NDCG@5","MRR","IntentMatch@1"],
                "Radar — Multi-metric")

    # Legend — 2 rows for 12 methods
    handles = [mpatches.Patch(facecolor=get_color(m), label=m, hatch=get_hatch(m),
                              alpha=0.88, edgecolor="white")
               for m in df.index]
    ncols = min(len(df.index), 6)
    fig.legend(handles=handles, loc="lower center", ncol=ncols,
               fontsize=8, framealpha=0.3, bbox_to_anchor=(0.5, 0.0))

    out1 = str(fig_dir / "fig1_main_comparison.png")
    fig.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out1}")

    # ── Figure 2: Latency breakdown ───────────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Latency & Efficiency Analysis", fontsize=12, fontweight="bold")

    # Bar latency
    methods = df.index.tolist()
    lats = [float(df.at[m,"Latency_ms"]) if "Latency_ms" in df.columns and pd.notna(df.at[m,"Latency_ms"])
            else 0 for m in methods]
    colors = [get_color(m) for m in methods]
    bars = axes[0].bar(methods, lats, color=colors, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, lats):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f"{val:.0f}ms", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Average Latency (ms/query)", fontweight="bold")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels([m.replace(" ","\n") for m in methods], fontsize=8)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    axes[0].yaxis.grid(True, alpha=0.3); axes[0].set_axisbelow(True)

    # Recall@1 vs Latency scatter
    rec1 = [float(df.at[m,"Recall@1"]) if "Recall@1" in df.columns and pd.notna(df.at[m,"Recall@1"])
            else 0 for m in methods]
    for m, lat, r in zip(methods, lats, rec1):
        axes[1].scatter(lat, r, s=180, color=get_color(m), label=m, zorder=3)
        axes[1].annotate(m.replace(" ","\n"), (lat, r),
                         textcoords="offset points", xytext=(5,5), fontsize=7)
    axes[1].set_title("Recall@1 vs Latency Trade-off", fontweight="bold")
    axes[1].set_xlabel("Latency (ms/query)"); axes[1].set_ylabel("Recall@1")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    axes[1].yaxis.grid(True, alpha=0.3); axes[1].xaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)

    out2 = str(fig_dir / "fig2_latency_analysis.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved: {out2}")

    # ── Figure 3: IntentMatch detailed breakdown ───────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("Intent Match Analysis — Country + Variety Alignment", fontsize=12, fontweight="bold")

    intent_cols_k1 = ["CountryMatch@1","VarietyMatch@1","IntentMatch@1"]
    valid_k1 = [c for c in intent_cols_k1 if c in df.columns]
    grouped_bar(axes3[0], df, valid_k1, "At K=1")

    intent_cols_k10 = ["CountryMatch@10","VarietyMatch@10","IntentMatch@10"]
    valid_k10 = [c for c in intent_cols_k10 if c in df.columns]
    grouped_bar(axes3[1], df, valid_k10, "At K=10")

    handles3 = [mpatches.Patch(facecolor=get_color(m), label=m, alpha=0.88)
                for m in df.index]
    fig3.legend(handles=handles3, loc="lower center", ncol=len(df.index),
                fontsize=9, framealpha=0.3, bbox_to_anchor=(0.5, 0.0))

    out3 = str(fig_dir / "fig3_intent_match.png")
    fig3.tight_layout(rect=[0, 0.08, 1, 1])
    fig3.savefig(out3, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig3)
    print(f"  ✓ Saved: {out3}")

    # ── Figure 4: LLM-LoRA Spotlight ──────────────────────────────────────────
    # Direct comparison: LLM-LoRA vs top-5 baselines.
    # Uses large, easy-to-read grouped bars with % labels.
    llm_key = "LLM-LoRA (Proposed)"
    spotlight_methods = [
        "BM25+ Enhanced",
        "Struct-Filter BM25",
        "Struct-Filter+Price",
        "GNN-Filter",
        "Base RAG (no LoRA)",
        llm_key,
    ]
    spotlight_methods = [m for m in spotlight_methods if m in df.index]

    if llm_key in df.index and len(spotlight_methods) >= 2:
        df4 = df.loc[spotlight_methods]

        metrics_4 = [
            ("Recall@1",       "Recall@1"),
            ("Recall@10",      "Recall@10"),
            ("NDCG@5",         "NDCG@5"),
            ("MRR",            "MRR"),
            ("IntentMatch@1",  "IntentMatch@1"),
            ("ROUGE_L",        "ROUGE-L"),
            ("BERTScore_F1",   "BERTScore F1"),
        ]
        # Keep only metrics that exist
        metrics_4 = [(c, lbl) for c, lbl in metrics_4 if c in df4.columns]

        n_metrics = len(metrics_4)
        n_methods = len(spotlight_methods)
        fig4, axes4 = plt.subplots(1, n_metrics,
                                   figsize=(3.2 * n_metrics, 6))
        if n_metrics == 1:
            axes4 = [axes4]

        fig4.suptitle(
            "LLM-LoRA (Proposed) vs Top Baselines\n"
            "Direct metric-by-metric comparison",
            fontsize=13, fontweight="bold"
        )

        bar_w = 0.7 / n_methods
        x_pos = np.arange(1)   # single group per subplot

        for ax_i, (col, label) in enumerate(metrics_4):
            ax = axes4[ax_i]
            for j, method in enumerate(spotlight_methods):
                raw = df4.at[method, col] if col in df4.columns else np.nan
                val = float(raw) if pd.notna(raw) else 0.0
                color = get_color(method)
                lw    = 2.5 if method == llm_key else 0.5
                ec    = "white" if method != llm_key else "#FFD700"
                offset = (j - n_methods/2 + 0.5) * bar_w
                bar = ax.bar(
                    x_pos + offset, [val], bar_w,
                    color=color, edgecolor=ec, linewidth=lw,
                    alpha=0.92, hatch=get_hatch(method),
                    label=method
                )
                # Percentage label on each bar
                if val > 0:
                    ax.text(
                        x_pos[0] + offset, val + 0.005,
                        f"{val*100:.1f}%",
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold" if method == llm_key else "normal",
                        color="#FFD700" if method == llm_key else "white",
                        rotation=90
                    )
                elif pd.isna(raw):
                    ax.text(x_pos[0]+offset, 0.01, "N/A",
                            ha="center", va="bottom", fontsize=6,
                            color="#888", rotation=90)

            ax.set_title(label, fontsize=10, fontweight="bold", pad=6)
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            all_vals = [float(df4.at[m, col]) for m in spotlight_methods
                        if col in df4.columns and pd.notna(df4.at[m, col])]
            ax.set_ylim(0, min(max(all_vals) * 1.45 + 0.01, 1.05) if all_vals else 1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
            # Gold star marker on LLM bar axis
            if method == llm_key:
                ax.spines["bottom"].set_color("#FFD700")
                ax.spines["bottom"].set_linewidth(2)

        # Shared legend
        handles4 = [mpatches.Patch(facecolor=get_color(m),
                                    edgecolor="#FFD700" if m == llm_key else "white",
                                    linewidth=2.5 if m == llm_key else 0.5,
                                    label=m, alpha=0.92)
                    for m in spotlight_methods]
        fig4.legend(handles=handles4, loc="lower center",
                    ncol=len(spotlight_methods), fontsize=9,
                    framealpha=0.3, bbox_to_anchor=(0.5, 0.0))

        fig4.tight_layout(rect=[0, 0.1, 1, 1])
        out4 = str(fig_dir / "fig4_llm_spotlight.png")
        fig4.savefig(out4, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig4)
        print(f"  ✓ Saved: {out4}")

    print(f"\n  ✓ Tất cả biểu đồ đã lưu vào: {fig_dir}")
    print(f"  Files:")
    for f in sorted(fig_dir.iterdir()):
        print(f"    - {f.name}")
    print()

if __name__ == "__main__":
    main()
