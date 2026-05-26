"""
figures/figure6_resilience_variance.py — Figure 6 (SRQ2): DP Resilience Variance Plot.

Answers SRQ1: how much does DP stochasticity destabilise training?

Single panel:
  x-axis: ε (log scale)
  y-axis: std(AUC) across seeds — variance inflation index
  One line per task (IHM, Decomp, Pheno)

Usage:
    python figures/figure6_resilience_variance.py \
        --input results/privacy_utility_combined.csv \
        --output plots/Figure6_SQR2.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":    12,
    "axes.titleweight":  "normal",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
})

# Brand palette — matches plot_results_summary.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

TASK_COLS = {
    "IHM":    ("val_ihm_auroc",         _C[1]),   # purple
    "Decomp": ("val_decomp_auroc",      _C[2]),   # dark purple/navy
    "Pheno":  ("val_pheno_macro_auroc", _C[3]),   # dark red
}
EPS_ORDER  = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = ["0.5", "1", "2", "5", "10", "∞"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/privacy_utility_combined.csv")
    parser.add_argument("--output", default="Manuscript/figures/Figure6_SQR2.png")
    parser.add_argument("--mode",   default="uniform",
                        help="Which DP mode to plot (uniform or stratified)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["epsilon_level"] = pd.to_numeric(df["epsilon_level"], errors="coerce").fillna(float("inf"))

    df = df.groupby(["mode", "epsilon_level", "seed"]).last().reset_index()
    df = df[df["mode"] == args.mode]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for task_name, (task_col, color) in TASK_COLS.items():
        x_plot, y_std = [], []
        for eps in EPS_ORDER:
            sub = df[df["epsilon_level"] == eps][task_col]
            if sub.empty:
                continue
            x_plot.append(EPS_ORDER.index(eps))
            y_std.append(float(sub.std()))

        ax.plot(x_plot, y_std, color=color, marker="o", ms=4, linewidth=1.4,
                label=task_name)

    ax.set_xticks(range(len(EPS_ORDER)))
    ax.set_xticklabels(EPS_LABELS)
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Std(AUC-ROC) across seeds")
    ax.set_title("AUC-ROC standard deviation across seeds as a function of privacy budget ε")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=800, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
