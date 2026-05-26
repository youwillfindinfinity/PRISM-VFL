"""
figures/figure5a_negative_transfer.py

Task × model heatmap of loss reduction vs. IHM-only single-task baseline.
Positive = MTL reduces loss (helps); negative = MTL increases loss (negative transfer).

Usage:
    python figures/figure5a_negative_transfer.py \
        --exp2 results/exp2.csv --output plots/Figure5a_TaskRelatedness.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

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
})

BRAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "brand_div", ["#8a3c48", "#f5f0ea", "#2f4a6d"]
)

CONFIG_ORDER  = ["all_tasks", "ihm_decomp", "ihm_pheno"]
CONFIG_LABELS = {"all_tasks": "All tasks", "ihm_decomp": "IHM+Decomp", "ihm_pheno": "IHM+Pheno"}


def build_loss_delta_matrix(df: pd.DataFrame) -> pd.DataFrame:
    final     = df.groupby(["task_config", "seed"]).last().reset_index()
    baseline  = final[final["task_config"] == "ihm_only"].set_index("seed")
    metrics   = ["ihm_loss", "decomp_loss", "pheno_loss"]
    col_labels = ["IHM Loss", "Decomp Loss", "Pheno Loss"]

    rows = []
    for cfg in CONFIG_ORDER:
        if cfg not in final["task_config"].values:
            continue
        grp = final[final["task_config"] == cfg].set_index("seed")
        row = {"task_config": CONFIG_LABELS[cfg]}
        for m in metrics:
            row[m] = round(float((baseline[m] - grp[m]).mean()), 4)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("task_config")
    result.columns = col_labels
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp2",   default="results/exp2.csv")
    parser.add_argument("--output", default="Manuscript/figures/Figure5a_TaskRelatedness.png")
    args = parser.parse_args()

    mat  = build_loss_delta_matrix(pd.read_csv(args.exp2))
    vmax = float(np.nanmax(np.abs(mat.values)))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(mat, annot=True, fmt=".3f", center=0,
                vmin=-vmax, vmax=vmax, cmap=BRAND_CMAP,
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Loss reduction relative to single-task IHM baseline per task configuration")
    ax.set_xlabel("")
    ax.set_ylabel("Model Configuration")
    ax.tick_params(axis="y", rotation=0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=800, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
