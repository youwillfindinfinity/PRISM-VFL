"""
figures/figS6_ablations.py — S6 Fig: Architectural ablation study.

Ablation study figure for the base PRISM architecture.
Horizontal lollipop chart: final-round validation metrics, mean +/- std across 3 seeds.
VFL-MTL (full system) shown as reference dashed vertical line; delta annotations vs. VFL-MTL.

Layout: 1 row x 3 columns -- IHM AUROC | Decomp AUROC | Pheno Macro-AUC

Usage:
    # validation (default)
    python figures/figS6_ablations.py \
        --input results/ablations.csv --output plots/S6_Ablations.png

    # test set
    python figures/figS6_ablations.py --source test \
        --input results/test_ablations.csv --output plots/ablations_test.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Display names and ordering
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    "VFL-MTL",
    "abl_no_mmoe",
    "abl_uniform_gating",
    "abl_experts_2",
    "abl_experts_8",
    "abl_embed_32",
    "abl_embed_128",
]

LABELS = {
    "VFL-MTL":            "PRISM (full)",
    "abl_no_mmoe":        "No MMoE (MLP)",
    "abl_uniform_gating": "Uniform Gating",
    "abl_experts_2":      "Experts = 2",
    "abl_experts_8":      "Experts = 8",
    "abl_embed_32":       "Embed = 32",
    "abl_embed_128":      "Embed = 128",
}

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

COLORS = {
    "VFL-MTL":            _C[0],
    "abl_no_mmoe":        _C[1],
    "abl_uniform_gating": _C[2],
    "abl_experts_2":      _C[3],
    "abl_experts_8":      _C[4],
    "abl_embed_32":       _C[5],
    "abl_embed_128":      _C[6],
}

# groups: [VFL-MTL] | [no_mmoe, uniform_gating] | [experts_2, experts_8] | [embed_32, embed_128]
GROUP_SPANS  = [(0, 0), (1, 2), (3, 4), (5, 6)]
GROUP_LABELS = ["Reference", "MMoE ablations", "Expert count", "Embed dim"]
GROUP_END    = [0, 2, 4]  # add gap in y after these model indices

VAL_METRICS = [
    ("val_ihm_auroc",         "IHM AUC-ROC",    "IHM AUC-ROC"),
    ("val_decomp_auroc",      "Decomp AUC-ROC",  "Decomp AUC-ROC"),
    ("val_pheno_macro_auroc", "Pheno Macro-AUC", "Pheno Macro-AUC"),
]

TEST_METRICS = [
    ("ihm_auc_roc",     "IHM AUC-ROC",    "IHM AUC-ROC"),
    ("decomp_auc_roc",  "Decomp AUC-ROC",  "Decomp AUC-ROC"),
    ("pheno_macro_auc", "Pheno Macro-AUC", "Pheno Macro-AUC"),
]


def _y_positions():
    GAP = 0.55
    y_raw = []
    y = 0.0
    for i in range(len(MODEL_ORDER)):
        y_raw.append(y)
        y += 1.0
        if i in GROUP_END:
            y += GAP
    top = y_raw[-1]
    return [top - r for r in y_raw]


def last_round(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["model", "seed"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def agg(df: pd.DataFrame, model_order: list, metric: str):
    g = df.groupby("model")[metric]
    mu = g.mean().reindex(model_order)
    sd = g.std().reindex(model_order).fillna(0)
    return mu, sd


def draw_panel(ax, mu, sd, metric, xlabel, title, ref_mu, show_yticks=True):
    y_pos = _y_positions()

    x_lo = max(0.0, float(mu.min()) - float(sd.max()) - 0.05)
    x_hi = float(mu.max()) + float(sd.max()) + 0.08

    for i, model in enumerate(MODEL_ORDER):
        val = float(mu.values[i])
        err = float(sd.values[i])
        yi  = y_pos[i]
        color = COLORS[model]

        # dot with horizontal error bar only (no stem)
        ax.errorbar(val, yi, xerr=err, fmt="o", color=color,
                    capsize=3, capthick=1.0, elinewidth=1.0, markersize=6, zorder=3)

        # annotation
        if model == "VFL-MTL":
            ax.text(val + err + 0.004, yi, f"{val:.3f}",
                    va="center", ha="left", fontsize=11, fontweight="bold", color="#444444")
        else:
            delta = val - ref_mu
            sign  = "+" if delta >= 0 else "-"
            ann_color = _C[4] if delta >= 0 else _C[1]
            ax.text(val + err + 0.004, yi, f"{sign}{abs(delta):.3f}",
                    va="center", ha="left", fontsize=11, fontweight="bold", color=ann_color)

    # reference vertical line at VFL-MTL mean
    ax.axvline(ref_mu, color=_C[0], linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    # group separator horizontal lines
    for gi in GROUP_END:
        sep = (y_pos[gi] + y_pos[gi + 1]) / 2
        ax.axhline(sep, color="#cccccc", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_yticks(y_pos)
    if show_yticks:
        ax.set_yticklabels([LABELS[m] for m in MODEL_ORDER], fontsize=7.5)
    else:
        ax.set_yticklabels([""] * len(MODEL_ORDER))
    ax.tick_params(left=False)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, pad=6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(min(y_pos) - 0.5, max(y_pos) + 0.5)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # group bracket labels on secondary right y-axis
    group_y = [(y_pos[s] + y_pos[e]) / 2 for s, e in GROUP_SPANS]
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(group_y)
    ax2.set_yticklabels(GROUP_LABELS, fontsize=7, color="#666666")
    ax2.tick_params(right=False, length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--source", choices=["val", "test"], default="val")
    args = parser.parse_args()

    is_test = (args.source == "test")

    if args.input is None:
        args.input  = "results/test_ablations.csv" if is_test else "results/ablations.csv"
    if args.output is None:
        args.output = "Manuscript/figures/ablations_test.png" if is_test else "Manuscript/figures/S6_Ablations.png"

    df = pd.read_csv(args.input)
    if not is_test:
        df = last_round(df)
    df = df[df["model"].isin(MODEL_ORDER)]

    metrics = TEST_METRICS if is_test else VAL_METRICS
    ref_row = df[df["model"] == "VFL-MTL"]

    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        11,
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize":   12,
        "axes.titleweight": "normal",
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "axes.linewidth":   0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=False)
    suptitle = (
        "Test-set AUC-ROC per architectural ablation variant"
        if is_test else
        "Final-round validation AUC-ROC per architectural ablation variant"
    )
    fig.suptitle(suptitle, fontweight="normal", y=1.02)

    for i, (ax, (metric, xlabel, title)) in enumerate(zip(axes, metrics)):
        mu, sd = agg(df, MODEL_ORDER, metric)
        ref_mu = ref_row[metric].mean()
        draw_panel(ax, mu, sd, metric, xlabel, title, ref_mu, show_yticks=(i == 0))

    fig.tight_layout(rect=[0, 0, 1, 1])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=800, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
