"""
figures/figure5b_scalability.py

Two-panel figure: AUC slopegraph (left) + training cost — convergence rounds
and total wall-clock time (right).

Sources:
  test_exp3.csv -- held-out test AUC metrics (n_sites = 2 and 3)
  Convergence rounds: hardcoded from exp3 run summary (3 seeds: 42, 123, 7)
    n=2: 60.3 ± 31.1 rounds
    n=3: 70.0 ± 10.01 rounds
  Per-round wall-clock: from exp4 Snellius timing
    n=2: 4.18 s/round  |  n=3: 5.54 s/round

Usage:
    python figures/figure5b_scalability.py \
        --test_exp3 results/test_exp3.csv \
        --output plots/Figure5b_Scalability.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Brand palette — matches resilience_variance.py and plot_ablations_dp.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

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
    "legend.fontsize":   9,
})

_METRICS = [
    ("ihm_auc_roc",     "IHM",    _C[1], 0.75),   # purple
    ("decomp_auc_roc",  "Decomp", _C[2], 0.70),   # dark charcoal
    ("pheno_macro_auc", "Pheno",  _C[3], 0.65),   # burgundy
]

# Convergence round stats from exp3 (seeds 42, 123, 7): (mean, std)
_CONV_ROUNDS = {2: (60.3, 31.1), 3: (70.0, 10.01)}
# Per-round wall-clock seconds from exp4 Snellius timing
_SEC_PER_ROUND = {2: 4.18, 3: 5.54}


def _plot_slopegraph(ax, test_df):
    xs = sorted(test_df["n_sites"].unique())
    x_left, x_right = 0, 1
    slope_xs = [x_left, x_right]
    all_auc = []

    for metric, label, color, floor in _METRICS:
        g = test_df.groupby("n_sites")[metric]
        mu = g.mean().reindex(xs)
        sd = g.std().fillna(0).reindex(xs)

        pts = [(xi, float(mu[x]), float(sd[x]))
               for xi, x in zip(slope_xs, xs) if not np.isnan(mu[x])]
        all_auc.extend(m for _, m, _ in pts)

        if len(pts) == 2:
            ax.fill_between(
                [p[0] for p in pts],
                [p[1] - p[2] for p in pts],
                [p[1] + p[2] for p in pts],
                color=color, alpha=0.12, zorder=1,
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, linewidth=1.8, zorder=2)

        for xi, m, s in pts:
            ax.scatter(xi, m, color=color, s=60, zorder=4)

        if pts and pts[0][0] == x_left:
            ax.text(x_left - 0.04, pts[0][1], f"{label} {pts[0][1]:.3f}",
                    ha="right", va="center", fontsize=11, fontweight="bold", color=color)
        if pts:
            ax.text(x_right + 0.04, pts[-1][1], f"{pts[-1][1]:.3f} {label}",
                    ha="left", va="center", fontsize=11, fontweight="bold", color=color)

        ax.axhline(floor, color=color, linestyle=":", linewidth=0.9, alpha=0.5, zorder=1)

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["2 institutions", "3 institutions"], fontsize=10)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(a) Per-task test AUC-ROC", pad=6)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(bottom=False)
    if all_auc:
        ax.set_ylim(min(all_auc) - 0.06, max(all_auc) + 0.06)


def _plot_cost(ax):
    ns = [2, 3]
    mu_r  = [_CONV_ROUNDS[n][0] for n in ns]
    sd_r  = [_CONV_ROUNDS[n][1] for n in ns]
    mu_wc = [_CONV_ROUNDS[n][0] * _SEC_PER_ROUND[n] / 60 for n in ns]
    sd_wc = [_CONV_ROUNDS[n][1] * _SEC_PER_ROUND[n] / 60 for n in ns]

    bar_color  = "#b0b8c8"
    line_color = "#3a3a3a"
    xs = [0, 1]

    ax.bar(xs, mu_r, yerr=sd_r, color=bar_color, width=0.45,
           capsize=5, zorder=2, label="Rounds", error_kw={"linewidth": 1.2})
    ax.set_ylabel("Convergence rounds", color="#5a6a80")
    ax.tick_params(axis="y", labelcolor="#5a6a80")
    ax.set_ylim(0, max(mu_r) + max(sd_r) + 22)

    ax2 = ax.twinx()
    ax2.plot(xs, mu_wc, color=line_color, linewidth=2, marker="o",
             markersize=7, zorder=3, label="Wall-clock (min)")
    ax2.errorbar(xs, mu_wc, yerr=sd_wc, fmt="none", color=line_color,
                 capsize=5, linewidth=1.2, zorder=3)
    ax2.set_ylabel("Total training time (min)", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, max(mu_wc) + max(sd_wc) + 2.5)

    # "rds" centred above the error bar cap; "min" to the right of the dot
    for xi, r, sr, wc in zip(xs, mu_r, sd_r, mu_wc):
        ax.text(xi, r + sr + 4, f"{r:.0f} rds",
                ha="center", va="bottom", fontsize=9,
                color="#1a2a3a", fontweight="bold")
        ax2.text(xi + 0.18, wc, f"{wc:.1f} min",
                 ha="left", va="center", fontsize=9,
                 color=line_color, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(["2 institutions", "3 institutions"], fontsize=10)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("(b) Training cost", pad=6)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3",      default="results/exp3.csv",
                        help="Unused; kept for backwards compatibility.")
    parser.add_argument("--test_exp3", default="results/test_exp3.csv")
    parser.add_argument("--output",    default="Manuscript/figures/Figure5b_Scalability.png")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_exp3)

    fig, (ax_slope, ax_cost) = plt.subplots(
        1, 2, figsize=(10.5, 4.8),
        gridspec_kw={"width_ratios": [5, 3]},
    )
    fig.suptitle(
        "Scalability: 2 vs. 3 institutions",
        fontweight="normal", y=1.01,
    )

    _plot_slopegraph(ax_slope, test_df)
    _plot_cost(ax_cost)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=800, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
