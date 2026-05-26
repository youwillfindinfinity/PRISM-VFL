"""
figures/pcmu_vs_epsilon.py — PCMU vs. privacy budget (ε) for Paper 2.

Single-panel line plot:
  x-axis: ε ∈ {0.5, 1, 2, 5, 10, ∞} (evenly spaced, log-like positions, ∞ at right)
  y-axis: PCMU score (mean ± 1 std across 3 seeds)
  Individual seed traces shown faintly behind the mean line.
  Reference lines: PCMU = 1.0 (centralized oracle anchor) and ε ≈ 2 (utility floor).

Usage:
    python figures/pcmu_vs_epsilon.py
    python figures/pcmu_vs_epsilon.py --input results/pcmu_paper_results_raw.csv \
                                       --output figures/pcmu_vs_epsilon.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "normal",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

EPS_ORDER  = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = ["0.5", "1", "2", "5", "10", r"$\infty$"]
X_POS      = list(range(len(EPS_ORDER)))          # 0..5, evenly spaced
EPS_TO_X   = {e: x for e, x in zip(EPS_ORDER, X_POS)}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["source"] == "privacy"].copy()


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df["x"] = df["epsilon"].map(EPS_TO_X)
    grp = df.groupby("x")["pcmu"]
    return pd.DataFrame({
        "x":    list(EPS_TO_X.values()),
        "mean": [grp.get_group(x).mean() for x in X_POS],
        "std":  [grp.get_group(x).std(ddof=1) for x in X_POS],
    })


def plot(df_raw: pd.DataFrame, out: str) -> None:
    summary = summarise(df_raw)
    df_raw["x"] = df_raw["epsilon"].map(EPS_TO_X)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(top=0.88)

    # ── Per-seed traces ───────────────────────────────────────────────────────
    for seed, grp in df_raw.groupby("seed"):
        grp = grp.sort_values("x")
        ax.plot(grp["x"], grp["pcmu"],
                color=_C[0], lw=0.9, alpha=0.35, zorder=2)

    # ── Mean ± 1 std band ─────────────────────────────────────────────────────
    xs   = summary["x"].values
    mean = summary["mean"].values
    std  = summary["std"].values

    ax.fill_between(xs, mean - std, mean + std,
                    color=_C[3], alpha=0.15, zorder=3)
    ax.plot(xs, mean,
            color=_C[3], lw=2.2, marker="o", ms=5, zorder=4,
            label=r"PCMU mean $\pm$ 1 std (3 seeds)")

    # ── Reference: centralized oracle anchor ─────────────────────────────────
    ax.axhline(1.0, color=_C[6], lw=1.3, ls="--", zorder=1,
               label="Centralized oracle (PCMU = 1.0)")

    # ── Reference: ε ≈ 2 utility floor ───────────────────────────────────────
    ax.axvline(EPS_TO_X[2.0], color=_C[1], lw=1.3, ls=":", zorder=1,
               label=r"$\varepsilon \approx 2$ (utility floor)")

    # ── Annotate the sharpest drop ────────────────────────────────────────────
    y1 = summary.loc[summary["x"] == EPS_TO_X[2.0], "mean"].values[0]
    y2 = summary.loc[summary["x"] == EPS_TO_X[1.0], "mean"].values[0]
    ax.annotate(
        "",
        xy=(EPS_TO_X[1.0], y2), xytext=(EPS_TO_X[2.0], y1),
        arrowprops=dict(arrowstyle="-|>", color=_C[2], lw=1.1),
        zorder=5,
    )
    ax.text(
        EPS_TO_X[1.0] + 0.07, (y1 + y2) / 2,
        r"$\varepsilon^*$ cliff",
        fontsize=8.5, color=_C[2], va="center",
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(X_POS)
    ax.set_xticklabels(EPS_LABELS)
    ax.set_xlabel(r"Privacy budget $\varepsilon$  (tighter $\leftarrow$ looser)", labelpad=6)
    ax.set_ylabel("PCMU", labelpad=6)
    ax.set_xlim(-0.3, len(X_POS) - 0.7)

    # secondary y-axis tick to mark 1.0 explicitly
    yticks = ax.get_yticks().tolist()
    if 1.0 not in yticks:
        yticks.append(1.0)
    ax.set_yticks(sorted(yticks))

    ax.legend(loc="lower right", framealpha=0.85, edgecolor="none")

    fig.suptitle(
        "PCMU degradation under differential privacy across the $\\varepsilon$ sweep",
        fontsize=11, fontweight="normal", y=0.98,
    )

    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/pcmu_paper_results_raw.csv")
    parser.add_argument("--output", default="plots/pcmu_vs_epsilon.png")
    args = parser.parse_args()

    df = load_data(args.input)
    plot(df, args.output)


if __name__ == "__main__":
    main()
