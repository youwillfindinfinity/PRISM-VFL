"""
figures/figure8_pcmu.py — Figure 8 (PCMU)

PCMU composite score dot plot (Cleveland plot).
One row per privacy level of PRISM (embed_dim 64), PCMU on x-axis, 95% bootstrap CI bars
over seeds. PRISM ε=∞ (anchor, PCMU = 1.0) separated from the DP rows.
Primary definition (R = total training rounds); the best-checkpoint sensitivity rows
(metric pcmu_best_round) are reported in the supplementary table.

Usage:
    python figures/figure8_pcmu.py
    python figures/figure8_pcmu.py --input results_revision/prism_privacy_sweep_stats.csv \
                                    --output smoke_tests/revision1/Figure8_PCMU.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.transforms import blended_transform_factory
import pandas as pd

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "normal",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
})

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]
_DP = ["#c89a97", "#b07570", "#8a3c48", "#6b2d38", "#4d1f28"]

INF = float("inf")

# Row order (top to bottom); separator drawn after the first row
ROW_ORDER = [INF, 10.0, 5.0, 2.0, 1.0, 0.5]

LABELS = {
    INF:  r"PRISM  $\varepsilon=\infty$",
    10.0: r"PRISM  $\varepsilon=10$",
    5.0:  r"PRISM  $\varepsilon=5$",
    2.0:  r"PRISM  $\varepsilon=2$",
    1.0:  r"PRISM  $\varepsilon=1$",
    0.5:  r"PRISM  $\varepsilon=0.5$",
}

COLORS = {INF: _C[0], **dict(zip(ROW_ORDER[1:], _DP))}

N_P1 = 1   # PRISM no-DP
GAP  = 1.2

def _ypos() -> dict[float, float]:
    pos = {}
    for i, eps in enumerate(ROW_ORDER[:N_P1]):
        pos[eps] = float(N_P1 - 1 - i)
    for i, eps in enumerate(ROW_ORDER[N_P1:]):
        pos[eps] = float(-(GAP + i))
    return pos


def load(path: str) -> dict[float, tuple[float, float, float]]:
    """(mean, ci_lo, ci_hi) per ε from the primary PCMU rows of the sweep stats CSV."""
    df = pd.read_csv(path)
    df = df[(df["metric"] == "pcmu") & (df["split"] == "val")].set_index("epsilon_level")
    missing = [e for e in ROW_ORDER if e not in df.index]
    if missing:
        raise ValueError(f"No PCMU rows in {path} for ε = {missing}")
    return {e: (float(df.loc[e, "mean"]), float(df.loc[e, "ci_lo"]), float(df.loc[e, "ci_hi"]))
            for e in ROW_ORDER}


def plot(data: dict[float, tuple[float, float, float]], out: str) -> None:
    ypos = _ypos()
    x_lo = min(lo for _, lo, _ in data.values()) - 0.5
    x_hi = max(hi for _, _, hi in data.values()) + 0.6

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.14)

    # ── Reference line: anchor PCMU = 1.0 (PRISM ε=∞) ─────────────────────────
    ax.axvline(1.0, color=_C[0], lw=1.0, ls=":", alpha=0.55, zorder=1)

    # ── Group separator ───────────────────────────────────────────────────────
    ax.axhline(-(GAP / 2), color="#cccccc", lw=0.8, ls="-", zorder=1)

    # ── Dots and CI bars ──────────────────────────────────────────────────────
    for eps in ROW_ORDER:
        mu, lo, hi = data[eps]
        y, color = ypos[eps], COLORS[eps]
        ax.plot([lo, hi], [y, y], color=color, lw=1.8, alpha=0.5, zorder=2, solid_capstyle="round")
        ax.plot(mu, y, marker="o", ms=9, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(hi + 0.06, y, f"{mu:.3f}", va="center", ha="left",
                fontsize=13, fontweight="bold", color="#444444")

    # ── y-axis tick labels ────────────────────────────────────────────────────
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([LABELS[e] for e in ROW_ORDER], fontsize=9)
    ax.tick_params(axis="y", length=0)

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.set_xlabel("PCMU", labelpad=6)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(True, axis="x", alpha=0.2, zorder=0)
    ax.axvline(0, color="#aaaaaa", lw=0.6, zorder=1)

    # ── Direction arrow ───────────────────────────────────────────────────────
    # Blended transform: data-x coords, axes-fraction y, below the "PCMU" x-axis label.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        "",
        xy=(1.0, -0.16), xytext=(x_lo + 0.2, -0.16),
        xycoords=trans, textcoords=trans,
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0, mutation_scale=11),
        annotation_clip=False,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = [
        mlines.Line2D([], [], color=_C[0], ls=":", lw=1.0,
                      label=r"Anchor: PRISM $\varepsilon=\infty$ (PCMU = 1)"),
        mlines.Line2D([], [], color="#888888", lw=1.8, alpha=0.5,
                      label="95% bootstrap CI (10 seeds)"),
    ]
    ax.legend(handles=leg, loc="lower right", framealpha=0.85, edgecolor="none", fontsize=9)

    fig.suptitle(
        "Privacy-Communication-adjusted Multi-task Utility (PCMU)\n"
        "across privacy budgets",
        fontsize=13, fontweight="normal", y=0.99, x=0.56, ha="center",
    )

    plt.savefig(out, dpi=800, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results_revision/prism_privacy_sweep_stats.csv")
    parser.add_argument("--output", default="Manuscript/figures_revision/Figure8_PCMU.png")
    args = parser.parse_args()

    plot(load(args.input), args.output)


if __name__ == "__main__":
    main()
