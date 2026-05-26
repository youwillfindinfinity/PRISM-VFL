"""
figures/figure8_pcmu.py — Figure 8 (PCMU)

PCMU composite score dot plot (Cleveland plot) for the unified Paper 1 + 2 results.
One row per configuration, PCMU on x-axis, ±1 std CI bars.
Configurations grouped by research question with a visual separator.

Usage:
    python figures/figure8_pcmu.py
    python figures/figure8_pcmu.py --input results/pcmu_paper_results_raw.csv \
                                    --output plots/Figure8_PCMU.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.transforms import blended_transform_factory
import numpy as np
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

# Row order (top to bottom)
ROW_ORDER = [
    "centralized",
    "VFL-MTL",
    "ST-IHM", "ST-Decomp", "ST-Pheno",
    # separator inserted in drawing
    "eps_10.0", "eps_5.0", "eps_2.0", "eps_1.0", "eps_0.5",
]

LABELS = {
    "centralized": "Centralized oracle",
    "VFL-MTL":     r"PRISM  $\varepsilon=\infty$",
    "ST-IHM":      r"ST-IHM  $\varepsilon=\infty$",
    "ST-Decomp":   r"ST-Decomp  $\varepsilon=\infty$",
    "ST-Pheno":    r"ST-Pheno  $\varepsilon=\infty$",
    "eps_10.0":    r"PRISM  $\varepsilon=10$",
    "eps_5.0":     r"PRISM  $\varepsilon=5$",
    "eps_2.0":     r"PRISM  $\varepsilon=2$",
    "eps_1.0":     r"PRISM  $\varepsilon=1$",
    "eps_0.5":     r"PRISM  $\varepsilon=0.5$",
}

COLORS = {
    "centralized": _C[6],
    "VFL-MTL":     _C[0],
    "ST-IHM":      _C[1],
    "ST-Decomp":   _C[2],
    "ST-Pheno":    _C[4],
    "eps_10.0":    _DP[0],
    "eps_5.0":     _DP[1],
    "eps_2.0":     _DP[2],
    "eps_1.0":     _DP[3],
    "eps_0.5":     _DP[4],
}

# y-positions: gap of 1.5 between the two groups
N_P1 = 5   # centralized + VFL-MTL + 3 ST
N_P2 = 5   # 5 DP configs
GAP  = 1.2

def _ypos() -> dict[str, float]:
    pos = {}
    for i, cfg in enumerate(ROW_ORDER[:N_P1]):
        pos[cfg] = float(N_P1 - 1 - i)           # 4 → 0
    for i, cfg in enumerate(ROW_ORDER[N_P1:]):
        pos[cfg] = float(-(GAP + i))              # -(1.2) → -(1.2+4)
    return pos


def load(path: str) -> dict[str, tuple[float, float]]:
    df  = pd.read_csv(path)
    out: dict[str, tuple[float, float]] = {}

    out["centralized"] = (1.0, 0.0)

    exp1 = df[df["source"] == "exp1"]
    for model in ("VFL-MTL", "ST-IHM", "ST-Decomp", "ST-Pheno"):
        rows = exp1[exp1["config"] == model]["pcmu"].dropna()
        out[model] = (float(rows.mean()), float(rows.std(ddof=1)) if len(rows) > 1 else 0.0)

    priv = df[df["source"] == "privacy"]
    for key in ("eps_10.0", "eps_5.0", "eps_2.0", "eps_1.0", "eps_0.5"):
        rows = priv[priv["config"] == key]["pcmu"].dropna()
        out[key] = (float(rows.mean()), float(rows.std(ddof=1)) if len(rows) > 1 else 0.0)

    return out


def plot(data: dict[str, tuple[float, float]], out: str) -> None:
    ypos  = _ypos()
    ref   = data["VFL-MTL"][0]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.28, right=0.95, top=0.91, bottom=0.10)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axvline(1.0, color=_C[6], lw=1.2, ls="--", alpha=0.55, zorder=1)
    ax.axvline(ref, color=_C[0], lw=1.0, ls=":",  alpha=0.55, zorder=1)

    # ── Group separator ───────────────────────────────────────────────────────
    sep_y = -(GAP / 2)
    ax.axhline(sep_y, color="#cccccc", lw=0.8, ls="-", zorder=1)


    # ── Dots and CI bars ──────────────────────────────────────────────────────
    for cfg in ROW_ORDER:
        mu, sd = data[cfg]
        y      = ypos[cfg]
        color  = COLORS[cfg]

        # CI bar
        if sd > 0:
            ax.plot([mu - sd, mu + sd], [y, y],
                    color=color, lw=1.8, alpha=0.5, zorder=2, solid_capstyle="round")

        # Dot — diamond for centralized (not an experimental result), circle for rest
        marker = "D" if cfg == "centralized" else "o"
        ms     = 7  if cfg == "centralized" else 9
        ax.plot(mu, y, marker=marker, ms=ms, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)

        # Value annotation to the right of CI bar
        xann = mu + sd + 0.06 if sd > 0 else mu + 0.06
        value_str = "1.000" if cfg == "centralized" else f"{mu:.3f}"
        ax.text(xann, y, value_str, va="center", ha="left",
                fontsize=13, fontweight="bold", color="#444444")

    # ── y-axis tick labels ────────────────────────────────────────────────────
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([LABELS[c] for c in ROW_ORDER], fontsize=9)
    ax.tick_params(axis="y", length=0)

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.set_xlabel("PCMU", labelpad=6)
    ax.set_xlim(-3.8, 1.8)
    ax.grid(True, axis="x", alpha=0.2, zorder=0)
    ax.axvline(0, color="#aaaaaa", lw=0.6, zorder=1)

    # ── Direction arrow ───────────────────────────────────────────────────────
    # Uses a blended transform: data-x coords, axes-fraction y.
    # y=-0.16 sits below the "PCMU" x-axis label without overlapping it.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        "",
        xy=(1.0, -0.10), xytext=(-3.0, -0.10),
        xycoords=trans, textcoords=trans,
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0, mutation_scale=11),
        annotation_clip=False,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = [
        mlines.Line2D([], [], color=_C[6], ls="--", lw=1.2,
                      label="Upper bound anchor"),
        mlines.Line2D([], [], color=_C[0], ls=":",  lw=1.0,
                      label=r"PRISM no-DP reference"),
    ]
    ax.legend(handles=leg, loc="lower right", framealpha=0.85, edgecolor="none", fontsize=9)

    fig.suptitle(
        "Privacy-Communication-adjusted Multi-task Utility (PCMU)\n"
        "across federated and privacy configurations",
        fontsize=13, fontweight="normal", y=0.98, x=0.56, ha="center",
    )

    plt.savefig(out, dpi=800, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/pcmu_paper_results_raw.csv")
    parser.add_argument("--output", default="Manuscript/figures/Figure8_PCMU.png")
    args = parser.parse_args()

    data = load(args.input)
    plot(data, args.output)


if __name__ == "__main__":
    main()
