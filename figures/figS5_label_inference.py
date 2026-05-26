"""
figures/figS5_label_inference.py — S5 Fig: Theoretical Bound vs. Empirical Inference AUC.

Shows that the multi-task label inference bound g(σ, ρ) = Φ(C·√(1+ρ)/σ) holds:
theoretical bound lines ≥ empirical accuracy points at each ε level.

Single panel:
  x-axis: ε levels (∞ → 0.5), ordered high privacy → low privacy
  y-axis: label inference AUC (bound / empirical)
  One dashed line per ρ value (theoretical bound)
  Scatter points: empirical IHM inference AUC (uniform mode, mean across seeds)

Usage:
    python figures/figS5_label_inference.py \
        --input results/bound_validation.csv \
        --output plots/S5_LabelInference.png
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

# One brand colour per ρ value (5 values → _C[0..4])
RHO_COLORS = [_C[0], _C[1], _C[2], _C[3], _C[4]]

EPS_DISPLAY_ORDER = ["inf", "10.0", "5.0", "2.0", "1.0", "0.5"]
EPS_LABELS        = ["∞", "10", "5", "2", "1", "0.5"]


def _eps_to_x(eps_str: str) -> int:
    return EPS_DISPLAY_ORDER.index(str(eps_str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/bound_validation.csv")
    parser.add_argument("--output", default="Manuscript/figures/S5_LabelInference.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["epsilon_level"] = df["epsilon_level"].astype(str)

    rho_values = sorted(df["rho"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Theoretical bound lines (one per ρ)
    for rho, color in zip(rho_values, RHO_COLORS):
        sub  = df[df["rho"] == rho].copy()
        sub["x"] = sub["epsilon_level"].apply(_eps_to_x)
        sub  = sub.sort_values("x")
        ax.plot(sub["x"], sub["bound_ihm"], color=color, ls="--", linewidth=1.4,
                label=f"Bound ρ={rho}")

    # Empirical IHM inference AUC (mean across ρ — same value for all ρ rows)
    emp = df.groupby("epsilon_level")["empirical_ihm_auroc_mean"].first().reset_index()
    emp["x"] = emp["epsilon_level"].apply(_eps_to_x)
    emp = emp.sort_values("x")
    ax.scatter(emp["x"], emp["empirical_ihm_auroc_mean"], color=_C[6], zorder=5,
               s=50, marker="D", label="Empirical (IHM, uniform)")

    # chance level — matches existing #888888 reference line style
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, label="Chance (0.5)")

    ax.set_xticks(range(len(EPS_DISPLAY_ORDER)))
    ax.set_xticklabels(EPS_LABELS)
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Label Inference AUC")
    ax.set_title("Theoretical label inference bound vs. empirical attack AUC across privacy budgets")
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=800, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
