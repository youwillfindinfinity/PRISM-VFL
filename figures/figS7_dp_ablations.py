"""
figures/figS7_dp_ablations.py — S7 Fig: DP ablation figures.

Reads results/dp_ablations.csv and generates one figure per ablation:

  Abl 1 (S7A) - dumbbell chart: uniform vs. stratified sigma at epsilon=5
  Abl 2 (S7B) - dumbbell chart: gradient coupling rho and IHM inference AUC
                for related (IHM+Decomp) vs. unrelated (IHM+Pheno) task pairs
  Abl 3 (S7C) - line plot: per-task AUC vs. epsilon for embed_dim in {32, 64, 128}

Usage
-----
  python figures/figS7_dp_ablations.py                          # unified figure → plots/S7A_DP_ABL1.png (via --abl 1 etc.)
  python figures/figS7_dp_ablations.py --abl 1                  # → plots/S7A_DP_ABL1.png
  python figures/figS7_dp_ablations.py --abl 2                  # → plots/S7B_DP_ABL2.png
  python figures/figS7_dp_ablations.py --abl 3                  # → plots/S7C_DP_ABL3.png
  python figures/figS7_dp_ablations.py --input results/dp_ablations.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS = [
    ("val_ihm_auroc",         "IHM AUC-ROC",    0.75),
    ("val_decomp_auroc",      "Decomp AUC-ROC",  0.70),
    ("val_pheno_macro_auroc", "Pheno Macro-AUC", 0.65),
]

_PLT_DEFAULTS = {
    "figure.dpi":       150,
    "font.size":        10,
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
}

# Brand palette — shared across all figure scripts in this repo
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

_C_UNIFORM     = _C[0]
_C_STRATIFIED  = _C[6]
_C_IHM_DECOMP  = _C[1]
_C_IHM_PHENO   = _C[4]
_EMBED_COLORS  = {32: _C[3], 64: _C[6], 128: _C[1]}
_FLOOR_COLOR   = "#888888"
# Task colours — IHM/Decomp/Pheno match resilience_variance.py
_TASK_COLORS   = {
    "val_ihm_auroc":         _C[1],   # purple
    "val_decomp_auroc":      _C[2],   # dark charcoal
    "val_pheno_macro_auroc": _C[3],   # burgundy
}


def _last_round_agg(df: pd.DataFrame, group_col: str, metric: str):
    if "round" in df.columns:
        idx = df.groupby([group_col, "seed"])["round"].idxmax()
        df = df.loc[idx]
    g = df.groupby(group_col)[metric]
    return g.mean(), g.std().fillna(0)


# ---------------------------------------------------------------------------
# Ablation 1 - uniform vs. stratified sigma (paired slope chart, 3 task panels)
#
# Each seed is drawn as a grey slope line so the reader can see whether
# stratified sigma consistently helps or hurts per task.  Bold task-coloured
# mean ± std markers dominate visually; a delta annotation states the
# mean shift between conditions numerically.
# ---------------------------------------------------------------------------

def _fill_abl1(axes, df: pd.DataFrame) -> None:
    abl1    = df[df["ablation"] == "abl1"].copy()
    configs = ["uniform_eps5", "stratified_eps5"]
    xlabels = ["Uniform σ", "Stratified σ"]
    x       = [0, 1]

    for ax, (metric, ylabel, floor) in zip(axes, TASKS):
        color = _TASK_COLORS[metric]
        seeds = sorted(abl1["seed"].unique())

        # per-seed slopes in neutral grey — shows consistency of direction
        for seed in seeds:
            vals = []
            for cfg in configs:
                row = abl1[(abl1["config"] == cfg) & (abl1["seed"] == seed)][metric]
                vals.append(float(row.iloc[0]) if len(row) else float("nan"))
            if any(np.isnan(v) for v in vals):
                continue
            ax.plot(x, vals, color="#aaaaaa", alpha=0.55, linewidth=1.2, zorder=2)
            ax.scatter(x, vals, color="#aaaaaa", alpha=0.65, s=22, zorder=3)

        # mean ± std at each condition (bold, task colour)
        means = []
        for xi, cfg in enumerate(configs):
            vals = abl1[abl1["config"] == cfg][metric].dropna()
            m = float(vals.mean()) if len(vals) else float("nan")
            s = float(vals.std())  if len(vals) > 1 else 0.0
            means.append(m)
            if np.isnan(m):
                continue
            ax.errorbar(xi, m, yerr=s, fmt="o", color=color,
                        markerfacecolor=color, markeredgewidth=1.5,
                        capsize=4, capthick=1.5, elinewidth=1.5, markersize=9,
                        zorder=5)
            ax.text(xi, m + s + 0.008, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=color)

        # dashed line connecting means + Δ annotation
        if not any(np.isnan(m) for m in means):
            ax.plot(x, means, color=color, linewidth=2.0, alpha=0.8,
                    linestyle="--", zorder=4)
            delta = means[1] - means[0]
            sign  = "+" if delta >= 0 else ""
            mid_y = (means[0] + means[1]) / 2
            data_spread = abl1[metric].dropna()
            y_offset = max(data_spread.max() - data_spread.min(), 0.05) * 0.18
            ax.text(0.5, mid_y + y_offset, f"{sign}{delta:.3f}",
                    ha="center", va="bottom", fontsize=11,
                    color=color, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=color,
                              linewidth=0.6, alpha=1.0, pad=2.5,
                              boxstyle="round,pad=0.3"))

        # clinical utility floor
        ax.axhline(floor, color=_FLOOR_COLOR, linestyle=":", linewidth=0.9, zorder=1)
        ax.text(0.98, floor + 0.005, f"floor={floor:.2f}",
                ha="right", va="bottom", fontsize=7, color=_FLOOR_COLOR,
                transform=ax.get_yaxis_transform())

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9, pad=6)
        ax.set_xlim(-0.35, 1.35)
        ax.grid(True, axis="y", alpha=0.2, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def plot_abl1(df: pd.DataFrame, output: Path) -> None:
    if df[df["ablation"] == "abl1"].empty:
        print("[abl1] no data - skipping"); return
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    fig.suptitle(
        "Uniform vs. stratified noise allocation at ε = 5  "
        "(grey lines = per-seed trajectories, bold = mean ± std)",
        fontsize=10, y=0.99,
    )
    _fill_abl1(axes, df)
    # Shared y-axis from 0 — keeps the absolute AUC scale honest and
    # makes floor lines comparable across tasks
    abl1 = df[df["ablation"] == "abl1"]
    all_task_maxes = [abl1[m].dropna().max() for m, _, _ in TASKS if not abl1[m].dropna().empty]
    all_floors     = [f for _, _, f in TASKS]
    global_hi = max(all_task_maxes + all_floors) + 0.08
    axes[0].set_ylim(0, global_hi)   # propagates to all panels via sharey
    # shared y-axis: single "AUC" label on leftmost panel only
    axes[0].set_ylabel("AUC", fontsize=9)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=800, bbox_inches="tight")
    print(f"Saved: {output}"); plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation 2 - related vs. unrelated task pair (two dot-plot panels)
# ---------------------------------------------------------------------------

def _fill_abl2(axes, df: pd.DataFrame) -> None:
    abl2    = df[df["ablation"] == "abl2"].copy()
    configs = ["ihm_decomp", "ihm_pheno"]
    colors  = [_C_IHM_DECOMP, _C_IHM_PHENO]
    xlabels = ["IHM + Decomp\n(high rho)", "IHM + Pheno\n(low rho)"]

    panels = [
        ("rho",                "Gradient coupling rho",  None),
        ("inference_auroc_ihm","IHM label inference AUC", 0.5),
    ]

    for ax, (col, ylabel, ref) in zip(axes, panels):
        means, stds = [], []
        for cfg in configs:
            vals = abl2[abl2["config"] == cfg][col].dropna()
            means.append(float(vals.mean()) if len(vals) else float("nan"))
            stds.append(float(vals.std())   if len(vals) > 1 else 0.0)

        x = np.arange(len(configs))
        for ci, (color, m, s) in enumerate(zip(colors, means, stds)):
            if np.isnan(m):
                continue
            mfc = color if ci == 0 else "white"
            ax.errorbar(ci, m, yerr=s, fmt="o", color=color,
                        markerfacecolor=mfc, markeredgewidth=1.5,
                        capsize=4, capthick=1.0, elinewidth=1.0, markersize=9, zorder=3)
            ax.text(ci, m + s + 0.008, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

        if ref is not None:
            ax.axhline(ref, color=_FLOOR_COLOR, linestyle="--",
                       linewidth=0.9, label=f"Random ({ref})")
            ax.legend(fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9, pad=6)

        valid = [m for m in means if not np.isnan(m)]
        valid_s = [s for s in stds if not np.isnan(s)]
        if valid:
            pad = max(valid_s + [0]) + 0.05
            ax.set_ylim(min(valid) - pad, max(valid) + pad + 0.04)

        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        ax.set_xlim(-0.6, len(configs) - 0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def plot_abl2(df: pd.DataFrame, output: Path) -> None:
    if df[df["ablation"] == "abl2"].empty:
        print("[abl2] no data - skipping"); return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.suptitle("Coupling amplification: task pair comparison at epsilon=5",
                 fontsize=10, y=1.02)
    _fill_abl2(axes, df)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=800, bbox_inches="tight")
    print(f"Saved: {output}"); plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation 3 - embed_dim x DP interaction (line plot, unchanged)
# ---------------------------------------------------------------------------

_EPS_ORDER  = [1.0, 5.0, float("inf")]
_EPS_LABELS = ["epsilon=1", "epsilon=5", "epsilon=inf\n(no DP)"]


def _fill_abl3(axes, df: pd.DataFrame) -> None:
    abl3 = df.copy()
    embed_dims = sorted(abl3["embed_dim"].dropna().unique().astype(int))
    x = np.arange(len(_EPS_ORDER))

    for ax, (metric, ylabel, floor) in zip(axes, TASKS):
        for ed in embed_dims:
            sub   = abl3[abl3["embed_dim"] == ed]
            means, stds = [], []
            for eps in _EPS_ORDER:
                rows = sub[np.isclose(sub["epsilon_level"].astype(float), eps, equal_nan=True)
                           if np.isfinite(eps)
                           else ~np.isfinite(sub["epsilon_level"].astype(float))]
                vals = rows[metric].dropna()
                means.append(float(vals.mean()) if len(vals) else float("nan"))
                stds.append(float(vals.std())   if len(vals) > 1 else 0.0)

            ax.errorbar(x, means, yerr=stds, marker="o", markersize=5, linewidth=1.5,
                        capsize=3, label=f"embed={ed}",
                        color=_EMBED_COLORS.get(ed, "#888888"))

        ax.axhline(floor, color=_FLOOR_COLOR, linestyle="--", linewidth=0.9,
                   label=f"Floor ({floor})")
        ax.set_xticks(x); ax.set_xticklabels(_EPS_LABELS)
        ax.set_xlabel("Privacy budget epsilon")
        ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def plot_abl3(df: pd.DataFrame, output: Path) -> None:
    if df.empty:
        print("[abl3] no data - skipping"); return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=False)
    fig.suptitle(
        "Per-task AUC-ROC as a function of privacy budget epsilon across embedding dimensions",
        fontsize=10, y=1.02,
    )
    _fill_abl3(axes, df)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=800, bbox_inches="tight")
    print(f"Saved: {output}"); plt.close(fig)


# ---------------------------------------------------------------------------
# Unified panel
# ---------------------------------------------------------------------------

def plot_unified(df: pd.DataFrame, df3: pd.DataFrame, output: Path) -> None:
    mosaic = [
        ["a1_ihm",  "a1_decomp", "a1_pheno"],
        ["a2_rho",  "a2_inf",    "."],
        ["a3_ihm",  "a3_decomp", "a3_pheno"],
    ]
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=(13, 11),
        gridspec_kw={"hspace": 0.5, "wspace": 0.38},
    )
    fig.suptitle(
        "Differential privacy ablation study: noise strategy, task coupling, and embedding dimension",
        fontweight="normal", y=0.99,
    )

    abl1_axes = [axd["a1_ihm"], axd["a1_decomp"], axd["a1_pheno"]]
    _fill_abl1(abl1_axes, df)
    # Shared y-axis from 0 — same logic as standalone plot_abl1
    abl1 = df[df["ablation"] == "abl1"]
    all_task_maxes = [abl1[m].dropna().max() for m, _, _ in TASKS if not abl1[m].dropna().empty]
    all_floors     = [f for _, _, f in TASKS]
    global_hi = max(all_task_maxes + all_floors) + 0.08
    for ax in abl1_axes:
        ax.set_ylim(0, global_hi)
    abl1_axes[0].set_ylabel("AUC", fontsize=9)
    for ax in abl1_axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    _fill_abl2([axd["a2_rho"], axd["a2_inf"]], df)
    _fill_abl3([axd["a3_ihm"], axd["a3_decomp"], axd["a3_pheno"]], df3)

    fig.subplots_adjust(hspace=1.1, wspace=0.38, top=0.94, bottom=0.06, left=0.07, right=0.97)

    row_labels = [
        ("Ablation 1: Uniform vs. stratified noise allocation at epsilon = 5", "a1_ihm"),
        ("Ablation 2: Task coupling and label inference at epsilon = 5",        "a2_rho"),
        ("Ablation 3: Embedding dimension across privacy budgets",               "a3_ihm"),
    ]
    for label, ax_key in row_labels:
        y_top = axd[ax_key].get_position().y1 + 0.018
        fig.text(0.5, y_top, label, ha="center", va="bottom",
                 fontsize=12, fontweight="normal", color="#333333")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=800, bbox_inches="tight")
    print(f"Saved: {output}"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_abl3_from_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["ablation"] == "abl3"].copy()
    df = df.rename(columns={
        "ihm_auroc":         "val_ihm_auroc",
        "decomp_auroc":      "val_decomp_auroc",
        "pheno_macro_auroc": "val_pheno_macro_auroc",
    })
    df["epsilon_level"] = pd.to_numeric(df["epsilon_level"], errors="coerce").fillna(float("inf"))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input",       default="results/dp_ablations.csv")
    parser.add_argument("--input_test",  default="results/test_ablations_dp.csv")
    parser.add_argument("--outdir", default="Manuscript/figures")
    parser.add_argument("--abl",    type=int, choices=[1, 2, 3],
                        help="Generate only this ablation figure (omit for unified panel).")
    args = parser.parse_args()

    plt.rcParams.update(_PLT_DEFAULTS)

    df  = pd.read_csv(args.input)
    df3 = _load_abl3_from_test(args.input_test)
    out = Path(args.outdir)

    if args.abl is None:
        plot_unified(df, df3, out / "S7A_DP_ABL1.png")
    else:
        if args.abl == 1: plot_abl1(df, out / "S7A_DP_ABL1.png")
        if args.abl == 2: plot_abl2(df, out / "S7B_DP_ABL2.png")
        if args.abl == 3: plot_abl3(df3, out / "S7C_DP_ABL3.png")


if __name__ == "__main__":
    main()
