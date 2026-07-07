"""
figures/plot_brier.py — Calibration curve figures using Brier score.

Generates two output files from per-sample predictions saved by
experiments/calibration_predictions.py:

  Manuscript/figures/brier_nodp.png  — calibration curves per model × task (no-DP)
  Manuscript/figures/brier_dp.png    — calibration curves per ε level × task (DP sweep)

Plot style: predicted probability (x) vs observed frequency (y), diagonal
perfect-calibration reference, rug plot at base, Brier score in text box.
Matches the colour palette, font, and DPI used across all other figures.

Usage:
    python figures/plot_brier.py \
        --nodp results/predictions_nodp.csv \
        --dp   results/predictions_dp.csv \
        --out_dir Manuscript/figures/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        15,
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":   17,
    "axes.titleweight": "normal",
    "axes.labelsize":   15,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  12,
})

# Brand palette — identical to all other figure scripts
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

# Qualitatively distinct colors for models and ε levels
_RED    = "#e6194b"
_GREEN  = "#2ca02c"
_BLUE   = "#1f77b4"
_DARK   = "#2f283d"
_ORANGE = "#ff7f0e"
_PURPLE = "#9467bd"
_TEAL   = "#17becf"

# Model colours + markers (no-DP plot)
MODEL_CFG = {
    "local_A":            {"label": "Local A",     "color": _RED,    "marker": "o"},
    "local_B":            {"label": "Local B",     "color": _RED,    "marker": "o"},
    "ST-IHM":             {"label": "ST-IHM",      "color": _GREEN,  "marker": "s"},
    "ST-Decomp":          {"label": "ST-Decomp",   "color": _GREEN,  "marker": "s"},
    "VFL-MTL":            {"label": "PRISM",     "color": _BLUE,   "marker": "^"},
    "centralized_oracle": {"label": "Centralised", "color": _DARK,   "marker": "D"},
}

# ε level colours + markers — maximally distinct, ordered high→low ε
EPS_FINITE   = [10.0,    5.0,      2.0,      1.0,      0.5]
EPS_COLORS   = [_TEAL,   _GREEN,   _ORANGE,  _PURPLE,  _RED]
EPS_MARKERS  = ["o",     "s",      "^",      "v",      "P"]
INF_COLOR    = _DARK    # no-DP reference — dark charcoal
INF_MARKER   = "D"

N_BINS = 10


# ── shared helpers ────────────────────────────────────────────────────────────

def _panel_label(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes,
            fontsize=19, fontweight="bold", va="top")


def _hide_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _brier_box(ax, lines: list[str]):
    """Annotate Brier scores in a text box — mirrors reference figure style."""
    text = "Brier score\n" + "\n".join(lines)
    ax.text(0.04, 0.96, text, transform=ax.transAxes,
            fontsize=13, va="top", ha="left", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#aaaaaa", alpha=0.92))


def _calib_curve_from_seeds(df_group: pd.DataFrame, n_bins: int = N_BINS
                             ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Pool all seeds' predictions into one calibration curve; report Brier mean±std per seed.
    Returns (prob_pred, prob_true, mean_brier, std_brier).
    Uses quantile binning so bins are populated even under low-prevalence tasks.
    """
    yt_all = df_group["y_true"].values
    yp_all = df_group["y_prob"].values
    if len(np.unique(yt_all)) < 2:
        return np.array([]), np.array([]), float("nan"), float("nan")

    prob_true, prob_pred = calibration_curve(
        yt_all, yp_all, n_bins=n_bins, strategy="quantile"
    )
    briers = [
        brier_score_loss(sg["y_true"].values, sg["y_prob"].values)
        for _, sg in df_group.groupby("seed")
    ]
    return prob_pred, prob_true, float(np.mean(briers)), float(np.std(briers))


def _rug(ax, y_prob: np.ndarray, color: str, alpha: float = 0.25):
    ax.plot(y_prob, np.full_like(y_prob, -0.03),
            "|", color=color, alpha=alpha, markersize=3, markeredgewidth=0.6,
            transform=ax.get_xaxis_transform(), clip_on=False)


def _draw_calibration_panel(ax, curves: list[dict], title: str, panel_letter: str,
                            show_legend: bool = True):
    """
    curves: list of dicts with keys:
        label, color, marker, grid, mean_frac, mean_brier, std_brier, rug_probs
    """
    ax.plot([0, 1], [0, 1], ls="--", color="#aaaaaa", lw=1.2, zorder=0,
            label="Perfect calibration")

    brier_lines = []
    for c in curves:
        if len(c["grid"]) == 0:
            continue
        ax.plot(c["grid"], c["mean_frac"],
                color=c["color"], lw=2.0, label=c["label"],
                marker=c.get("marker", "o"), markersize=7,
                markerfacecolor=c["color"], markeredgecolor="white",
                markeredgewidth=0.8)
        _rug(ax, c["rug_probs"], c["color"])
        brier_lines.append(
            f"{c['label']}  {c['mean_brier']:.3f} [±{c['std_brier']:.3f}]"
        )

    _brier_box(ax, brier_lines)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False, fontsize=11, loc="lower right")
    _panel_label(ax, panel_letter)
    _hide_spines(ax)


# ── no-DP plot ────────────────────────────────────────────────────────────────

def plot_nodp(nodp_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(nodp_csv)

    ihm_models   = ["local_A",  "ST-IHM",    "VFL-MTL", "centralized_oracle"]
    decomp_models= ["local_B",  "ST-Decomp", "VFL-MTL", "centralized_oracle"]

    def _build_curves(models, task):
        curves = []
        for m in models:
            sub = df[(df.model == m) & (df.task == task)]
            if sub.empty:
                continue
            grid, mean_frac, mb, sb = _calib_curve_from_seeds(sub)
            curves.append({
                "label":      MODEL_CFG[m]["label"],
                "color":      MODEL_CFG[m]["color"],
                "marker":     MODEL_CFG[m]["marker"],
                "grid":       grid,
                "mean_frac":  mean_frac,
                "mean_brier": mb,
                "std_brier":  sb,
                "rug_probs":  sub["y_prob"].values[::3],
            })
        return curves

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Calibration Curves Under No Privacy Constraint", fontsize=18, y=1.02, fontweight="bold")

    _draw_calibration_panel(axes[0], _build_curves(ihm_models,    "ihm"),
                            "IHM", "A")
    _draw_calibration_panel(axes[1], _build_curves(decomp_models, "decomp"),
                            "Decompensation", "B", show_legend=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── DP plot ───────────────────────────────────────────────────────────────────

def plot_dp(dp_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(dp_csv)

    def _build_dp_curves(task):
        curves = []
        # No-DP reference (ε = ∞, uniform)
        sub_inf = df[np.isinf(df["epsilon_level"]) &
                     (df["mode"] == "uniform") & (df["task"] == task)]
        if not sub_inf.empty:
            grid, mf, mb, sb = _calib_curve_from_seeds(sub_inf)
            curves.append({
                "label": "No-DP (ε=∞)", "color": INF_COLOR, "marker": INF_MARKER,
                "grid": grid, "mean_frac": mf,
                "mean_brier": mb, "std_brier": sb,
                "rug_probs": sub_inf["y_prob"].values[::3],
            })
        # Finite ε levels (uniform σ only — cleaner plot)
        for eps, col, mkr in zip(EPS_FINITE, EPS_COLORS, EPS_MARKERS):
            sub = df[(df["epsilon_level"] == eps) &
                     (df["mode"] == "uniform") & (df["task"] == task)]
            if sub.empty:
                continue
            grid, mf, mb, sb = _calib_curve_from_seeds(sub)
            curves.append({
                "label": f"ε={eps:g}", "color": col, "marker": mkr,
                "grid": grid, "mean_frac": mf,
                "mean_brier": mb, "std_brier": sb,
                "rug_probs": sub["y_prob"].values[::3],
            })
        return curves

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Calibration Curves under Differential Privacy (Uniform σ)",
                 fontsize=18, y=1.02, fontweight="bold")

    _draw_calibration_panel(axes[0], _build_dp_curves("ihm"),
                            "IHM", "A")
    _draw_calibration_panel(axes[1], _build_dp_curves("decomp"),
                            "Decompensation", "B", show_legend=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodp",    default="results/predictions_nodp.csv")
    parser.add_argument("--dp",      default="results/predictions_dp.csv")
    parser.add_argument("--out_dir", default="Manuscript/figures/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    nodp_csv = Path(args.nodp)
    dp_csv   = Path(args.dp)

    if nodp_csv.exists():
        plot_nodp(nodp_csv, out_dir / "brier_nodp.png")
    else:
        print(f"[SKIP] {nodp_csv} not found — run calibration_predictions.py --mode nodp")

    if dp_csv.exists():
        plot_dp(dp_csv, out_dir / "brier_dp.png")
    else:
        print(f"[SKIP] {dp_csv} not found — run calibration_predictions.py --mode dp")


if __name__ == "__main__":
    main()
