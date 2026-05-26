"""
experiments/evaluate_phase4.py — PCMU Phase 4: Sensitivity surface (privacy-focused).

Pipeline: Morris screening → Sobol indices → sensitivity heatmap → (ε, w1) contour plot.

Inputs varied:
  ε       — privacy budget ∈ [0.5, 10]  (continuous, log-interpolated from factorial)
  w1      — Δm weight ∈ [0.40, 0.90]   (w2=(1-w1)·2/3, w3=(1-w1)·1/3)
  embed_dim — ∈ {32, 64, 128}           (encoded as continuous 0→1→2, interpolated)

PCMU surrogate: z-score parameters fixed from the 108-cell factorial; raw components
(Δm, η_priv, η_comm) interpolated bilinearly from mean-across-seeds factorial values
at fixed task_config=all_tasks. Weight variation applied analytically on the z-scale.

Outputs:
  results/pcmu_phase4_sensitivity.csv   — Morris μ*, σ and Sobol S1, ST per input
  plots/S9_PCMUSensitivity.png          — 3-panel: heatmap + two contour plots (S9 Fig)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.evaluate_phase2 import (
    load_centralized_aurocs,
    load_st_aurocs,
    compute_components,
    compute_centralized_components,
    add_additive_pcmu,
)

PCMU_W_DEFAULT = {"delta_m": 0.70, "eta_priv": 0.20, "eta_comm": 0.10}
EPS_LEVELS     = [0.5, 1.0, 5.0]          # finite ε available in factorial
EMBED_DIMS     = [32, 64, 128]
TASK_CONFIG    = "all_tasks"


# ---------------------------------------------------------------------------
# Build surrogate from factorial
# ---------------------------------------------------------------------------

def build_surrogate(
    factorial_path: str = "results/pcmu_phase2_factorial.csv",
    centralized_path: str = "results/centralized.csv",
    exp1_path: str = "results/exp1.csv",
) -> tuple[RegularGridInterpolator, RegularGridInterpolator, RegularGridInterpolator,
           dict, float]:
    """
    Returns three interpolators for Δm, η_priv, η_comm over (log_eps_idx, embed_idx),
    z-score params dict, and cen_pcmu_raw (the unshifted centralized anchor score).

    Grid axes:
      log_eps_idx ∈ {0,1,2}  →  ε ∈ {0.5, 1.0, 5.0}
      embed_idx   ∈ {0,1,2}  →  embed_dim ∈ {32, 64, 128}
    """
    df = pd.read_csv(factorial_path)
    df["embed_dim"]     = df["embed_dim"].astype(int)
    df["epsilon_level"] = df["epsilon_level"].astype(float)
    df["seed"]          = df["seed"].astype(int)

    cen_df  = pd.read_csv(centralized_path)
    exp1_df = pd.read_csv(exp1_path)
    ca  = load_centralized_aurocs(cen_df)
    st  = load_st_aurocs(exp1_df)
    df, _ = compute_components(df, ca, st)
    eta_max = float(df["eta_comm"].dropna().max())
    cen     = compute_centralized_components(ca, st, eta_max)
    df      = add_additive_pcmu(df, cen)

    # z-score params fixed from full factorial
    z_params: dict[str, tuple[float, float]] = {}
    for col in ("delta_m", "eta_priv", "eta_comm"):
        vals = df[col].dropna()
        z_params[col] = (float(vals.mean()), float(vals.std(ddof=1)))

    # centralized anchor (unshifted)
    mu_dm,  s_dm  = z_params["delta_m"]
    mu_ep,  s_ep  = z_params["eta_priv"]
    mu_ec,  s_ec  = z_params["eta_comm"]
    cen_raw = (
        PCMU_W_DEFAULT["delta_m"]  * (cen["delta_m"]  - mu_dm) / s_dm +
        PCMU_W_DEFAULT["eta_priv"] * (cen["eta_priv"] - mu_ep) / s_ep +
        PCMU_W_DEFAULT["eta_comm"] * (cen["eta_comm"] - mu_ec) / s_ec
    )

    # Mean components per (epsilon_level, embed_dim) for all_tasks
    sub = df[(df["task_config"] == TASK_CONFIG) & (df["epsilon_level"].isin(EPS_LEVELS))]
    grid_means = (
        sub.groupby(["epsilon_level", "embed_dim"])[["delta_m", "eta_priv", "eta_comm"]]
        .mean()
        .reset_index()
    )

    # Build 2-D grids: axis0 = eps_idx (0,1,2), axis1 = embed_idx (0,1,2)
    dm_grid  = np.zeros((3, 3))
    ep_grid  = np.zeros((3, 3))
    ec_grid  = np.zeros((3, 3))

    for ei, eps in enumerate(EPS_LEVELS):
        for di, edim in enumerate(EMBED_DIMS):
            row = grid_means[
                (grid_means["epsilon_level"] == eps) &
                (grid_means["embed_dim"]     == edim)
            ]
            if row.empty:
                dm_grid[ei, di] = ep_grid[ei, di] = ec_grid[ei, di] = float("nan")
            else:
                dm_grid[ei, di] = float(row["delta_m"].iloc[0])
                ep_grid[ei, di] = float(row["eta_priv"].iloc[0])
                ec_grid[ei, di] = float(row["eta_comm"].iloc[0])

    axes = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    interp_dm = RegularGridInterpolator(axes, dm_grid,  method="linear", bounds_error=False, fill_value=None)
    interp_ep = RegularGridInterpolator(axes, ep_grid,  method="linear", bounds_error=False, fill_value=None)
    interp_ec = RegularGridInterpolator(axes, ec_grid,  method="linear", bounds_error=False, fill_value=None)

    return interp_dm, interp_ep, interp_ec, z_params, cen_raw


# ---------------------------------------------------------------------------
# PCMU surrogate function
# ---------------------------------------------------------------------------

def _eps_to_idx(eps: float) -> float:
    """Map ε ∈ [0.5, 5] to grid index ∈ [0, 2] via log interpolation."""
    log_bounds = (np.log(0.5), np.log(5.0))
    return 2.0 * (np.log(np.clip(eps, 0.5, 5.0)) - log_bounds[0]) / (log_bounds[1] - log_bounds[0])


def _embed_to_idx(embed: float) -> float:
    """Map embed_dim ∈ [32, 128] to index ∈ [0, 2] via log interpolation."""
    log_bounds = (np.log(32), np.log(128))
    return 2.0 * (np.log(np.clip(embed, 32, 128)) - log_bounds[0]) / (log_bounds[1] - log_bounds[0])


def make_pcmu_fn(interp_dm, interp_ep, interp_ec, z_params, cen_raw):
    """Returns a function pcmu(X) where X has columns [eps, w1, embed_dim]."""
    mu_dm, s_dm = z_params["delta_m"]
    mu_ep, s_ep = z_params["eta_priv"]
    mu_ec, s_ec = z_params["eta_comm"]

    def pcmu_fn(X: np.ndarray) -> np.ndarray:
        eps   = X[:, 0]
        w1    = X[:, 1]
        embed = X[:, 2]

        eps_idx   = _eps_to_idx(eps)
        embed_idx = _embed_to_idx(embed)
        pts = np.column_stack([eps_idx, embed_idx])

        dm = interp_dm(pts)
        ep = interp_ep(pts)
        ec = interp_ec(pts)

        dm_z = (dm - mu_dm) / s_dm
        ep_z = (ep - mu_ep) / s_ep
        ec_z = (ec - mu_ec) / s_ec

        w2 = (1.0 - w1) * (2.0 / 3.0)
        w3 = (1.0 - w1) * (1.0 / 3.0)

        return w1 * dm_z + w2 * ep_z + w3 * ec_z - cen_raw + 1.0

    return pcmu_fn


# ---------------------------------------------------------------------------
# Morris screening
# ---------------------------------------------------------------------------

def run_morris(pcmu_fn) -> pd.DataFrame:
    problem = {
        "num_vars": 3,
        "names":    ["epsilon", "w1", "embed_dim"],
        "bounds":   [[0.5, 5.0], [0.40, 0.90], [32.0, 128.0]],
    }
    X  = morris_sample.sample(problem, N=500, num_levels=8)
    Y  = pcmu_fn(X)
    Si = morris_analyze.analyze(problem, X, Y, num_resamples=100, print_to_console=False)
    return pd.DataFrame({
        "input":  problem["names"],
        "mu_star": Si["mu_star"],
        "sigma":   Si["sigma"],
    })


# ---------------------------------------------------------------------------
# Sobol indices
# ---------------------------------------------------------------------------

def run_sobol(pcmu_fn) -> pd.DataFrame:
    problem = {
        "num_vars": 3,
        "names":    ["epsilon", "w1", "embed_dim"],
        "bounds":   [[0.5, 5.0], [0.40, 0.90], [32.0, 128.0]],
    }
    X  = sobol_sample.sample(problem, N=2048)
    Y  = pcmu_fn(X)
    Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
    return pd.DataFrame({
        "input": problem["names"],
        "S1":    Si["S1"],
        "ST":    Si["ST"],
    })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_sensitivity(morris_df: pd.DataFrame, sobol_df: pd.DataFrame, pcmu_fn, out: str) -> None:
    plt.rcParams.update({
        "figure.dpi":       150,
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.titleweight": "normal",
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    _C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]
    CONTOUR_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "brand_div", ["#8a3c48", "#f5f0ea", "#2f4a6d"]
    )
    HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "brand_seq", ["#f5f0ea", "#9d7b78", "#8a3c48"]
    )
    N_LEVELS = 14

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.42, top=0.82)

    labels     = [r"$\varepsilon$  (privacy)", r"$w_1$  ($\Delta_m$ weight)", "embed dim"]
    input_keys = ["epsilon", "w1", "embed_dim"]

    # ── Panel 1: Sensitivity heatmap ─────────────────────────────────────────
    ax = axes[0]
    si = sobol_df.set_index("input")
    mi = morris_df.set_index("input")
    data = np.array([
        [si.loc[k, "S1"]       for k in input_keys],
        [si.loc[k, "ST"]       for k in input_keys],
        [mi.loc[k, "mu_star"]  for k in input_keys],
    ])
    im = ax.imshow(data, cmap=HEAT_CMAP, aspect="auto", vmin=0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r"Sobol  $S_1$", r"Sobol  $S_T$", r"Morris  $\mu^*$"], fontsize=9)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else "#222222", fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("sensitivity", fontsize=8)
    ax.set_title("Sensitivity indices\n(Morris & Sobol)", pad=10)

    # shared grid for contour panels
    eps_vals   = np.linspace(0.5, 5.0, 120)
    w1_vals    = np.linspace(0.40, 0.90, 120)
    embed_vals = np.linspace(32, 128, 120)

    def _add_contour(ax, X, Y, Z, xlabel, ylabel, title, ref_h=None, ref_v=None,
                     ref_h_label="", ref_v_label=""):
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        levels = np.linspace(vmin, vmax, N_LEVELS)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=CONTOUR_CMAP, alpha=0.92)
        cs = ax.contour(X, Y, Z, levels=levels[::2], colors="white",
                        linewidths=0.6, alpha=0.55)
        ax.clabel(cs, fmt="%.2f", fontsize=7, colors="white")
        cb = plt.colorbar(cf, ax=ax, shrink=0.80, pad=0.03)
        cb.set_label("PCMU", fontsize=8)
        cb.ax.tick_params(labelsize=8)
        if ref_h is not None:
            ax.axhline(ref_h, color=_C[5], lw=1.5, ls="--",
                       label=ref_h_label, zorder=5)
        if ref_v is not None:
            ax.axvline(ref_v, color=_C[0], lw=1.5, ls=":",
                       label=ref_v_label, zorder=5)
        ax.legend(fontsize=7.5, loc="upper left",
                  framealpha=0.7, edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
        ax.set_title(title, fontsize=10, pad=10)

    # ── Panel 2: ε × w₁  (embed_dim=64) ─────────────────────────────────────
    EPS, W1 = np.meshgrid(eps_vals, w1_vals)
    X_grid  = np.column_stack([EPS.ravel(), W1.ravel(),
                                np.full(EPS.size, 64.0)])
    Z1 = pcmu_fn(X_grid).reshape(EPS.shape)

    _add_contour(axes[1], EPS, W1, Z1,
                 xlabel=r"$\varepsilon$  (privacy budget)",
                 ylabel=r"$w_1$  ($\Delta_m$ weight)",
                 title=r"PCMU surface: $\varepsilon \times w_1$" + "\n" + r"(embed_dim = 64, all_tasks)",
                 ref_h=0.70, ref_v=2.0,
                 ref_h_label=r"default  $w_1 = 0.70$",
                 ref_v_label=r"$\varepsilon \approx 2$  (utility floor)")

    # ── Panel 3: ε × embed_dim  (w₁=0.70) ───────────────────────────────────
    EPS2, ED = np.meshgrid(eps_vals, embed_vals)
    X_grid2  = np.column_stack([EPS2.ravel(), np.full(EPS2.size, 0.70),
                                 ED.ravel()])
    Z2 = pcmu_fn(X_grid2).reshape(EPS2.shape)

    _add_contour(axes[2], EPS2, ED, Z2,
                 xlabel=r"$\varepsilon$  (privacy budget)",
                 ylabel="embedding dim",
                 title=r"PCMU surface: $\varepsilon \times$ embed_dim" + "\n" + r"($w_1 = 0.70$, all_tasks)",
                 ref_h=64.0, ref_v=2.0,
                 ref_h_label="default  embed_dim = 64",
                 ref_v_label=r"$\varepsilon \approx 2$  (utility floor)")

    fig.suptitle(
        r"Global sensitivity of PCMU to privacy budget ($\varepsilon$), task weight ($w_1$), and embedding dimension",
        fontsize=12, fontweight="normal", fontfamily="serif", y=1.02,
    )

    plt.savefig(out, dpi=800, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved figure → {out}")


# ---------------------------------------------------------------------------
# Weight sensitivity table
# ---------------------------------------------------------------------------

def compute_weight_sensitivity_table(
    paper_results_path: str = "results/pcmu_paper_results.csv",
    out_csv: str = "results/pcmu_weight_sensitivity.csv",
) -> pd.DataFrame:
    """
    Compute Spearman ρ of PCMU rankings under ±0.10 w1 perturbations vs. default w1=0.70.

    For each w1 ∈ {0.50, 0.60, 0.70, 0.80}:
      - w2 = (1 - w1) * 2/3,  w3 = (1 - w1) * 1/3   (maintains w2:w3 = 2:1 ratio)
      - PCMU recomputed as z-scored weighted sum of raw components
      - Spearman ρ computed against default w1=0.70 rankings

    Rows with missing component data are excluded from the ranking comparison.
    """
    from scipy.stats import spearmanr

    df = pd.read_csv(paper_results_path)

    # Keep only rows with all three components available
    component_cols = ["delta_m_mean", "eta_priv_mean", "eta_comm_mean"]
    df_valid = df.dropna(subset=component_cols).copy()

    if df_valid.empty:
        raise ValueError(
            f"No rows with complete component data in {paper_results_path}. "
            "Check delta_m_mean / eta_priv_mean / eta_comm_mean columns."
        )

    # Z-score parameters estimated from the valid rows (same pool for all w1 sweeps)
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in component_cols:
        vals = df_valid[col].astype(float)
        means[col] = float(vals.mean())
        stds[col] = float(vals.std(ddof=1))

    def _recompute_pcmu(w1: float) -> pd.Series:
        w2 = (1.0 - w1) * (2.0 / 3.0)
        w3 = (1.0 - w1) * (1.0 / 3.0)
        dm_z = (df_valid["delta_m_mean"].astype(float) - means["delta_m_mean"]) / stds["delta_m_mean"]
        ep_z = (df_valid["eta_priv_mean"].astype(float) - means["eta_priv_mean"]) / stds["eta_priv_mean"]
        ec_z = (df_valid["eta_comm_mean"].astype(float) - means["eta_comm_mean"]) / stds["eta_comm_mean"]
        return w1 * dm_z + w2 * ep_z + w3 * ec_z

    W1_SWEEP = [0.50, 0.60, 0.70, 0.80]
    DEFAULT_W1 = 0.70

    scores_default = _recompute_pcmu(DEFAULT_W1)
    rank_default = scores_default.rank(ascending=False)

    records = []
    for w1 in W1_SWEEP:
        w2 = (1.0 - w1) * (2.0 / 3.0)
        w3 = (1.0 - w1) * (1.0 / 3.0)
        scores = _recompute_pcmu(w1)
        rho, pval = spearmanr(rank_default, scores.rank(ascending=False))
        records.append({
            "w1": round(w1, 2),
            "w2": round(w2, 4),
            "w3": round(w3, 4),
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(pval), 6),
        })

    result_df = pd.DataFrame(records)

    # Print table
    print("\n[Weight Sensitivity Table — PCMU ranking Spearman ρ vs. default w1=0.70]")
    print(f"{'w1':>6}  {'w2':>7}  {'w3':>7}  {'Spearman ρ':>12}  {'p-value':>10}")
    print("-" * 52)
    for _, row in result_df.iterrows():
        print(
            f"{row['w1']:>6.2f}  {row['w2']:>7.4f}  {row['w3']:>7.4f}"
            f"  {row['spearman_rho']:>12.4f}  {row['p_value']:>10.6f}"
        )

    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building PCMU surrogate from factorial data...")
    interp_dm, interp_ep, interp_ec, z_params, cen_raw = build_surrogate()
    pcmu_fn = make_pcmu_fn(interp_dm, interp_ep, interp_ec, z_params, cen_raw)

    print("\n[Morris screening]")
    morris_df = run_morris(pcmu_fn)
    print(morris_df.to_string(index=False))

    print("\n[Sobol indices]")
    sobol_df = run_sobol(pcmu_fn)
    print(sobol_df.to_string(index=False))

    # Save sensitivity results
    out_csv = "results/pcmu_phase4_sensitivity.csv"
    merged = morris_df.merge(sobol_df, on="input")
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # Figures
    import os; os.makedirs("Manuscript/figures", exist_ok=True)
    plot_sensitivity(morris_df, sobol_df, pcmu_fn, "Manuscript/figures/S9_PCMUSensitivity.png")

    # Weight sensitivity table (standalone — uses pcmu_paper_results.csv directly)
    compute_weight_sensitivity_table()


if __name__ == "__main__":
    main()
