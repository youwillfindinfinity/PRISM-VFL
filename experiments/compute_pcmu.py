"""
experiments/compute_pcmu.py

Privacy-Communication-adjusted Multi-task Utility (PCMU) metric — additive form.

    PCMU = w1·Δm_z + w2·η_priv_z + w3·η_comm_z − PCMU_baseline + 1.0

where x_z = (x − μ) / σ, z-scored over the full evaluation pool (all configurations
including the ε=∞ baseline). The shift anchors 1.0 = centralized baseline (ε=∞ proxy).

Weights (Rieke et al. 2020 clinical deployment priority hierarchy):
    w1=0.70  multi-task utility gain
    w2=0.20  privacy efficiency
    w3=0.10  communication efficiency

Reformulation rationale
-----------------------
Phase 2 ANOVA (2026-05-10) found η²≥0.12 for embed_dim×ε interactions across all
components, with |r|≥0.51 partial correlations between Δ_m and η_priv / η_comm.
ε is a structural driver of all three components simultaneously; multiplicative
aggregation compounds this shared variance three-fold. Additive aggregation with
z-score normalisation handles correlated inputs correctly (Nardo et al. 2008).
See PCMUmetric.md §Reformulation for full rationale.

Geometric form archived as compute_pcmu_geometric(); gate results in
results/pcmu_geom_results/.

Components
----------
Δ_m     : multi-task gain weighted over active tasks (Maninis et al. CVPR 2019)
η_priv  : privacy efficiency — utility-under-DP relative to no-DP reference
η_comm  : communication efficiency — log-ratio of convergence rounds

Evaluation entry-point
----------------------
    python experiments/compute_pcmu.py          # sanity-check on exp1 + privacy CSV
    python experiments/compute_pcmu.py --help
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class PCMUConfig:
    """Hyperparameters for PCMU computation. All defaults from PCMUmetric.md."""

    task_weights: dict[str, float] = field(
        default_factory=lambda: {"ihm": 0.5, "decomp": 0.3, "pheno": 0.2}
    )
    # Component weights for additive PCMU (Rieke et al. 2020 priority hierarchy)
    pcmu_weights: dict[str, float] = field(
        default_factory=lambda: {"delta_m": 0.70, "eta_priv": 0.20, "eta_comm": 0.10}
    )
    conv_threshold: float = 0.90  # fraction of per-run max AUC for convergence_round

    def __post_init__(self) -> None:
        total = sum(self.task_weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"task_weights must sum to 1.0, got {total:.6f}")
        total_p = sum(self.pcmu_weights.values())
        if not math.isclose(total_p, 1.0, abs_tol=1e-6):
            raise ValueError(f"pcmu_weights must sum to 1.0, got {total_p:.6f}")


# ── Result container ───────────────────────────────────────────────────────

class PCMUResult(NamedTuple):
    delta_m: float
    eta_priv: float       # nan when ε = ∞ (no DP, privacy term undefined)
    eta_comm: float
    pcmu: float           # additive form; nan when any component is nan
    pcmu_geometric: float # archived geometric form


# ── Column mapping ─────────────────────────────────────────────────────────

_TASK_AUC_COLS: dict[str, str] = {
    "ihm":    "val_ihm_auroc",
    "decomp": "val_decomp_auroc",
    "pheno":  "val_pheno_macro_auroc",
}

_CEN_AUC_COLS: dict[str, str] = {
    "ihm":    "val_ihm_auc_roc",
    "decomp": "val_decomp_auc_roc",
    "pheno":  "val_pheno_macro_auc",
}


# ── Core math ─────────────────────────────────────────────────────────────

def multitask_gain(
    mtl_aurocs: dict[str, float],
    st_aurocs: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Δ_m = (1/T) Σ_t  w_t · (M_t^MTL − M_t^ST) / M_t^ST

    T counts only tasks with non-NaN values in both dicts and positive weight.
    """
    tasks = [
        t for t in weights
        if weights[t] > 0
        and math.isfinite(mtl_aurocs.get(t, float("nan")))
        and math.isfinite(st_aurocs.get(t, float("nan")))
        and st_aurocs.get(t, 0.0) > 0
    ]
    if not tasks:
        raise ValueError(
            "multitask_gain: no task has finite AUROC in both mtl_aurocs and st_aurocs. "
            f"Received mtl={mtl_aurocs}, st={st_aurocs}."
        )
    T = len(tasks)
    return sum(
        weights[t] * (mtl_aurocs[t] - st_aurocs[t]) / st_aurocs[t]
        for t in tasks
    ) / T


def privacy_efficiency(
    aurocs: dict[str, float],
    cen_aurocs: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    η_priv = Σ_t  w_t · (M_t / M_t^cen)

    Plain weighted utility ratio. At ε=∞ (no DP) this is ≈ 1.0; falls as DP
    degrades utility. Log factor removed (Option A, PCMUmetric.md §Option A).
    """
    tasks = [
        t for t in weights
        if weights[t] > 0
        and math.isfinite(aurocs.get(t, float("nan")))
        and math.isfinite(cen_aurocs.get(t, float("nan")))
        and cen_aurocs.get(t, 0.0) > 0
    ]
    if not tasks:
        raise ValueError(
            "privacy_efficiency: no task has finite AUROC in both aurocs and cen_aurocs."
        )
    return sum(weights[t] * (aurocs[t] / cen_aurocs[t]) for t in tasks)


def comm_efficiency(r: float, r_ref: float) -> float:
    """η_comm = log(1 + R_ref / R)"""
    if r <= 0:
        raise ValueError(f"R must be positive, got {r}")
    if r_ref <= 0:
        raise ValueError(f"R_ref must be positive, got {r_ref}")
    return math.log(1.0 + r_ref / r)


def compute_pcmu_geometric(delta_m: float, eta_priv: float, eta_comm: float) -> float:
    """
    Archived geometric form: PCMU = ∛((1 + Δ_m) · η_priv · η_comm)

    Superseded by the additive form. See PCMUmetric.md §Reformulation.
    Gate results from this form: results/pcmu_geom_results/pcmu_phase2_gate.csv.
    """
    if any(math.isnan(x) for x in (delta_m, eta_priv, eta_comm)):
        return float("nan")
    product = (1.0 + delta_m) * eta_priv * eta_comm
    if product < 0:
        return float("nan")
    return product ** (1.0 / 3.0)


def compute_pcmu_additive(
    delta_m_z: float,
    eta_priv_z: float,
    eta_comm_z: float,
    weights: dict[str, float],
) -> float:
    """
    PCMU_add = w1·Δm_z + w2·η_priv_z + w3·η_comm_z

    Inputs must be z-scored over the evaluation pool before calling.
    Apply the baseline shift (− PCMU_baseline + 1.0) after pooled computation.
    Returns nan if any component is nan.
    """
    vals = {"delta_m": delta_m_z, "eta_priv": eta_priv_z, "eta_comm": eta_comm_z}
    if any(math.isnan(v) for v in vals.values()):
        return float("nan")
    return sum(weights[k] * vals[k] for k in weights)


# Backward-compatibility alias
compute_pcmu = compute_pcmu_geometric


# ── Task-relatedness (Phase 1c) ────────────────────────────────────────────

def hellinger_binary(p: float, q: float) -> float:
    """Hellinger distance between Bernoulli(p) and Bernoulli(q). Bounded [0, 1]."""
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError(f"Prevalences must be in [0, 1], got p={p}, q={q}")
    sq = (math.sqrt(p) - math.sqrt(q)) ** 2 + (math.sqrt(1 - p) - math.sqrt(1 - q)) ** 2
    return math.sqrt(sq) / math.sqrt(2.0)


def hellinger_multilabel(p_vec: np.ndarray, q_vec: np.ndarray) -> float:
    """Mean Hellinger distance across label dimensions for multi-label tasks."""
    assert p_vec.shape == q_vec.shape, "p_vec and q_vec must have the same shape"
    dists = []
    for p, q in zip(p_vec, q_vec):
        if p == 0.0 and q == 0.0:
            continue
        dists.append(hellinger_binary(float(np.clip(p, 0, 1)), float(np.clip(q, 0, 1))))
    if not dists:
        return float("nan")
    return float(np.mean(dists))


def task_relatedness_from_prevalences(
    prevalences: dict[str, float | np.ndarray],
) -> dict[tuple[str, str], float]:
    """
    Pairwise Hellinger distances between task label distributions.

    Lower distance = more related tasks (same direction as Δ_m hypothesis in Phase 1c).
    """
    tasks = list(prevalences.keys())
    out: dict[tuple[str, str], float] = {}
    for i, t_i in enumerate(tasks):
        for t_j in tasks[i + 1:]:
            p_i, p_j = prevalences[t_i], prevalences[t_j]
            if isinstance(p_i, np.ndarray) or isinstance(p_j, np.ndarray):
                arr_i = np.atleast_1d(p_i).astype(float)
                arr_j = np.atleast_1d(p_j).astype(float)
                if arr_i.shape != arr_j.shape:
                    if arr_i.size == 1:
                        arr_i = np.full_like(arr_j, arr_i[0])
                    else:
                        arr_j = np.full_like(arr_i, arr_j[0])
                dist = hellinger_multilabel(arr_i, arr_j)
            else:
                dist = hellinger_binary(float(p_i), float(p_j))
            out[(t_i, t_j)] = dist
    return out


# ── Data extraction helpers ────────────────────────────────────────────────

def peak_aurocs(df: pd.DataFrame) -> dict[str, float]:
    """Peak val AUROC per task across all rows in df."""
    out: dict[str, float] = {}
    for task, col in _TASK_AUC_COLS.items():
        if col in df.columns:
            valid = df[col].dropna()
            out[task] = float(valid.max()) if not valid.empty else float("nan")
        else:
            out[task] = float("nan")
    return out


def convergence_round(
    df: pd.DataFrame,
    auc_col: str,
    *,
    threshold: float = 0.90,
    round_col: str = "round",
) -> int:
    """
    First round where auc_col ≥ threshold × max(auc_col across all rounds).

    Returns -1 if auc_col is absent or entirely NaN.
    Returns the final round if the threshold is never crossed.
    """
    if auc_col not in df.columns:
        return -1
    vals = df[auc_col].dropna()
    if vals.empty:
        return -1
    target = threshold * float(vals.max())
    for _, row in df.sort_values(round_col).iterrows():
        v = row.get(auc_col, float("nan"))
        if math.isfinite(float(v)) and float(v) >= target:
            return int(row[round_col])
    return int(df[round_col].max())


def convergence_round_mean(
    df: pd.DataFrame,
    auc_col: str,
    *,
    threshold: float = 0.90,
    round_col: str = "round",
    seed_col: str = "seed",
) -> float:
    """Mean convergence round across seeds in df."""
    rounds = []
    for _, seed_df in df.groupby(seed_col):
        r = convergence_round(seed_df, auc_col, threshold=threshold, round_col=round_col)
        if r > 0:
            rounds.append(r)
    return float(np.mean(rounds)) if rounds else float("nan")


def load_st_aurocs(exp1: pd.DataFrame) -> dict[str, float]:
    """Mean-across-seeds peak val AUROC per task from single-task VFL baselines."""
    mapping: dict[str, tuple[str, str]] = {
        "ihm":    ("ST-IHM",    "val_ihm_auroc"),
        "decomp": ("ST-Decomp", "val_decomp_auroc"),
        "pheno":  ("ST-Pheno",  "val_pheno_macro_auroc"),
    }
    out: dict[str, float] = {}
    for task, (model_name, col) in mapping.items():
        rows = exp1[exp1["model"] == model_name]
        if rows.empty or col not in rows.columns:
            out[task] = float("nan")
            continue
        per_seed = rows.groupby("seed")[col].max()
        out[task] = float(per_seed.mean())
    return out


def load_mtl_aurocs(
    df: pd.DataFrame,
    model_col: str = "model",
    model_val: str = "VFL-MTL",
) -> dict[str, float]:
    """Mean-across-seeds peak val AUROC per task from VFL-MTL rows."""
    subset = df[df[model_col] == model_val] if model_col in df.columns else df
    out: dict[str, float] = {}
    for task, col in _TASK_AUC_COLS.items():
        if col not in subset.columns:
            out[task] = float("nan")
            continue
        per_seed = subset.groupby("seed")[col].max() if "seed" in subset.columns else subset[col]
        valid = per_seed.dropna()
        out[task] = float(valid.mean()) if not valid.empty else float("nan")
    return out


def load_centralized_aurocs(centralized_df: pd.DataFrame) -> dict[str, float]:
    """Peak AUROC from centralized oracle CSV."""
    out: dict[str, float] = {}
    for task, col in _CEN_AUC_COLS.items():
        if col in centralized_df.columns:
            valid = centralized_df[col].dropna()
            out[task] = float(valid.max()) if not valid.empty else float("nan")
        else:
            out[task] = float("nan")
    return out


def load_nodp_aurocs(
    privacy_df: pd.DataFrame,
    mode: str = "uniform",
) -> dict[str, float]:
    """Peak val AUROC at ε=∞ (no DP) from privacy_utility_combined.csv."""
    nodp: pd.DataFrame = privacy_df[  # type: ignore[assignment]
        (privacy_df["epsilon_level"] == float("inf")) &
        (privacy_df["mode"] == mode)
    ]
    return load_mtl_aurocs(nodp, model_col="__none__")


# ── High-level evaluators ──────────────────────────────────────────────────

def evaluate(
    *,
    eps: float,
    mtl_aurocs: dict[str, float],
    st_aurocs: dict[str, float],
    cen_aurocs: dict[str, float],
    r: float,
    r_ref: float,
    config: PCMUConfig | None = None,
) -> PCMUResult:
    """
    Compute raw PCMU components for a single configuration.

    pcmu field contains the geometric form (for single-run use or comparison).
    For the validated additive PCMU use evaluate_sweep(), which z-scores
    components over the full population before aggregating.
    """
    cfg = config or PCMUConfig()
    dm  = multitask_gain(mtl_aurocs, st_aurocs, cfg.task_weights)
    ep  = privacy_efficiency(eps, mtl_aurocs, cen_aurocs, cfg.task_weights, cfg.eps_ref)
    ec  = comm_efficiency(r, r_ref)
    geom = compute_pcmu_geometric(dm, ep, ec)
    return PCMUResult(
        delta_m=dm,
        eta_priv=ep,
        eta_comm=ec,
        pcmu=geom,
        pcmu_geometric=geom,
    )


def evaluate_sweep(
    privacy_df: pd.DataFrame,
    exp1_df: pd.DataFrame,
    mode: str = "uniform",
    config: PCMUConfig | None = None,
) -> pd.DataFrame:
    """
    Compute additive PCMU for every ε level in privacy_utility_combined.csv.

    Two-pass pipeline
    -----------------
    Pass 1 — compute raw Δ_m, η_priv, η_comm per (ε, seed).
    Pass 2 — z-score components over the full pool, apply additive weights,
             shift so PCMU = 1.0 at ε=∞ baseline.

    Output columns: delta_m, eta_priv, eta_comm (raw), their _z variants,
    pcmu_additive (pre-shift), pcmu (shifted, final), pcmu_geometric (archived).
    """
    cfg = config or PCMUConfig()
    st_aurocs = load_st_aurocs(exp1_df)

    nodp_df: pd.DataFrame = privacy_df[  # type: ignore[assignment]
        (privacy_df["epsilon_level"] == float("inf")) &
        (privacy_df["mode"] == mode)
    ]
    nodp_aurocs = load_mtl_aurocs(nodp_df, model_col="__none__")
    primary_col = "val_ihm_auroc"
    # R_ref = single-task (ihm_only) baseline, not all_tasks (Option A, PCMUmetric.md)
    r_ref = convergence_round_mean(nodp_df, primary_col, threshold=cfg.conv_threshold)

    # Pass 1: raw components
    rows = []
    subset: pd.DataFrame = privacy_df[privacy_df["mode"] == mode]  # type: ignore[assignment]
    for eps_val, eps_df in subset.groupby("epsilon_level"):
        for seed_val, seed_df in eps_df.groupby("seed"):
            seed_df = pd.DataFrame(seed_df)
            mtl_aurocs = peak_aurocs(seed_df)
            r = convergence_round(seed_df, primary_col, threshold=cfg.conv_threshold)
            if r <= 0:
                r = int(seed_df["round"].max())
            dm = multitask_gain(mtl_aurocs, st_aurocs, cfg.task_weights)
            ep = privacy_efficiency(mtl_aurocs, nodp_aurocs, cfg.task_weights)
            ec = comm_efficiency(float(r), float(r_ref))
            rows.append({
                "epsilon_level":  eps_val,
                "seed":           seed_val,
                "mode":           mode,
                "delta_m":        dm,
                "eta_priv":       ep,
                "eta_comm":       ec,
                "pcmu_geometric": compute_pcmu_geometric(dm, ep, ec),
                "r":              r,
                "r_ref":          r_ref,
                **{f"auroc_{t}": mtl_aurocs.get(t, float("nan")) for t in cfg.task_weights},
            })

    df = pd.DataFrame(rows)

    # Pass 2: z-score over the full pool
    w = cfg.pcmu_weights
    for col in ("delta_m", "eta_priv", "eta_comm"):
        vals = df[col].dropna()
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=1))
        df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    df["pcmu_additive"] = (
        w["delta_m"]  * df["delta_m_z"] +
        w["eta_priv"] * df["eta_priv_z"] +
        w["eta_comm"] * df["eta_comm_z"]
    )

    # Shift: 1.0 = ε=∞ baseline (PCMUmetric.md §Reformulation)
    baseline_mask = df["epsilon_level"] == float("inf")
    baseline_pcmu = float(df.loc[baseline_mask, "pcmu_additive"].mean())
    df["pcmu"] = df["pcmu_additive"] - baseline_pcmu + 1.0

    return df


# ── CLI entry-point ────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="PCMU metric computation (additive form)")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--mode", default="uniform", choices=["uniform", "stratified"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    exp1_path   = results_dir / "exp1.csv"
    priv_path   = results_dir / "privacy_utility_combined.csv"
    cen_path    = results_dir / "centralized.csv"

    for p in (exp1_path, priv_path, cen_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    exp1_df    = pd.read_csv(exp1_path)
    privacy_df = pd.read_csv(priv_path)
    cen_df     = pd.read_csv(cen_path)

    print(f"Centralized oracle AUROCs: {load_centralized_aurocs(cen_df)}")

    result_df = evaluate_sweep(privacy_df, exp1_df, mode=args.mode)

    summary = (
        result_df.groupby("epsilon_level")[
            ["delta_m", "eta_priv", "eta_comm", "pcmu", "pcmu_geometric"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\nPCMU summary across seeds:")
    print(summary.to_string())

    if args.out:
        result_df.to_csv(args.out, index=False)
        print(f"\nSaved per-seed results → {args.out}")


if __name__ == "__main__":
    _cli()
