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
η_comm  : communication efficiency — log(1 + R_ST / R), R_ST from ST-IHM at the same ε;
          R = total training rounds (primary) or rounds to the best-validation checkpoint
          (sensitivity). Rounds are a proxy for bytes only at a fixed embed_dim.

Evaluation entry-point
----------------------
    python experiments/compute_pcmu.py          # sanity-check on exp1 + privacy CSV
    python experiments/compute_pcmu.py --help
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

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
    round_def: str = "total"   # R for η_comm: "total" rounds trained or "best" (checkpoint round)
    embed_dim: int = 64        # sweep embed_dim; selects the ST reference and the payload size
    n_train: int = 14_679      # aligned training stays (audit_revision/dp_accounting_audit.json)

    def __post_init__(self) -> None:
        if self.round_def not in ("total", "best"):
            raise ValueError(f"round_def must be 'total' or 'best', got {self.round_def!r}")
        total = sum(self.task_weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"task_weights must sum to 1.0, got {total:.6f}")
        total_p = sum(self.pcmu_weights.values())
        if not math.isclose(total_p, 1.0, abs_tol=1e-6):
            raise ValueError(f"pcmu_weights must sum to 1.0, got {total_p:.6f}")


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

def best_round(df: pd.DataFrame, tasks: list[str]) -> int:
    """First round with the highest mean val AUROC over the active tasks (training's checkpoint rule)."""
    df = df.sort_values("round")
    score = df[[_TASK_AUC_COLS[t] for t in tasks]].mean(axis=1)
    return int(df.loc[score.idxmax(), "round"])


def rounds_used(df: pd.DataFrame, tasks: list[str], round_def: str) -> int:
    """R of one run: total rounds trained (rounds are 1-based) or the best-checkpoint round."""
    return int(df["round"].max()) if round_def == "total" else best_round(df, tasks)


def best_round_aurocs(df: pd.DataFrame, tasks: list[str]) -> dict[str, float]:
    """Val AUROC per task at the run's best-checkpoint round."""
    row = df[df["round"] == best_round(df, tasks)].iloc[0]
    return {t: float(row[c]) for t, c in _TASK_AUC_COLS.items()}


def bytes_per_round(embed_dim: int, n_train: int, n_sites: int = 3) -> int:
    """Each site sends one float32 embed_dim vector per training stay and receives a same-size gradient."""
    return 2 * n_sites * n_train * embed_dim * 4


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


# ── High-level evaluators ──────────────────────────────────────────────────

def st_reference_rounds(
    factorial_rounds_df: pd.DataFrame,
    eps_levels: list[float],
    seeds: set,
    cfg: PCMUConfig,
) -> dict[float, float]:
    """
    R_ST per ε: mean R of ST-IHM (factorial ihm_only at cfg.embed_dim) over seeds.

    Raises if any ε lacks a same-ε reference covering every seed of the sweep.
    """
    st = factorial_rounds_df[
        (factorial_rounds_df["task_config"] == "ihm_only")
        & (factorial_rounds_df["embed_dim"] == cfg.embed_dim)
    ]
    r_ref: dict[float, float] = {}
    for eps, e in st.groupby("epsilon_level"):
        if seeds <= set(e["seed"]):
            r_ref[eps] = float(np.mean([
                rounds_used(e[e["seed"] == s], ["ihm"], cfg.round_def) for s in seeds
            ]))
    missing = sorted(eps for eps in eps_levels if eps not in r_ref)
    if missing:
        raise ValueError(
            f"No same-ε ST-IHM reference (ihm_only, embed_dim={cfg.embed_dim}, "
            f"seeds={sorted(seeds)}) for ε = {missing}"
        )
    return r_ref


def raw_components(
    runs_df: pd.DataFrame,
    st_aurocs: dict[str, float],
    factorial_rounds_df: pd.DataFrame,
    cfg: PCMUConfig,
) -> pd.DataFrame:
    """
    Pass 1: raw Δ_m, η_priv, η_comm per (ε, seed) run of runs_df (per-round val curves).

    AUROCs at each run's best-checkpoint round; η_priv against the ε=∞ runs of runs_df;
    R_ST from st_reference_rounds() at the same ε and cfg.embed_dim.
    """
    tasks = [t for t, w in cfg.task_weights.items() if w > 0]
    r_ref = st_reference_rounds(
        factorial_rounds_df, list(runs_df["epsilon_level"].unique()), set(runs_df["seed"]), cfg,
    )
    payload = bytes_per_round(cfg.embed_dim, cfg.n_train)

    runs = {
        (eps, seed): (best_round_aurocs(g, tasks), rounds_used(g, tasks, cfg.round_def))
        for (eps, seed), g in runs_df.groupby(["epsilon_level", "seed"])
    }
    nodp = [a for (eps, _), (a, _) in runs.items() if eps == float("inf")]
    nodp_aurocs = {t: float(np.mean([a[t] for a in nodp])) for t in _TASK_AUC_COLS}

    rows = []
    for (eps_val, seed_val), (mtl_aurocs, r) in runs.items():
        dm = multitask_gain(mtl_aurocs, st_aurocs, cfg.task_weights)
        ep = privacy_efficiency(mtl_aurocs, nodp_aurocs, cfg.task_weights)
        ec = comm_efficiency(float(r), r_ref[eps_val])
        rows.append({
            "epsilon_level":  eps_val,
            "seed":           seed_val,
            "delta_m":        dm,
            "eta_priv":       ep,
            "eta_comm":       ec,
            "pcmu_geometric": compute_pcmu_geometric(dm, ep, ec),
            "r":              r,
            "r_ref":          r_ref[eps_val],
            "comm_bytes":     r * payload,
            **{f"auroc_{t}": mtl_aurocs.get(t, float("nan")) for t in cfg.task_weights},
        })
    return pd.DataFrame(rows)


def additive_pcmu(
    df: pd.DataFrame,
    weights: dict[str, float],
    anchor: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Pass 2: z-score components over df, apply weights, shift so the anchor rows
    (default: ε=∞) average PCMU = 1.0 (PCMUmetric.md §Reformulation).
    """
    df = df.copy()
    for col in ("delta_m", "eta_priv", "eta_comm"):
        vals = df[col].dropna()
        sigma = float(vals.std(ddof=1))
        df[f"{col}_z"] = (df[col] - float(vals.mean())) / sigma if sigma > 0 else 0.0
    df["pcmu_additive"] = sum(weights[c] * df[f"{c}_z"] for c in ("delta_m", "eta_priv", "eta_comm"))
    if anchor is None:
        anchor = df["epsilon_level"] == float("inf")
    df["pcmu"] = df["pcmu_additive"] - float(df.loc[anchor, "pcmu_additive"].mean()) + 1.0
    return df


def evaluate_sweep(
    privacy_df: pd.DataFrame,
    exp1_df: pd.DataFrame,
    factorial_rounds_df: pd.DataFrame,
    mode: str = "uniform",
    config: PCMUConfig | None = None,
) -> pd.DataFrame:
    """
    Additive PCMU for every ε level of the per-round PRISM validation curves:
    raw_components() then additive_pcmu() over the sweep pool.

    Output columns: delta_m, eta_priv, eta_comm (raw), their _z variants,
    pcmu_additive (pre-shift), pcmu (shifted, final), pcmu_geometric (archived),
    r, r_ref, comm_bytes.
    """
    cfg = config or PCMUConfig()
    subset: pd.DataFrame = privacy_df[privacy_df["mode"] == mode]  # type: ignore[assignment]
    df = raw_components(subset, load_st_aurocs(exp1_df), factorial_rounds_df, cfg)
    df.insert(2, "mode", mode)
    return additive_pcmu(df, cfg.pcmu_weights)


def factorial_components(
    factorial_rounds_df: pd.DataFrame,
    st_aurocs: dict[str, float],
    config: PCMUConfig | None = None,
) -> pd.DataFrame:
    """
    raw_components() for every all_tasks factorial run, each embed_dim against its own
    same-ε ST-IHM and ε=∞ references, so the per-round payload cancels in η_comm.
    """
    cfg = config or PCMUConfig()
    fact = factorial_rounds_df
    return pd.concat([
        raw_components(
            fact[(fact["task_config"] == "all_tasks") & (fact["embed_dim"] == d)],
            st_aurocs, fact, replace(cfg, embed_dim=int(d)),
        ).assign(embed_dim=int(d))
        for d in sorted(fact["embed_dim"].unique())
    ], ignore_index=True)


# ── CLI entry-point ────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="PCMU metric computation (additive form)")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--mode", default="uniform", choices=["uniform", "stratified"])
    parser.add_argument("--round_def", default="total", choices=["total", "best"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    exp1_path   = results_dir / "exp1.csv"
    priv_path   = results_dir / "privacy_utility_combined.csv"
    cen_path    = results_dir / "centralized.csv"
    fact_path   = results_dir / "pcmu_phase2_factorial_rounds.csv"

    for p in (exp1_path, priv_path, cen_path, fact_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    exp1_df    = pd.read_csv(exp1_path)
    privacy_df = pd.read_csv(priv_path)
    cen_df     = pd.read_csv(cen_path)

    print(f"Centralized oracle AUROCs: {load_centralized_aurocs(cen_df)}")

    result_df = evaluate_sweep(privacy_df, exp1_df, pd.read_csv(fact_path), mode=args.mode,
                               config=PCMUConfig(round_def=args.round_def))

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
