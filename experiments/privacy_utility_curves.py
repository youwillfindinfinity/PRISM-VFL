"""
experiments/privacy_utility_curves.py — ε sweep (SRQ2 + SRQ3).

Uniform sweep: ε ∈ {∞ (no DP), 10, 5, 2, 1, 0.5}; 5 seeds per level.
Stratified run: ε_total=5 allocated as ε_IHM=2, ε_Decomp=2, ε_Pheno=1.

For each (mode, ε_level, seed) triple, one round-level row is written per round:
  round, seed, epsilon_level, mode,
  val_ihm_auroc, val_ihm_auprc,
  val_decomp_auroc, val_decomp_auprc,
  val_pheno_macro_auroc,
  epsilon_ihm, epsilon_decomp, epsilon_pheno,  ← from RenyiAccountant
  train_loss, elapsed_s

Convergence round (first round where val_AUC ≥ 0.90 × no-DP plateau) is
derived in figures/privacy_utility_plot.py from this CSV — not computed here.

Usage
-----
  # Smoke test (synthetic, fast):
  python experiments/privacy_utility_curves.py --use_synthetic --n_rounds 5

  # Full run on Snellius:
  python experiments/privacy_utility_curves.py \
      --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
      --n_rounds 100 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_training, TrainConfig
from data_prep.dataset import build_site_loaders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 7]
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

# Uniform ε sweep — float('inf') = no DP baseline
EPSILON_LEVELS = [float("inf"), 10.0, 5.0, 2.0, 1.0, 0.5]

# Stratified allocation at ε_total=5: tighter budget for clinical-risk hierarchy.
# ε_IHM=2, ε_Decomp=2, ε_Pheno=1  (sum = 5, composition bound)
STRATIFIED_EPS = {"ihm": 2.0, "decomp": 2.0, "pheno": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sigma(
    target_epsilon: float,
    sample_rate: float,
    n_rounds: int,
    delta: float = DELTA,
) -> float | None:
    """
    Return the noise multiplier σ that achieves (target_epsilon, delta)-DP
    over n_rounds rounds with the given sample_rate.

    Returns None when target_epsilon is infinite (no-DP run).
    Raises RuntimeError if opacus cannot find a valid σ.
    """
    if not math.isfinite(target_epsilon):
        return None
    from opacus.accountants.utils import get_noise_multiplier
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=n_rounds,
        )
    )


def _build_uniform_privacy_config(
    sigma: float | None,
    delta: float = DELTA,
) -> dict | None:
    if sigma is None:
        return None
    return {
        "mode":          "uniform",
        "sigma":         sigma,
        "max_grad_norm": MAX_GRAD_NORM,
        "delta":         delta,
    }


def _build_stratified_privacy_config(
    sigma_map: dict[str, float],
    delta: float = DELTA,
) -> dict:
    return {
        "mode":          "stratified",
        "sigma_ihm":     sigma_map["ihm"],
        "sigma_decomp":  sigma_map["decomp"],
        "sigma_pheno":   sigma_map["pheno"],
        "max_grad_norm": MAX_GRAD_NORM,
        "delta":         delta,
    }


def _epsilon_label(eps: float) -> str:
    return "inf" if not math.isfinite(eps) else str(eps)


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run_one(
    cfg: TrainConfig,
    epsilon_level: float,
    mode: str,
) -> list[dict]:
    """Run training and annotate each round row with epsilon_level and mode."""
    rows = run_training(cfg)
    for r in rows:
        r["epsilon_level"] = _epsilon_label(epsilon_level)
        r["mode"] = mode
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output",        default="results/privacy_utility.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--patience",      type=int, default=0,
                        help="Early stopping patience (0 = disabled for DP runs to ensure full sweep)")
    parser.add_argument("--epsilon_levels", default=None,
                        help="Comma-separated ε levels to run, e.g. 'inf' or '5.0' or '1.0,0.5'. "
                             "Default: all levels. Use this to parallelise across Snellius jobs.")
    parser.add_argument("--run_stratified", action="store_true",
                        help="Also run the task-stratified variant (ε_total=5, σ_IHM<σ_Decomp<σ_Pheno). "
                             "Pass alongside --epsilon_levels 5.0 for the ε=5 Snellius job.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes. On Snellius with --cpus-per-task=18, "
                             "set to 4 to use reserved CPUs for data loading (no extra SBU cost).")
    args = parser.parse_args()

    # Filter ε levels if --epsilon_levels was provided
    if args.epsilon_levels is not None:
        _requested: list[float] = []
        for _s in args.epsilon_levels.split(","):
            _s = _s.strip()
            _requested.append(float("inf") if _s == "inf" else float(_s))
        epsilon_levels_to_run = [e for e in EPSILON_LEVELS if e in _requested]
        if not epsilon_levels_to_run:
            print(f"[privacy_curves] WARNING: none of {_requested} matched EPSILON_LEVELS={EPSILON_LEVELS}")
    else:
        epsilon_levels_to_run = EPSILON_LEVELS
        if not args.run_stratified:
            # Default: run everything (all levels + stratified) for backwards-compat
            args.run_stratified = True

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    decomp_pos_weight = 1.0
    prebuilt = None

    if not args.use_synthetic:
        print("[privacy_curves] Pre-loading data loaders...")
        project_root = Path(args.splits_dir).parents[1]
        site_b_csv   = Path(args.splits_dir) / "site_B_labs.csv"
        _b = pd.read_csv(site_b_csv, usecols=["y_decomp", "split"])
        pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
        decomp_pos_weight = (1.0 - pos_rate) / pos_rate
        print(f"[privacy_curves] decomp pos_weight={decomp_pos_weight:.1f}")
        prebuilt = {
            "train": build_site_loaders(project_root, "train", args.batch_size, num_workers=args.num_workers),
            "val":   build_site_loaders(project_root, "val",   args.batch_size, num_workers=args.num_workers),
            "decomp_pos_weight": decomp_pos_weight,
        }

    # ---- Sample rate for σ computation ----
    if args.use_synthetic:
        n_batches  = max(1, args.n_synthetic // args.batch_size)
        sample_rate = 1.0 / n_batches
    else:
        assert prebuilt is not None
        sample_rate = 1.0 / max(len(prebuilt["train"]["A"]), 1)

    print(f"[privacy_curves] sample_rate={sample_rate:.5f}  n_rounds={args.n_rounds}")

    # ---- Pre-compute σ for each ε level ----
    sigma_for_eps: dict[float, float | None] = {}
    for eps in EPSILON_LEVELS:
        if not math.isfinite(eps):
            sigma_for_eps[eps] = None
            print(f"[privacy_curves] ε=inf → no DP (baseline)")
        else:
            try:
                s = _compute_sigma(eps, sample_rate, args.n_rounds)
                sigma_for_eps[eps] = s
                print(f"[privacy_curves] ε={eps} → σ={s:.4f}")
            except Exception as e:
                print(f"[privacy_curves] WARNING: could not compute σ for ε={eps}: {e}")
                sigma_for_eps[eps] = None

    # ---- Pre-compute σ for stratified run ----
    stratified_sigma: dict[str, float] = {}
    stratified_ok = True
    for task, eps_t in STRATIFIED_EPS.items():
        try:
            s = _compute_sigma(eps_t, sample_rate, args.n_rounds)
            assert s is not None  # STRATIFIED_EPS contains only finite ε values
            stratified_sigma[task] = s
            print(f"[privacy_curves] stratified ε_{task}={eps_t} → σ_{task}={s:.4f}")
        except Exception as e:
            print(f"[privacy_curves] WARNING: stratified σ for task={task} failed: {e}")
            stratified_ok = False

    # ---- Base TrainConfig shared across all runs ----
    base_cfg_kwargs = dict(
        splits_dir=args.splits_dir,
        n_rounds=args.n_rounds,
        batch_size=args.batch_size,
        device=args.device,
        use_fedavg=True,
        fedavg_every=5,
        use_synthetic=args.use_synthetic,
        n_synthetic=args.n_synthetic,
        patience=args.patience,
        decomp_pos_weight=decomp_pos_weight,
        task_weights={"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
        uncertainty_weighting=True,
        eval_every=1,
        grad_sim_every=5,
    )

    # ---- CSV writer (open once, stream rows) ----
    all_rows: list[dict] = []

    # ---- Uniform ε sweep ----
    for eps in epsilon_levels_to_run:
        sigma = sigma_for_eps.get(eps)
        privacy_cfg = _build_uniform_privacy_config(sigma)
        eps_label = _epsilon_label(eps)
        mode = "uniform"

        for seed in SEEDS:
            model_name = f"DP-uniform-eps{eps_label}-seed{seed}"
            print(f"\n=== uniform | ε={eps_label} | seed={seed} | σ={sigma} ===")
            cfg = TrainConfig(
                seed=seed,
                model_name=model_name,
                privacy_config=privacy_cfg,
                **base_cfg_kwargs,
            )
            rows = run_one(cfg, eps, mode)
            for r in rows:
                r["seed"] = seed
            all_rows.extend(rows)

    # ---- Stratified run (ε_total=5) ----
    if stratified_ok and args.run_stratified:
        privacy_cfg_strat = _build_stratified_privacy_config(stratified_sigma)
        for seed in SEEDS:
            model_name = f"DP-stratified-eps5-seed{seed}"
            print(f"\n=== stratified | ε_total=5 | seed={seed} ===")
            cfg = TrainConfig(
                seed=seed,
                model_name=model_name,
                privacy_config=privacy_cfg_strat,
                **base_cfg_kwargs,
            )
            rows = run_one(cfg, 5.0, "stratified")
            for r in rows:
                r["seed"] = seed
            all_rows.extend(rows)
    elif not args.run_stratified:
        print("[privacy_curves] Skipping stratified run (not requested; pass --run_stratified with --epsilon_levels 5.0).")
    else:
        print("[privacy_curves] Skipping stratified run — σ computation failed.")

    # ---- Write CSV ----
    if not all_rows:
        print("[privacy_curves] No rows collected — exiting.")
        return

    fieldnames = list(all_rows[0].keys())
    # Ensure consistent column order regardless of which rows have which keys
    _priority = [
        "round", "seed", "epsilon_level", "mode",
        "val_ihm_auroc", "val_ihm_auprc",
        "val_decomp_auroc", "val_decomp_auprc",
        "val_pheno_macro_auroc",
        "epsilon_ihm", "epsilon_decomp", "epsilon_pheno",
        "train_loss", "ihm_loss", "decomp_loss", "pheno_loss",
        "elapsed_s",
    ]
    seen = set()
    ordered = []
    for col in _priority:
        if col in fieldnames and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in fieldnames:
        if col not in seen:
            ordered.append(col)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[privacy_curves] Done. {len(all_rows)} rows → {output_path}")
    _summarise(all_rows)


def _summarise(rows: list[dict]) -> None:
    """Print a brief summary table of final-round val AUROCs per (mode, ε_level)."""
    import collections
    groups: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        groups[(r.get("mode", "?"), r.get("epsilon_level", "?"))].append(r)

    print("\n── Privacy-Utility Summary (last round, mean across seeds) ──")
    header = f"{'mode':12s} {'ε':>8s} │ {'IHM AUC':>9s} {'Decomp AUC':>11s} {'Pheno AUC':>10s}"
    print(header)
    print("─" * len(header))

    import numpy as np
    for (mode, eps_label), group_rows in sorted(groups.items()):
        # Only the last round for each seed
        by_seed: dict[int, dict] = {}
        for r in group_rows:
            s = r.get("seed", 0)
            rnd = r.get("round", 0)
            if s not in by_seed or rnd > by_seed[s].get("round", 0):
                by_seed[s] = r
        last_rows = list(by_seed.values())

        def _mean(key):
            vals = [r[key] for r in last_rows if key in r and r[key] == r[key]]
            return float(np.mean(vals)) if vals else float("nan")

        ihm   = _mean("val_ihm_auroc")
        decomp = _mean("val_decomp_auroc")
        pheno  = _mean("val_pheno_macro_auroc")
        print(f"{mode:12s} {eps_label:>8s} │ {ihm:9.4f} {decomp:11.4f} {pheno:10.4f}")


if __name__ == "__main__":
    main()
