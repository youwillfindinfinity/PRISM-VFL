"""
experiments/run_phase2_factorial.py — PCMU Phase 2: Cross-factor independence gate.

Full factorial: embed_dim × privacy level (ε) × task count.
Every cell run at 3 seeds. Required by PCMUmetric.md Phase 2 before PCMU can be
defended as a multiplicative composite.

Grid
----
  embed_dim   : {32, 64, 128}
  epsilon     : {∞ (no DP), 5.0, 1.0, 0.5}
  task_config : {all_tasks (3), ihm_decomp (2), ihm_only (1)}
  seeds       : [42, 123, 7]

Total cells: 3 × 4 × 3 = 36 configurations × 3 seeds = 108 runs.

Outputs
-------
  results/phase2_factorial.csv        — one row per (embed_dim, eps, task_config, seed)
  results/phase2_factorial_rounds.csv — per-round val metrics for convergence-locked R

Convergence round R
-------------------
Logged in phase2_factorial.csv as `convergence_round`. Definition (PCMUmetric.md Phase 3):
the first round at which val AUC for the primary active task exceeds
CONV_THRESHOLD × (max val AUC over all rounds in that run). Computed within the script
from per-round data; the Phase 3 convergence-locked R will be recomputed post-hoc via
experiments/evaluate_phase2.py using a single cross-configuration threshold.

Usage
-----
  # Smoke test (no data, 5 rounds):
  python experiments/run_phase2_factorial.py --use_synthetic --n_rounds 5

  # Real data (Snellius):
  python experiments/run_phase2_factorial.py \\
      --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \\
      --n_rounds 100 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_training, TrainConfig, make_synthetic_loaders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS        = [42, 123, 7]
DELTA        = 1e-5
MAX_GRAD_NORM = 1.0
CONV_THRESHOLD = 0.90   # fraction of per-run max AUC; replaced by locked threshold in Phase 3

EMBED_DIMS   = [32, 64, 128]
EPS_LEVELS   = [float("inf"), 5.0, 1.0, 0.5]

# Three task configurations covering task_count ∈ {1, 2, 3}
TASK_CONFIGS: dict[str, dict] = {
    "all_tasks":  {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
    "ihm_decomp": {"ihm": 1.0, "decomp": 1.0, "pheno": 0.0},
    "ihm_only":   {"ihm": 1.0, "decomp": 0.0, "pheno": 0.0},
}

SUMMARY_COLS = [
    "embed_dim", "epsilon_level", "task_config", "task_count", "seed",
    "sigma",
    "val_ihm_auroc", "val_decomp_auroc", "val_pheno_macro_auroc",
    "convergence_round",
    "grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno",
]

ROUND_COLS = [
    "embed_dim", "epsilon_level", "task_config", "seed", "round",
    "val_ihm_auroc", "val_decomp_auroc", "val_pheno_macro_auroc",
    "train_loss",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sigma(eps: float, sample_rate: float, n_rounds: int) -> float | None:
    if not math.isfinite(eps):
        return None
    from opacus.accountants.utils import get_noise_multiplier
    return float(get_noise_multiplier(
        target_epsilon=eps,
        target_delta=DELTA,
        sample_rate=sample_rate,
        epochs=n_rounds,
    ))


def _primary_auc_key(task_weights: dict) -> str:
    """Return the primary metric key for convergence tracking (highest-weight active task)."""
    order = [("ihm", "val_ihm_auroc"), ("decomp", "val_decomp_auroc"),
             ("pheno", "val_pheno_macro_auroc")]
    for task, key in order:
        if task_weights.get(task, 0.0) > 0:
            return key
    return "val_ihm_auroc"


def _convergence_round(per_round: list[dict], auc_key: str) -> int:
    """
    First round where auc_key ≥ CONV_THRESHOLD × max(auc_key over all rounds).
    Returns -1 if the key is absent or all values are NaN.
    """
    vals = [r.get(auc_key, float("nan")) for r in per_round]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return -1
    threshold = CONV_THRESHOLD * max(finite)
    for r in per_round:
        v = r.get(auc_key, float("nan"))
        if math.isfinite(v) and v >= threshold:
            return int(r["round"])
    return int(per_round[-1]["round"])


def _mean_grad_sim(results: list[dict]) -> dict[str, float]:
    """Mean pairwise grad-sim values across rounds (NaN if absent)."""
    out = {}
    for key in ("grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno"):
        vals = [r[key] for r in results if key in r and math.isfinite(r.get(key, float("nan")))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run_cell(
    *,
    embed_dim: int,
    eps: float,
    task_config: str,
    seed: int,
    args: argparse.Namespace,
    sample_rate: float,
) -> tuple[dict, list[dict]]:
    """
    Run one factorial cell. Returns (summary_row, per_round_rows).
    """
    task_weights = TASK_CONFIGS[task_config]
    task_count = sum(1 for v in task_weights.values() if v > 0)
    sigma = _compute_sigma(eps, sample_rate, args.n_rounds)

    privacy_config: dict | None = None
    if sigma is not None:
        privacy_config = {
            "mode": "uniform",
            "sigma": sigma,
            "max_grad_norm": MAX_GRAD_NORM,
            "delta": DELTA,
        }

    cfg = TrainConfig(
        seed=seed,
        n_rounds=args.n_rounds,
        device=args.device,
        embed_dim=embed_dim,
        splits_dir=args.splits_dir,
        task_weights=task_weights,
        uncertainty_weighting=(task_count > 1),
        privacy_config=privacy_config,
        use_synthetic=args.use_synthetic,
        batch_size=args.batch_size,
        grad_sim_every=1,
    )

    label = (
        f"embed{embed_dim}_eps{eps if math.isfinite(eps) else 'inf'}"
        f"_{task_config}_seed{seed}"
    )
    print(f"[phase2] {label}")

    results = run_training(cfg, None)  # run_training builds loaders from cfg.splits_dir

    # Per-round rows
    auc_key = _primary_auc_key(task_weights)
    round_rows = []
    for r in results:
        round_rows.append({
            "embed_dim":            embed_dim,
            "epsilon_level":        eps,
            "task_config":          task_config,
            "seed":                 seed,
            "round":                r.get("round", float("nan")),
            "val_ihm_auroc":        r.get("val_ihm_auroc",       float("nan")),
            "val_decomp_auroc":     r.get("val_decomp_auroc",    float("nan")),
            "val_pheno_macro_auroc": r.get("val_pheno_macro_auroc", float("nan")),
            "train_loss":           r.get("train_loss",          float("nan")),
        })

    last = results[-1] if results else {}
    grad_sim = _mean_grad_sim(results)
    conv_round = _convergence_round(round_rows, auc_key)

    summary = {
        "embed_dim":              embed_dim,
        "epsilon_level":          eps,
        "task_config":            task_config,
        "task_count":             task_count,
        "seed":                   seed,
        "sigma":                  sigma if sigma is not None else float("nan"),
        "val_ihm_auroc":          last.get("val_ihm_auroc",       float("nan")),
        "val_decomp_auroc":       last.get("val_decomp_auroc",    float("nan")),
        "val_pheno_macro_auroc":  last.get("val_pheno_macro_auroc", float("nan")),
        "convergence_round":      conv_round,
        **grad_sim,
    }
    return summary, round_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PCMU Phase 2 full factorial")
    parser.add_argument("--splits_dir",   default="data/vertical_splits")
    parser.add_argument("--n_rounds",     type=int, default=100)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--batch_size",   type=int, default=256)
    # Allow targeting a sub-grid for parallelism, e.g. --embed_dim 32 --eps 5
    parser.add_argument("--embed_dim",    type=int, default=None,
                        help="Run only this embed_dim (default: all)")
    parser.add_argument("--eps",          type=float, default=None,
                        help="Run only this ε level (default: all)")
    parser.add_argument("--task_config",  default=None,
                        help="Run only this task config (default: all)")
    parser.add_argument("--out_summary", default=None,
                        help="Override summary CSV path (default: results/phase2_factorial.csv)")
    parser.add_argument("--out_rounds",  default=None,
                        help="Override rounds CSV path (default: results/phase2_factorial_rounds.csv)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip cells already present in the summary CSV (resume mode)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    summary_path = Path(args.out_summary) if args.out_summary else out_dir / "phase2_factorial.csv"
    rounds_path  = Path(args.out_rounds)  if args.out_rounds  else out_dir / "phase2_factorial_rounds.csv"

    # Determine the sub-grid
    embed_dims  = [args.embed_dim] if args.embed_dim is not None else EMBED_DIMS
    eps_levels  = [args.eps]       if args.eps       is not None else EPS_LEVELS
    task_cfgs   = [args.task_config] if args.task_config is not None else list(TASK_CONFIGS)

    total = len(embed_dims) * len(eps_levels) * len(task_cfgs) * len(SEEDS)
    print(f"[phase2] grid: embed={embed_dims} eps={eps_levels} tasks={task_cfgs} "
          f"seeds={SEEDS} → {total} runs")

    # Build set of already-completed (embed_dim, eps, task_config, seed) keys for resume mode.
    existing: set[tuple] = set()
    if args.skip_existing and summary_path.exists():
        import pandas as _pd
        try:
            _done = _pd.read_csv(summary_path)
            for _, _r in _done.iterrows():
                existing.add((int(_r["embed_dim"]), float(_r["epsilon_level"]),
                              str(_r["task_config"]), int(_r["seed"])))
            print(f"[phase2] skip_existing: {len(existing)} cells already done, skipping them")
        except _pd.errors.EmptyDataError:
            print(f"[phase2] skip_existing: CSV exists but is empty, starting fresh")

    sample_rate = args.batch_size / 10000.0  # conservative estimate for sigma pre-computation

    # Open CSV writers (append-safe: check if file already exists)
    write_summary_header = not summary_path.exists()
    write_rounds_header  = not rounds_path.exists()

    with (
        open(summary_path, "a", newline="") as sf,
        open(rounds_path,  "a", newline="") as rf,
    ):
        sw = csv.DictWriter(sf, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        rw = csv.DictWriter(rf, fieldnames=ROUND_COLS,   extrasaction="ignore")
        if write_summary_header:
            sw.writeheader()
        if write_rounds_header:
            rw.writeheader()

        for embed_dim, eps, task_cfg, seed in product(embed_dims, eps_levels, task_cfgs, SEEDS):
            if (embed_dim, eps, task_cfg, seed) in existing:
                print(f"[phase2] skip {embed_dim} eps={eps} {task_cfg} seed={seed}")
                continue
            summary_row, round_rows = run_cell(
                embed_dim=embed_dim,
                eps=eps,
                task_config=task_cfg,
                seed=seed,
                args=args,
                sample_rate=sample_rate,
            )
            sw.writerow(summary_row)
            sf.flush()
            rw.writerows(round_rows)
            rf.flush()

    print(f"[phase2] done → {summary_path}  |  {rounds_path}")


if __name__ == "__main__":
    main()
