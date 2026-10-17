"""
experiments/run_exp2.py — Exp 2: Task relatedness and negative transfer.

Compares four task-weight configurations to measure how task relatedness
affects IHM performance:
  all_tasks  : ihm=1 + decomp=1 + pheno=1  (full MTL)
  ihm_only   : ihm=1 + decomp=0 + pheno=0  (single-task baseline)
  ihm_decomp : ihm=1 + decomp=1 + pheno=0  (related pair: acute outcomes)
  ihm_pheno  : ihm=1 + decomp=0 + pheno=1  (unrelated pair: acute vs. chronic)

Negative transfer rate = fraction of seeds where IHM val AUC under an MTL
config drops below the ihm_only baseline.

In synthetic mode val AUC evaluation is skipped (random labels → meaningless AUC).

Seeds: [42, 123, 7]
Output: results/exp2.csv

Usage:
    # Real data (on Snellius):
    python experiments/run_exp2.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
        --n_rounds 50 --device cpu

    # Smoke test (local, no data):
    python experiments/run_exp2.py --n_rounds 3 --use_synthetic
"""

import argparse
import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_config import BATCH_SIZE, MAX_ROUNDS, PATIENCE, SEEDS
from train import run_training, TrainConfig
from data_prep.dataset import build_site_loaders


# Equal weights for every active task (manuscript Eq. 3).
TASK_CONFIGS = {
    "all_tasks":   {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
    "ihm_only":    {"ihm": 1.0, "decomp": 0.0, "pheno": 0.0},
    "ihm_decomp":  {"ihm": 1.0, "decomp": 1.0, "pheno": 0.0},
    "ihm_pheno":   {"ihm": 1.0, "decomp": 0.0, "pheno": 1.0},
}


def compute_negative_transfer(rows: list[dict]) -> None:
    """
    Print negative transfer rate: fraction of (config, seed) pairs where
    final-round val IHM AUC < ihm_only baseline for the same seed.
    Skipped if val metrics are absent (synthetic mode).
    """
    if not any("val_ihm_auroc" in r for r in rows):
        return

    # Last round per (config, seed)
    final: dict[tuple, float] = {}
    for r in rows:
        key = (r["task_config"], r["seed"])
        if "val_ihm_auroc" in r:
            final[key] = r["val_ihm_auroc"]

    neg_transfer = 0
    total = 0
    for config_name in TASK_CONFIGS:
        if config_name == "ihm_only":
            continue
        for seed in SEEDS:
            baseline = final.get(("ihm_only", seed))
            config_val = final.get((config_name, seed))
            if baseline is not None and config_val is not None:
                total += 1
                if config_val < baseline:
                    neg_transfer += 1

    if total > 0:
        print(f"\nNegative transfer rate: {neg_transfer}/{total} "
              f"({100*neg_transfer/total:.1f}%) configs where MTL < single-task IHM AUC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=MAX_ROUNDS)
    parser.add_argument("--batch_size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--device",        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output",        default="results/exp2.csv")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use random synthetic data (smoke test, no real data needed)")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--patience",      type=int, default=PATIENCE,
                        help="Early stopping patience in rounds (0 = disabled)")
    parser.add_argument("--task_config",   default=None,
                        help="Run only this task config (default: all). "
                             "One of: all_tasks, ihm_only, ihm_decomp, ihm_pheno")
    args = parser.parse_args()

    if args.use_synthetic:
        prebuilt = None
    else:
        print("[exp2] Pre-loading data loaders (one-time GPFS read)...")
        project_root = Path(args.splits_dir).parents[1]
        prebuilt = {
            "train": build_site_loaders(project_root, "train", args.batch_size),
            "val":   build_site_loaders(project_root, "val",   args.batch_size),
        }
        print("[exp2] Data loaded. Starting training runs...")

    # Filter to a single config if requested
    configs_to_run = TASK_CONFIGS
    if args.task_config is not None:
        if args.task_config not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_config '{args.task_config}'. "
                             f"Choose from: {list(TASK_CONFIGS)}")
        configs_to_run = {args.task_config: TASK_CONFIGS[args.task_config]}

    all_rows = []

    for config_name, task_weights in configs_to_run.items():
        for seed in SEEDS:
            print(f"\n=== config={config_name} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                model_name=f"exp2_{config_name}",
                use_fedavg=True,
                fedavg_every=5,
                task_weights=task_weights,
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
                patience=args.patience,
            )
            results = run_training(cfg, prebuilt_loaders=prebuilt)
            for r in results:
                all_rows.append({"task_config": config_name, "seed": seed, **r})

    compute_negative_transfer(all_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 3 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
