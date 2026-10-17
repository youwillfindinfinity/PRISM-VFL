
"""
experiments/run_exp1.py — Exp 1: Task heterogeneity vs. homogeneity.

Compares:
  VFL-MTL      : 3 sites, 3 tasks (ihm + decomp + pheno)
  Per-site single-task baselines: ST-IHM, ST-Decomp, ST-Pheno

Seeds: [42, 123, 7]
Output: results/exp1.csv
  columns: model, seed, round, train_loss, ihm_loss, decomp_loss, pheno_loss,
           val_ihm_auroc, val_ihm_auprc, val_decomp_auroc, val_decomp_auprc,
           val_pheno_macro_auroc, elapsed_s

Usage:
    # Real data (on Snellius):
    python experiments/run_exp1.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
        --n_rounds 50 --device gpu_h100

    # Smoke test (local, no data):
    python experiments/run_exp1.py --n_rounds 3 --use_synthetic
"""

import argparse
import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_config import BATCH_SIZE, MAX_ROUNDS, PATIENCE, SEEDS
from train import run_training, TrainConfig
from data_prep.dataset import build_site_loaders


CONFIGS = {
    # Equal task weights (manuscript Eq. 3).
    "VFL-MTL": {
        "task_weights": {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
    },
    # Per-site single-task baselines: each trains on its own site's task only.
    # ST-IHM (Site A), ST-Decomp (Site B), ST-Pheno (Site C) together form the
    # local single-task reference — MTL contribution is measured against these.
    "ST-IHM": {
        "task_weights": {"ihm": 1.0, "decomp": 0.0, "pheno": 0.0},
    },
    "ST-Decomp": {
        "task_weights": {"ihm": 0.0, "decomp": 1.0, "pheno": 0.0},
    },
    "ST-Pheno": {
        "task_weights": {"ihm": 0.0, "decomp": 0.0, "pheno": 1.0},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=MAX_ROUNDS)
    parser.add_argument("--batch_size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--device",        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output",        default="results/exp1.csv")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use random synthetic data (smoke test, no real data needed)")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--patience",      type=int, default=PATIENCE,
                        help="Early stopping patience in rounds (0 = disabled)")
    args = parser.parse_args()

    # Pre-build loaders once — all configs share the same data, only task weights differ.
    # Avoids 12 repeated GPFS reads (4 configs × 3 seeds) that dominate runtime.
    if args.use_synthetic:
        prebuilt = None
    else:
        print("[exp1] Pre-loading data loaders (one-time GPFS read)...")
        project_root = Path(args.splits_dir).parents[1]
        prebuilt = {
            "train": build_site_loaders(project_root, "train", args.batch_size),
            "val":   build_site_loaders(project_root, "val",   args.batch_size),
        }
        print("[exp1] Data loaded. Starting training runs...")

    all_rows = []

    for model_name, model_cfg in CONFIGS.items():
        for seed in SEEDS:
            print(f"\n=== {model_name} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                use_fedavg=True,
                fedavg_every=5,
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
                model_name=model_name,
                patience=args.patience,
                **model_cfg,
            )
            results = run_training(cfg, prebuilt_loaders=prebuilt)
            for r in results:
                all_rows.append({"model": model_name, "seed": seed, **r})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 1 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
