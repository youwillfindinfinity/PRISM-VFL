"""
experiments/run_exp3.py — Exp 3: Scalability (2 vs. 3 institutions).

Measures:
  - Communication rounds to convergence (loss delta < 0.001 over 5 rounds)
  - Wall-clock time per round
  - Per-task AUC at final round

Configurations:
  n_sites=2 : Sites A + B only (IHM + Decomp tasks; pheno weight = 0)
  n_sites=3 : All three sites (IHM + Decomp + Pheno)

Seeds: [42, 123, 7]
Output: results/exp3.csv

Usage:
    # Real data (on Snellius):
    python experiments/run_exp3.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
        --n_rounds 100 --device cpu

    # Smoke test (local, no data):
    python experiments/run_exp3.py --n_rounds 3 --use_synthetic
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig
from data_prep.dataset import build_site_loaders

SEEDS = [42, 123, 7]


def rounds_to_convergence(losses: list[float], threshold: float = 0.001, window: int = 5) -> int:
    """Return round index where total loss delta over `window` rounds drops below threshold."""
    for i in range(window, len(losses)):
        delta = abs(losses[i - window] - losses[i])
        if delta < threshold:
            return i
    return len(losses)  # did not converge within budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output",        default="results/exp3.csv")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use random synthetic data (smoke test, no real data needed)")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--patience",      type=int, default=15,
                        help="Early stopping patience in rounds (0 = disabled)")
    parser.add_argument("--n_sites",       type=int, default=None,
                        help="Run only this n_sites value (default: all). One of: 2, 3")
    args = parser.parse_args()

    if args.use_synthetic:
        prebuilt_by_nsites = {2: None, 3: None}
        decomp_pos_weight = 1.0
    else:
        print("[exp4] Pre-loading data loaders (one-time GPFS read)...")
        project_root = Path(args.splits_dir).parents[1]
        site_b_csv = Path(args.splits_dir) / "site_B_labs.csv"
        _b = pd.read_csv(site_b_csv, usecols=["y_decomp", "split"])
        pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
        decomp_pos_weight = (1.0 - pos_rate) / pos_rate
        print(f"[exp4] decomp pos_weight={decomp_pos_weight:.1f} (pos_rate={pos_rate:.3%})")
        # n_sites=3 loaders contain all sites; n_sites=2 reuses A+B from the same load.
        all_loaders = {
            "train": build_site_loaders(project_root, "train", args.batch_size),
            "val":   build_site_loaders(project_root, "val",   args.batch_size),
            "decomp_pos_weight": decomp_pos_weight,
        }
        prebuilt_by_nsites = {2: all_loaders, 3: all_loaders}
        print("[exp4] Data loaded. Starting training runs...")

    sites_to_run = [args.n_sites] if args.n_sites is not None else [2, 3]

    all_rows = []

    for n_sites in sites_to_run:
        task_weights = (
            {"ihm": 1.0, "decomp": 1.0, "pheno": 0.0}
            if n_sites == 2
            else {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0}
        )
        for seed in SEEDS:
            print(f"\n=== n_sites={n_sites} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                model_name=f"exp4_sites{n_sites}",
                n_sites=n_sites,
                task_weights=task_weights,
                uncertainty_weighting=True,
                use_fedavg=True,
                fedavg_every=5,
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
                patience=args.patience,
                decomp_pos_weight=decomp_pos_weight,
            )
            results = run_training(cfg, prebuilt_loaders=prebuilt_by_nsites[n_sites])
            losses = [r["train_loss"] for r in results]
            conv_round = rounds_to_convergence(losses)
            print(f"  Convergence round: {conv_round}/{args.n_rounds}")
            for r in results:
                all_rows.append({
                    "n_sites":          n_sites,
                    "seed":             seed,
                    "convergence_round": conv_round,
                    **r,
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 4 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
