"""
experiments/run_ablations.py — Week 4 architecture ablations.

All ablations compare against VFL-MTL (full system) using identical data,
seeds, and training conditions. Only the flagged component varies.

Ablations covered:
  VFL-MTL              : full system, baseline reference for this script
  abl_no_mmoe          : Abl 1 — shared-bottom MLP instead of MMoE (use_mmoe=False)
  abl_experts_2        : Abl 3 — num_experts=2
  abl_experts_4        : Abl 3 — num_experts=4 (matches VFL-MTL default; sanity check)
  abl_experts_8        : Abl 3 — num_experts=8
  abl_uniform_gating   : Abl 4 — fixed equal expert weights, no learned gating
  abl_embed_32         : Abl 5 — embed_dim=32  (less communication, less capacity)
  abl_embed_64         : Abl 5 — embed_dim=64  (matches VFL-MTL default; sanity check)
  abl_embed_128        : Abl 5 — embed_dim=128 (more capacity, more communication)
Not covered here (handled in exp1.csv):
  ST-IHM / ST-Decomp / ST-Pheno — MTL contribution ablation

Not covered here (requires PSI alignment fix before it is meaningful):
  abl_no_psi — random patient pairing ablation

Seeds: [42, 123, 7]
Output: results/ablations.csv
  columns: model, seed, round, train_loss, ihm_loss, decomp_loss, pheno_loss,
           val_ihm_auroc, val_ihm_auprc, val_decomp_auroc, val_decomp_auprc,
           val_pheno_macro_auroc, elapsed_s
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

# All ablations run all 3 tasks and all 3 sites unless a specific component varies.
_BASE = {
    "task_weights": {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
}

CONFIGS = {
    # ---- Baseline reference (matches run_exp1.py VFL-MTL config exactly) ----
    "VFL-MTL": {
        **_BASE,
        "uncertainty_weighting": True,
    },

    # ---- Abl 1: No MMoE — shared-bottom MLP, no gating ----
    "abl_no_mmoe": {
        **_BASE,
        "use_mmoe": False,
    },

    # ---- Abl 3: Expert count sensitivity ----
    "abl_experts_2": {
        **_BASE,
        "num_experts": 2,
    },
    "abl_experts_8": {
        **_BASE,
        "num_experts": 8,
    },

    # ---- Abl 4: Uniform gating — no learned expert routing ----
    "abl_uniform_gating": {
        **_BASE,
        "uniform_gating": True,
    },

    # ---- Abl 5: Embedding dimension sensitivity ----
    "abl_embed_32": {
        **_BASE,
        "embed_dim": 32,
    },
    "abl_embed_128": {
        **_BASE,
        "embed_dim": 128,
    },

}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output",        default="results/ablations.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--patience",      type=int, default=15)
    args = parser.parse_args()

    decomp_pos_weight = 1.0
    prebuilt = None

    if not args.use_synthetic:
        print("[ablations] Pre-loading data loaders (one-time GPFS read)...")
        project_root = Path(args.splits_dir).parents[1]
        site_b_csv = Path(args.splits_dir) / "site_B_labs.csv"
        _b = pd.read_csv(site_b_csv, usecols=["y_decomp", "split"])
        pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
        decomp_pos_weight = (1.0 - pos_rate) / pos_rate
        print(f"[ablations] decomp pos_weight={decomp_pos_weight:.1f} (pos_rate={pos_rate:.3%})")
        prebuilt = {
            "train": build_site_loaders(project_root, "train", args.batch_size),
            "val":   build_site_loaders(project_root, "val",   args.batch_size),
            "decomp_pos_weight": decomp_pos_weight,
        }
        print("[ablations] Data loaded. Starting ablation runs...")

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
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
                model_name=model_name,
                patience=args.patience,
                decomp_pos_weight=decomp_pos_weight,
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

    print(f"\nAblations complete. Results → {args.output}")


if __name__ == "__main__":
    main()
