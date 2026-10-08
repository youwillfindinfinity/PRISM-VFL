"""
revision_scripts/run_prism_privacy_sweep.py — task-heterogeneity + privacy-utility
sweep under one identical, seed-matched configuration.

Replaces run_exp1.py + privacy_utility_curves.py for the headline comparison (both
stay untouched, still runnable standalone).

Arms per seed:
  1. PRISM, equal weighting, full epsilon grid            -> weighting="equal"
  2. PRISM, uncertainty weighting, eps in ABLATION_EPS     -> weighting="uncertainty"
  3. ST-IHM / ST-Decomp / ST-Pheno, eps=inf only           -> weighting="n/a"

Usage:
    python revision_scripts/run_prism_privacy_sweep.py --use_synthetic --n_rounds 3 \
        --output smoke_tests/revision1/sweep.csv --ckpt_dir smoke_tests/revision1/checkpoints
    python revision_scripts/run_prism_privacy_sweep.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits --n_rounds 100 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig
from data_prep.dataset import build_site_loaders
from experiments.run_exp1 import CONFIGS as ST_CONFIGS
from experiments.privacy_utility_curves import (
    _compute_sigma, _build_uniform_privacy_config, _epsilon_label,
)

SEEDS = [7, 42, 123, 0, 1, 2, 3, 4, 5, 6]
EPSILON_LEVELS = [float("inf"), 10.0, 5.0, 2.0, 1.0, 0.5]
ABLATION_EPS = [float("inf"), 5.0]   # uncertainty-weighting side experiment only here

# Fixed upfront (not derived from the first row) so DP-only columns from later
# rows are never silently dropped by DictWriter's extrasaction="ignore".
FIELDNAMES = [
    "arm", "weighting", "epsilon_level", "seed", "round",
    "train_loss", "ihm_loss", "decomp_loss", "pheno_loss", "elapsed_s",
    "grad_norm_ihm", "grad_norm_decomp", "grad_norm_pheno",
    "grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno",
    "epsilon_ihm", "epsilon_decomp", "epsilon_pheno",
    "sigma_ihm", "sigma_decomp", "sigma_pheno",
    "val_ihm_auroc", "val_ihm_auprc", "val_decomp_auroc", "val_decomp_auprc",
    "val_pheno_macro_auroc", "val_ihm_loss", "val_decomp_loss", "val_pheno_loss",
    "best_round", "stopping_round",
]


def _check_row_alignment(splits_dir: str) -> None:
    """Log-only sanity check: do the three site CSVs share subject_id row order?"""
    splits_dir = Path(splits_dir)
    aligned_path = splits_dir / "aligned_patient_ids.csv"
    if not aligned_path.exists():
        print("[check] aligned_patient_ids.csv not found — skipping row-alignment check")
        return
    aligned = pd.read_csv(aligned_path)
    ids = {s: set(aligned.loc[aligned.split == s, "subject_id"]) for s in ("train", "val")}
    site_csvs = {"A": "site_A_vitals.csv", "B": "site_B_labs.csv", "C": "site_C_composite.csv"}
    for split in ("train", "val"):
        rows = {}
        for site, fname in site_csvs.items():
            df = pd.read_csv(splits_dir / fname, usecols=["subject_id", "split"])
            df = df[df.split == split].reset_index(drop=True)
            df = df[df.subject_id.isin(ids[split])].reset_index(drop=True)
            rows[site] = df["subject_id"].tolist()
        ok = rows["A"] == rows["B"] == rows["C"]
        print(f"[check] {split} row alignment across sites: {'OK' if ok else 'MISMATCH'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",  default="data/vertical_splits")
    parser.add_argument("--n_rounds",    type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",      default="results_revision/prism_privacy_sweep.csv")
    parser.add_argument("--ckpt_dir",    default="checkpoints_revision",
                        help="Separate from checkpoints/ to avoid colliding with run_exp1.py "
                             "checkpoints at overlapping seeds (7, 42, 123).")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    # Synthetic runs never write to the real output locations unless told to explicitly.
    if args.use_synthetic:
        if args.output == parser.get_default("output"):
            args.output = "smoke_tests/revision1/sweep.csv"
        if args.ckpt_dir == parser.get_default("ckpt_dir"):
            args.ckpt_dir = "smoke_tests/revision1/checkpoints"

    if not args.use_synthetic:
        _check_row_alignment(args.splits_dir)

    decomp_pos_weight = 1.0
    if not args.use_synthetic:
        site_b_csv = Path(args.splits_dir) / "site_B_labs.csv"
        _b = pd.read_csv(site_b_csv, usecols=["y_decomp", "split"])
        pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
        decomp_pos_weight = (1.0 - pos_rate) / pos_rate
        print(f"[sweep] decomp pos_weight={decomp_pos_weight:.1f} (pos_rate={pos_rate:.3%})")

    base_kwargs = dict(
        splits_dir=args.splits_dir, n_rounds=args.n_rounds, batch_size=args.batch_size,
        device=args.device, patience=args.patience, ckpt_dir=args.ckpt_dir,
        use_fedavg=True, fedavg_every=5, eval_every=1, grad_sim_every=5,
        use_synthetic=args.use_synthetic, n_synthetic=args.n_synthetic,
        decomp_pos_weight=decomp_pos_weight,
    )

    if args.use_synthetic:
        n_batches = max(1, args.n_synthetic // args.batch_size)
        sample_rate = 1.0 / n_batches
    else:
        project_root = Path(args.splits_dir).parents[1]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume: seeds already in the output are kept and skipped, never overwritten.
    done = set()
    if out_path.exists() and out_path.stat().st_size:
        done = set(pd.read_csv(out_path, usecols=["seed"])["seed"])
    out_file = open(out_path, "a" if done else "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not done:
        writer.writeheader()
    total_rows = 0

    for seed in args.seeds:
        if seed in done:
            print(f"[sweep] seed={seed} already in {out_path} — skipping")
            continue
        seed_rows: list[dict] = []
        print(f"\n########## seed={seed} ##########")
        g = torch.Generator().manual_seed(seed)
        if args.use_synthetic:
            prebuilt = None
        else:
            prebuilt = {
                "train": build_site_loaders(project_root, "train", args.batch_size, generator=g),
                "val":   build_site_loaders(project_root, "val",   args.batch_size, generator=g),
                "decomp_pos_weight": decomp_pos_weight,
            }
            sample_rate = 1.0 / max(len(prebuilt["train"]["A"]), 1)

        def _run(model_name, arm, eps, weighting, uncertainty_weighting, task_weights):
            sigma = _compute_sigma(eps, sample_rate, args.n_rounds) if math.isfinite(eps) else None
            privacy_cfg = _build_uniform_privacy_config(sigma)
            cfg = TrainConfig(
                seed=seed, model_name=model_name, task_weights=task_weights,
                uncertainty_weighting=uncertainty_weighting, privacy_config=privacy_cfg,
                **base_kwargs,
            )
            print(f"=== {model_name} | eps={_epsilon_label(eps)} | seed={seed} ===")
            g.manual_seed(seed)  # every arm of this seed sees the same batch order
            rows = run_training(cfg, prebuilt_loaders=prebuilt)
            for r in rows:
                r.update(arm=arm, weighting=weighting, epsilon_level=_epsilon_label(eps), seed=seed)
            return rows

        # Arm 1 — PRISM, equal weighting, full grid
        for eps in EPSILON_LEVELS:
            name = f"PRISM-equal-eps{_epsilon_label(eps)}-seed{seed}"
            seed_rows += _run(name, "PRISM", eps, "equal", False,
                               {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0})

        # Arm 2 — PRISM, uncertainty weighting, ablation eps only
        for eps in ABLATION_EPS:
            name = f"PRISM-uncertainty-eps{_epsilon_label(eps)}-seed{seed}"
            seed_rows += _run(name, "PRISM", eps, "uncertainty", True,
                               {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0})

        # Arm 3 — single-task baselines, eps=inf only, never under DP
        for st_name, st_cfg in ST_CONFIGS.items():
            if st_name == "VFL-MTL":
                continue
            seed_rows += _run(f"{st_name}-seed{seed}", st_name, float("inf"), "n/a",
                               st_cfg.get("uncertainty_weighting", False), st_cfg["task_weights"])

        # Write this seed's rows immediately — a crash later only loses the
        # in-progress seed, not the whole run.
        writer.writerows(seed_rows)
        out_file.flush()
        total_rows += len(seed_rows)
        print(f"[sweep] seed={seed} done, {len(seed_rows)} rows written "
              f"({total_rows} total so far) -> {out_path}")

    out_file.close()
    print(f"\n[sweep] Done. {total_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
