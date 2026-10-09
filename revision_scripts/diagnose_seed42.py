"""
revision_scripts/diagnose_seed42.py — seed-42 training-instability probes.

Five PRISM runs at seed 42, one factor changed per probe, everything else (data
order, architecture, weighting) held fixed via a freshly-seeded generator per run:
  baseline        - no change
  init_from_123   - encoder+server weights start from a seed-123 initialisation
  grad_balance    - cut-layer embedding gradients rescaled to unit norm
  patience30      - early-stopping patience 15 -> 30
  decomp_ablated  - decomp task weight set to 0

Usage:
    python revision_scripts/diagnose_seed42.py --use_synthetic --n_rounds 3 \
        --output smoke_tests/revision1/probes.csv --ckpt_dir smoke_tests/revision1/checkpoints
    python revision_scripts/diagnose_seed42.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits --n_rounds 100 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig
from data_prep.dataset import build_site_loaders

SEED = 42
INIT_DONOR_SEED = 123
PROBES = ["baseline", "init_from_123", "grad_balance", "patience30", "decomp_ablated"]

# Fixed upfront (not derived from the first row), consistent with run_prism_privacy_sweep.py.
FIELDNAMES = [
    "probe", "round", "train_loss", "ihm_loss", "decomp_loss", "pheno_loss", "elapsed_s",
    "grad_norm_ihm", "grad_norm_decomp", "grad_norm_pheno",
    "grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno",
    "val_ihm_auroc", "val_ihm_auprc", "val_decomp_auroc", "val_decomp_auprc",
    "val_pheno_macro_auroc", "val_ihm_loss", "val_decomp_loss", "val_pheno_loss",
    "best_round", "stopping_round",
]


def _build_prebuilt(args):
    if args.use_synthetic:
        return None
    g = torch.Generator().manual_seed(SEED)
    project_root = Path(args.splits_dir).parents[1]
    train = build_site_loaders(project_root, "train", args.batch_size, generator=g, align_stays=True)
    val   = build_site_loaders(project_root, "val",   args.batch_size, generator=g, align_stays=True)
    # Class weight from the rows actually trained on (after alignment).
    pos_rate = float(train["B"].dataset.labels.mean())
    return {"train": train, "val": val, "decomp_pos_weight": (1.0 - pos_rate) / pos_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",  default="data/vertical_splits")
    parser.add_argument("--n_rounds",    type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",      default="results_revision/seed42_probes.csv")
    parser.add_argument("--ckpt_dir",    default="checkpoints_revision")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic", type=int, default=256)
    args = parser.parse_args()

    # Synthetic runs never write to the real output locations unless told to explicitly.
    if args.use_synthetic:
        if args.output == parser.get_default("output"):
            args.output = "smoke_tests/revision1/probes.csv"
        if args.ckpt_dir == parser.get_default("ckpt_dir"):
            args.ckpt_dir = "smoke_tests/revision1/checkpoints"

    base_kwargs = dict(
        splits_dir=args.splits_dir, n_rounds=args.n_rounds, batch_size=args.batch_size,
        device=args.device, ckpt_dir=args.ckpt_dir, use_fedavg=True, fedavg_every=5,
        eval_every=1, grad_sim_every=1, use_synthetic=args.use_synthetic,
        n_synthetic=args.n_synthetic,
        task_weights={"ihm": 1.0, "decomp": 1.0, "pheno": 1.0}, uncertainty_weighting=False,
    )

    # One cheap round to produce the seed-123 init donor checkpoint.
    donor_kwargs = {**base_kwargs, "n_rounds": 1}
    donor_cfg = TrainConfig(
        seed=INIT_DONOR_SEED, model_name="probe-init-donor",
        save_init_checkpoint=True, **donor_kwargs,
    )
    print("=== building seed-123 init donor ===")
    run_training(donor_cfg, prebuilt_loaders=_build_prebuilt(args))
    init_donor_path = f"{args.ckpt_dir}/init_probe-init-donor_seed{INIT_DONOR_SEED}.pt"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(out_path, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    total_rows = 0

    for probe in PROBES:
        probe_kwargs = {**base_kwargs, "patience": args.patience}
        if probe == "init_from_123":
            probe_kwargs["init_from"] = init_donor_path
        elif probe == "grad_balance":
            probe_kwargs["grad_balance"] = True
        elif probe == "patience30":
            probe_kwargs["patience"] = 30
        elif probe == "decomp_ablated":
            probe_kwargs["task_weights"] = {"ihm": 1.0, "decomp": 0.0, "pheno": 1.0}

        cfg = TrainConfig(seed=SEED, model_name=f"probe-{probe}", **probe_kwargs)
        print(f"\n=== probe={probe} ===")
        rows = run_training(cfg, prebuilt_loaders=_build_prebuilt(args))
        for r in rows:
            r["probe"] = probe

        writer.writerows(rows)
        out_file.flush()
        total_rows += len(rows)
        print(f"[probes] probe={probe} done, {len(rows)} rows written "
              f"({total_rows} total so far) -> {out_path}")

    out_file.close()
    print(f"\n[probes] Done. {total_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
