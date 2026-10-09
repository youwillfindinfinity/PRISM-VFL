"""
revision_scripts/evaluate_test_prism_privacy_sweep.py — test-set scores for the sweep's models.

For every run in the sweep CSV: load the checkpoint saved at that run's best_round
(bestmean_<model>_seed<seed>.pt) and score it on the held-out test split, with the
sites aligned on shared ICU stays as in training.

Usage:
    python revision_scripts/evaluate_test_prism_privacy_sweep.py --use_synthetic
    python revision_scripts/evaluate_test_prism_privacy_sweep.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import _evaluate_sites, make_synthetic_loaders
from data_prep.dataset import build_site_loaders
from fl.client import VFLClient
from fl.server import VFLServer

SITE_DIMS = {"A": 7, "B": 4, "C": 3}
SITES = list(SITE_DIMS)
RUN_KEYS = ["arm", "weighting", "epsilon_level", "seed", "best_round"]
METRICS = ["ihm_auroc", "ihm_auprc", "decomp_auroc", "decomp_auprc", "pheno_macro_auroc"]


def _model_name(run) -> str:
    """Same names as run_prism_privacy_sweep.py."""
    if run.arm == "PRISM":
        return f"PRISM-{run.weighting}-eps{run.epsilon_level}-seed{run.seed}"
    return f"{run.arm}-seed{run.seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",  default="data/vertical_splits")
    parser.add_argument("--input",       default="results_revision/prism_privacy_sweep.csv")
    parser.add_argument("--ckpt_dir",    default="checkpoints_revision")
    parser.add_argument("--output",      default="results_revision/prism_privacy_sweep_test.csv")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    # Synthetic runs never touch the real input/output locations unless told to explicitly.
    if args.use_synthetic:
        if args.input == parser.get_default("input"):
            args.input = "smoke_tests/revision1/sweep.csv"
        if args.ckpt_dir == parser.get_default("ckpt_dir"):
            args.ckpt_dir = "smoke_tests/revision1/checkpoints"
        if args.output == parser.get_default("output"):
            args.output = "smoke_tests/revision1/sweep_test.csv"

    # Read as text: keeps "inf"/"5.0" and "n/a" exactly as the sweep wrote them.
    runs = pd.read_csv(args.input, usecols=RUN_KEYS, dtype=str,
                       keep_default_na=False).drop_duplicates()

    if args.use_synthetic:
        loaders = make_synthetic_loaders(args.batch_size, 48, 4)
    else:
        loaders = build_site_loaders(Path(args.splits_dir).parents[1], "test",
                                     args.batch_size, align_stays=True)
        stays = [loaders[s].dataset.stays for s in SITES]
        assert stays[0] == stays[1] == stays[2], "test: stays differ across sites"
        print(f"[eval] test stays, same row at every site: {len(stays[0])}")

    clients = {s: VFLClient(input_dim=SITE_DIMS[s], device=args.device) for s in SITES}
    server = VFLServer(device=args.device)

    rows = []
    for run in runs.itertuples(index=False):
        name = _model_name(run)
        ckpt = torch.load(Path(args.ckpt_dir) / f"bestmean_{name}_seed{run.seed}.pt",
                          weights_only=True, map_location=args.device)
        assert int(ckpt["round"]) == int(run.best_round), (
            f"{name}: checkpoint is from round {ckpt['round']}, sweep CSV says best_round="
            f"{run.best_round} — checkpoint and CSV come from different runs"
        )
        for s in SITES:
            clients[s].encoder.load_state_dict(ckpt[f"client_{s}"])
        server.model.load_state_dict(ckpt["server"])

        metrics = _evaluate_sites(clients, server, loaders, SITES, SITE_DIMS, SITE_DIMS)
        rows.append({**run._asdict(), **{f"test_{k}": metrics[k] for k in METRICS}})
        print(f"[eval] {name}: " + "  ".join(f"{k}={metrics[k]:.4f}" for k in METRICS))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[eval] Done. {len(rows)} runs -> {out_path}")


if __name__ == "__main__":
    main()
