"""
experiments/evaluate_exp2.py — Test-set evaluation for Exp 2 (task relatedness).

Loads the best checkpoint for each task-weight configuration and evaluates on
the held-out test split. Architecture is inferred from checkpoint keys.

Configurations: all_tasks, ihm_only, ihm_decomp, ihm_pheno
Seeds: [42, 123, 7]

Output: results/test_exp2.csv
  columns: task_config, seed, ihm_auc_roc, ihm_auc_pr,
           decomp_auc_roc, decomp_auc_pr, pheno_macro_auc, pheno_micro_auc

Usage:
    python experiments/evaluate_exp2.py --root /home/asoare/vfl_mlt
    python experiments/evaluate_exp2.py --root . --use_synthetic
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import build_site_loaders
from experiments.metrics import ihm_metrics, decomp_metrics, pheno_metrics
from fl.client import VFLClient
from fl.server import VFLServer

SEEDS = [42, 123, 7]

TASK_CONFIGS = ["all_tasks", "ihm_only", "ihm_decomp", "ihm_pheno"]

# Which tasks were trained (non-zero weight) per config.
ACTIVE_TASKS = {
    "all_tasks":  {"ihm", "decomp", "pheno"},
    "ihm_only":   {"ihm"},
    "ihm_decomp": {"ihm", "decomp"},
    "ihm_pheno":  {"ihm", "pheno"},
}


def _infer_config(ckpt: dict) -> dict:
    proj      = ckpt["client_A"]["projection.weight"]   # (embed_dim, hidden_dim)
    lstm      = ckpt["client_A"]["lstm.weight_ih_l0"]   # (4*hidden_dim, input_dim)
    embed_dim  = int(proj.shape[0])
    hidden_dim = int(lstm.shape[0] // 4)

    srv = ckpt["server"]
    use_mmoe = "shared_bottom.0.weight" not in srv
    n_experts = sum(
        1 for k in srv
        if k.startswith("mmoe.experts.") and k.endswith(".net.0.weight")
    ) if use_mmoe else 0

    return dict(embed_dim=embed_dim, hidden_dim=hidden_dim,
                use_mmoe=use_mmoe, num_experts=n_experts)


@torch.no_grad()
def _eval_vfl(clients: dict, server: VFLServer, loaders: dict,
              active: set[str]) -> dict[str, float]:
    for c in clients.values():
        c.encoder.eval()
    server.model.eval()

    preds  = {"ihm": [], "decomp": [], "pheno": []}
    labels = {"ihm": [], "decomp": [], "pheno": []}

    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA
        xB, mB, yD = bB
        xC, mC, yP = bC
        embs = {
            "A": clients["A"].eval_forward(xA, mA),
            "B": clients["B"].eval_forward(xB, mB),
            "C": clients["C"].eval_forward(xC, mC),
        }
        out = server.predict(embs)
        if "ihm"    in active: preds["ihm"].append(out["ihm"].squeeze(-1).cpu().numpy())
        if "decomp" in active: preds["decomp"].append(out["decomp"].squeeze(-1).cpu().numpy())
        if "pheno"  in active: preds["pheno"].append(out["pheno"].cpu().numpy())
        labels["ihm"].append(yI.numpy())
        labels["decomp"].append(yD.numpy())
        labels["pheno"].append(yP.numpy())

    nan_ihm   = {f"ihm_{k}":    float("nan") for k in ihm_metrics(np.zeros(1), np.zeros(1))}
    nan_decomp = {f"decomp_{k}": float("nan") for k in decomp_metrics(np.zeros(1), np.zeros(1))}
    nan_pheno  = {f"pheno_{k}":  float("nan") for k in pheno_metrics(np.zeros((1,25)), np.zeros((1,25)))}

    result = {**nan_ihm, **nan_decomp, **nan_pheno}

    y_ihm = np.concatenate(labels["ihm"])
    y_dec = np.concatenate(labels["decomp"])
    y_phn = np.concatenate(labels["pheno"])

    if "ihm"    in active:
        result.update({f"ihm_{k}":    v for k, v in ihm_metrics(y_ihm, np.concatenate(preds["ihm"])).items()})
    if "decomp" in active:
        result.update({f"decomp_{k}": v for k, v in decomp_metrics(y_dec, np.concatenate(preds["decomp"])).items()})
    if "pheno"  in active:
        result.update({f"pheno_{k}":  v for k, v in pheno_metrics(y_phn, np.concatenate(preds["pheno"])).items()})

    return result


def eval_exp2(ckpt_dir: Path, loaders: dict, device: str = "cpu") -> list[dict]:
    rows = []
    dev  = torch.device(device)

    for config_name in TASK_CONFIGS:
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_exp2_{config_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name} not found")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            cfg  = _infer_config(ckpt)

            clients = {
                s: VFLClient(input_dim=d,
                             hidden_dim=cfg["hidden_dim"],
                             embed_dim=cfg["embed_dim"],
                             lr=1e-3, device=dev)
                for s, d in [("A", 7), ("B", 4), ("C", 3)]
            }
            server = VFLServer(
                embed_dim=cfg["embed_dim"],
                use_mmoe=cfg["use_mmoe"],
                num_experts=cfg["num_experts"] if cfg["use_mmoe"] else 4,
                device=dev,
            )
            clients["A"].encoder.load_state_dict(ckpt["client_A"])
            clients["B"].encoder.load_state_dict(ckpt["client_B"])
            clients["C"].encoder.load_state_dict(ckpt["client_C"])
            server.model.load_state_dict(ckpt["server"])

            active = ACTIVE_TASKS[config_name]
            m = _eval_vfl(clients, server, loaders, active)
            rows.append({"task_config": config_name, "seed": seed, **m})
            print(f"  {config_name} seed={seed}: "
                  f"IHM={m['ihm_auc_roc']:.4f} "
                  + (f"Decomp={m['decomp_auc_roc']:.4f} " if "decomp" in active else "")
                  + (f"Pheno={m['pheno_macro_auc']:.4f}" if "pheno" in active else ""))

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",          default=".")
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output",        default="results/test_exp2.csv")
    p.add_argument("--use_synthetic", action="store_true")
    args = p.parse_args()

    ckpt_dir = Path(args.root) / args.ckpt_dir

    print("=" * 60)
    print("EXP 3 TEST-SET EVALUATION")
    print("=" * 60)

    if args.use_synthetic:
        N, T = 64, 48
        g = torch.Generator(); g.manual_seed(999)
        loaders = {
            "A": DataLoader(TensorDataset(
                torch.randn(N, T, 7), torch.ones(N, T),
                torch.randint(0, 2, (N,), generator=g).float()),
                batch_size=args.batch_size, drop_last=True),
            "B": DataLoader(TensorDataset(
                torch.randn(N, T, 4), torch.ones(N, T),
                torch.randint(0, 2, (N,), generator=g).float()),
                batch_size=args.batch_size, drop_last=True),
            "C": DataLoader(TensorDataset(
                torch.randn(N, T, 3), torch.ones(N, T),
                torch.randint(0, 2, (N, 25), generator=g).float()),
                batch_size=args.batch_size, drop_last=True),
        }
    else:
        loaders = build_site_loaders(Path(args.root), "test", args.batch_size)

    all_rows = eval_exp2(ckpt_dir, loaders, args.device)

    if not all_rows:
        print("[WARNING] No checkpoints found.")
        return

    fields = ["task_config", "seed", "ihm_auc_roc", "ihm_auc_pr",
              "decomp_auc_roc", "decomp_auc_pr",
              "pheno_macro_auc", "pheno_micro_auc"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nExp 3 test results -> {args.output}")


if __name__ == "__main__":
    main()
