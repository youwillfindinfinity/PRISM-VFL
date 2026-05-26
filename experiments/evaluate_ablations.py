"""
experiments/evaluate_ablations.py — Test-set evaluation for Week 4 ablations.

Loads the best checkpoint for each ablation variant and evaluates on the
held-out test split. Architecture config is inferred from the checkpoint keys
so no external config file is needed.

Models: VFL-MTL, abl_no_mmoe, abl_uniform_gating, abl_experts_2, abl_experts_8,
        abl_embed_32, abl_embed_128

Output: results/test_ablations.csv
  columns: model, seed, ihm_auc_roc, ihm_auc_pr, decomp_auc_roc, decomp_auc_pr,
           pheno_macro_auc, pheno_micro_auc

Usage:
    python experiments/evaluate_ablations.py --root /home/asoare/vfl_mlt
    python experiments/evaluate_ablations.py --root . --use_synthetic
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
from model.encoder import SiteEncoder

SEEDS = [42, 123, 7]

ABLATION_MODELS = [
    "VFL-MTL",
    "abl_no_mmoe",
    "abl_uniform_gating",
    "abl_experts_2",
    "abl_experts_8",
    "abl_embed_32",
    "abl_embed_128",
]


# ---------------------------------------------------------------------------
# Infer model config from checkpoint keys
# ---------------------------------------------------------------------------

def _infer_config(ckpt: dict) -> dict:
    """Return embed_dim, hidden_dim, use_mmoe, num_experts, uniform_gating."""
    proj  = ckpt["client_A"]["projection.weight"]        # (embed_dim, hidden_dim)
    lstm  = ckpt["client_A"]["lstm.weight_ih_l0"]        # (4*hidden_dim, input_dim)
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


def _uniform_gating_from_name(model_name: str) -> bool:
    return model_name == "abl_uniform_gating"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_vfl(clients, server, loaders) -> dict[str, float]:
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
        preds["ihm"].append(out["ihm"].squeeze(-1).cpu().numpy())
        preds["decomp"].append(out["decomp"].squeeze(-1).cpu().numpy())
        preds["pheno"].append(out["pheno"].cpu().numpy())
        labels["ihm"].append(yI.numpy())
        labels["decomp"].append(yD.numpy())
        labels["pheno"].append(yP.numpy())

    p_ihm = np.concatenate(preds["ihm"]);   y_ihm = np.concatenate(labels["ihm"])
    p_dec = np.concatenate(preds["decomp"]); y_dec = np.concatenate(labels["decomp"])
    p_phn = np.concatenate(preds["pheno"]);  y_phn = np.concatenate(labels["pheno"])

    return {
        **{f"ihm_{k}":    v for k, v in ihm_metrics(y_ihm, p_ihm).items()},
        **{f"decomp_{k}": v for k, v in decomp_metrics(y_dec, p_dec).items()},
        **{f"pheno_{k}":  v for k, v in pheno_metrics(y_phn, p_phn).items()},
    }


def eval_ablations(ckpt_dir: Path, loaders: dict, device: str = "cpu") -> list[dict]:
    rows = []
    dev  = torch.device(device)

    for model_name in ABLATION_MODELS:
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name} not found")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            cfg  = _infer_config(ckpt)
            uniform_gating = _uniform_gating_from_name(model_name)

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
                uniform_gating=uniform_gating,
                device=dev,
            )
            clients["A"].encoder.load_state_dict(ckpt["client_A"])
            clients["B"].encoder.load_state_dict(ckpt["client_B"])
            clients["C"].encoder.load_state_dict(ckpt["client_C"])
            server.model.load_state_dict(ckpt["server"])

            m = _eval_vfl(clients, server, loaders)
            rows.append({"model": model_name, "seed": seed, **m})
            print(f"  {model_name} seed={seed}: "
                  f"IHM={m['ihm_auc_roc']:.4f} "
                  f"Decomp={m['decomp_auc_roc']:.4f} "
                  f"Pheno={m['pheno_macro_auc']:.4f}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",          default=".")
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output",        default="results/test_ablations.csv")
    p.add_argument("--use_synthetic", action="store_true")
    args = p.parse_args()

    ckpt_dir = Path(args.root) / args.ckpt_dir

    print("=" * 60)
    print("ABLATION TEST-SET EVALUATION")
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

    all_rows = eval_ablations(ckpt_dir, loaders, args.device)

    if not all_rows:
        print("[WARNING] No checkpoints found.")
        return

    fields = ["model", "seed", "ihm_auc_roc", "ihm_auc_pr",
              "decomp_auc_roc", "decomp_auc_pr",
              "pheno_macro_auc", "pheno_micro_auc"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nTest ablation results → {args.output}")


if __name__ == "__main__":
    main()
