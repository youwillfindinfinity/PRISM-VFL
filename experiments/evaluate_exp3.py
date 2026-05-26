"""
experiments/evaluate_exp3.py — Test-set evaluation for Exp 3 (scalability).

Loads best checkpoint for each n_sites configuration and evaluates on the
held-out test split. n_sites is inferred from the checkpoint keys:
  n_sites=2 → checkpoint has client_A, client_B only; server input_dim = 2*embed_dim
  n_sites=3 → all three clients present; server input_dim = 3*embed_dim

Configurations: n_sites ∈ {2, 3}
Seeds: [42, 123, 7]

Output: results/test_exp3.csv
  columns: n_sites, seed, ihm_auc_roc, ihm_auc_pr,
           decomp_auc_roc, decomp_auc_pr, pheno_macro_auc, pheno_micro_auc
  (pheno metrics are NaN for n_sites=2 — pheno head was not trained)

Usage:
    python experiments/evaluate_exp3.py --root /home/asoare/vfl_mlt
    python experiments/evaluate_exp3.py --root . --use_synthetic
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

SEEDS   = [42, 123, 7]
N_SITES = [2, 3]


def _infer_config(ckpt: dict) -> dict:
    """Infer embed_dim, hidden_dim, use_mmoe, num_experts, n_sites from checkpoint."""
    proj      = ckpt["client_A"]["projection.weight"]   # (embed_dim, hidden_dim)
    lstm      = ckpt["client_A"]["lstm.weight_ih_l0"]   # (4*hidden_dim, input_dim)
    embed_dim  = int(proj.shape[0])
    hidden_dim = int(lstm.shape[0] // 4)

    n_sites = 3 if "client_C" in ckpt else 2

    srv = ckpt["server"]
    use_mmoe = "shared_bottom.0.weight" not in srv
    n_experts = sum(
        1 for k in srv
        if k.startswith("mmoe.experts.") and k.endswith(".net.0.weight")
    ) if use_mmoe else 0

    return dict(embed_dim=embed_dim, hidden_dim=hidden_dim,
                use_mmoe=use_mmoe, num_experts=n_experts, n_sites=n_sites)


@torch.no_grad()
def _eval_2site(clients: dict, server: VFLServer, loaders: dict) -> dict[str, float]:
    """Evaluate IHM + Decomp only (n_sites=2, no Pheno head trained)."""
    for c in clients.values():
        c.encoder.eval()
    server.model.eval()

    preds  = {"ihm": [], "decomp": []}
    labels = {"ihm": [], "decomp": []}

    for bA, bB in zip(loaders["A"], loaders["B"]):
        xA, mA, yI = bA
        xB, mB, yD = bB
        embs = {
            "A": clients["A"].eval_forward(xA, mA),
            "B": clients["B"].eval_forward(xB, mB),
        }
        out = server.predict(embs)
        preds["ihm"].append(out["ihm"].squeeze(-1).cpu().numpy())
        preds["decomp"].append(out["decomp"].squeeze(-1).cpu().numpy())
        labels["ihm"].append(yI.numpy())
        labels["decomp"].append(yD.numpy())

    p_ihm = np.concatenate(preds["ihm"]);    y_ihm = np.concatenate(labels["ihm"])
    p_dec = np.concatenate(preds["decomp"]); y_dec = np.concatenate(labels["decomp"])

    return {
        **{f"ihm_{k}":    v for k, v in ihm_metrics(y_ihm, p_ihm).items()},
        **{f"decomp_{k}": v for k, v in decomp_metrics(y_dec, p_dec).items()},
        "pheno_macro_auc": float("nan"),
        "pheno_micro_auc": float("nan"),
        "pheno_auc_pr":    float("nan"),
    }


@torch.no_grad()
def _eval_3site(clients: dict, server: VFLServer, loaders: dict) -> dict[str, float]:
    """Evaluate all three tasks (n_sites=3)."""
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

    p_ihm = np.concatenate(preds["ihm"]);    y_ihm = np.concatenate(labels["ihm"])
    p_dec = np.concatenate(preds["decomp"]); y_dec = np.concatenate(labels["decomp"])
    p_phn = np.concatenate(preds["pheno"]);  y_phn = np.concatenate(labels["pheno"])

    return {
        **{f"ihm_{k}":    v for k, v in ihm_metrics(y_ihm, p_ihm).items()},
        **{f"decomp_{k}": v for k, v in decomp_metrics(y_dec, p_dec).items()},
        **{f"pheno_{k}":  v for k, v in pheno_metrics(y_phn, p_phn).items()},
    }


def eval_exp3(ckpt_dir: Path, loaders: dict, device: str = "cpu") -> list[dict]:
    rows = []
    dev  = torch.device(device)

    for n_sites in N_SITES:
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_exp3_sites{n_sites}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name} not found")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            cfg  = _infer_config(ckpt)
            assert cfg["n_sites"] == n_sites, (
                f"Checkpoint n_sites mismatch: expected {n_sites}, got {cfg['n_sites']}"
            )

            site_dims = [("A", 7), ("B", 4), ("C", 3)][:n_sites]
            clients = {
                s: VFLClient(input_dim=d,
                             hidden_dim=cfg["hidden_dim"],
                             embed_dim=cfg["embed_dim"],
                             lr=1e-3, device=dev)
                for s, d in site_dims
            }
            server = VFLServer(
                embed_dim=cfg["embed_dim"],
                use_mmoe=cfg["use_mmoe"],
                num_experts=cfg["num_experts"] if cfg["use_mmoe"] else 4,
                n_sites=n_sites,
                device=dev,
            )
            for s, _ in site_dims:
                clients[s].encoder.load_state_dict(ckpt[f"client_{s}"])
            server.model.load_state_dict(ckpt["server"])

            site_loaders = {s: loaders[s] for s, _ in site_dims}
            m = _eval_2site(clients, server, site_loaders) if n_sites == 2 \
                else _eval_3site(clients, server, site_loaders)

            rows.append({"n_sites": n_sites, "seed": seed, **m})
            print(f"  n_sites={n_sites} seed={seed}: "
                  f"IHM={m['ihm_auc_roc']:.4f} "
                  f"Decomp={m['decomp_auc_roc']:.4f} "
                  + (f"Pheno={m['pheno_macro_auc']:.4f}" if n_sites == 3 else "Pheno=N/A"))

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",          default=".")
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output",        default="results/test_exp3.csv")
    p.add_argument("--use_synthetic", action="store_true")
    args = p.parse_args()

    ckpt_dir = Path(args.root) / args.ckpt_dir

    print("=" * 60)
    print("EXP 4 TEST-SET EVALUATION")
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

    all_rows = eval_exp3(ckpt_dir, loaders, args.device)

    if not all_rows:
        print("[WARNING] No checkpoints found.")
        return

    fields = ["n_sites", "seed", "ihm_auc_roc", "ihm_auc_pr",
              "decomp_auc_roc", "decomp_auc_pr",
              "pheno_macro_auc", "pheno_micro_auc"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nExp 4 test results -> {args.output}")


if __name__ == "__main__":
    main()
