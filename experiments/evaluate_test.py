"""
experiments/evaluate_test.py — Final test-set evaluation.

Run ONCE after all design decisions are final. Loads the best checkpoint
per seed for each model and evaluates on the held-out test split.
Never call this during development or hyperparameter tuning.

Models evaluated:
  VFL-MTL             : checkpoints/best_seed{N}.pt
  ST-IHM/Decomp/Pheno : checkpoints/best_ST-{name}_seed{N}.pt  (saved by run_exp1.py)
  local_A/B/C         : checkpoints/best_local_{site}_seed{N}.pt
  centralized_oracle  : checkpoints/best_centralized_seed{N}.pt

Output: results/test_results.csv
  columns: model, seed, ihm_auroc, ihm_auprc, decomp_auroc, decomp_auprc,
           pheno_macro_auroc, pheno_micro_auroc

Usage:
    python experiments/evaluate_test.py --root /home/asoare/vfl_mlt
    python experiments/evaluate_test.py --root . --use_synthetic   # smoke test
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
from baselines.local_only import _SITE_CFG, _LocalHead
from baselines.centralized import CentralizedEncoder, CentralizedDataset, _collate, _EMBED_DIM
from model.mmoe import MMoEServer

SEEDS = [42, 123, 7]


# ---------------------------------------------------------------------------
# Test loaders
# ---------------------------------------------------------------------------

def _real_test_loaders(root: str, batch_size: int, num_workers: int = 0,
                       max_seq_len: int = 48) -> dict:
    return build_site_loaders(Path(root), "test", batch_size, num_workers, max_seq_len)


def _synthetic_test_loaders(batch_size: int, seed: int) -> dict:
    """Minimal synthetic test loaders matching real data shapes."""
    N, T = 64, 48
    g = torch.Generator(); g.manual_seed(seed + 999)
    loaders = {
        "A": DataLoader(TensorDataset(
            torch.randn(N, T, 7), torch.ones(N, T),
            torch.randint(0, 2, (N,), generator=g).float()), batch_size=batch_size),
        "B": DataLoader(TensorDataset(
            torch.randn(N, T, 4), torch.ones(N, T),
            torch.randint(0, 2, (N,), generator=g).float()), batch_size=batch_size),
        "C": DataLoader(TensorDataset(
            torch.randn(N, T, 3), torch.ones(N, T),
            torch.randint(0, 2, (N, 25), generator=g).float()), batch_size=batch_size),
    }
    return loaders


# ---------------------------------------------------------------------------
# VFL-MTL evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_vfl(clients, server, loaders) -> dict[str, float]:
    for c in clients.values(): c.encoder.eval()
    server.model.eval()
    preds  = {"ihm": [], "decomp": [], "pheno": []}
    labels = {"ihm": [], "decomp": [], "pheno": []}
    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA; xB, mB, yD = bB; xC, mC, yP = bC
        embs = {"A": clients["A"].eval_forward(xA, mA),
                "B": clients["B"].eval_forward(xB, mB),
                "C": clients["C"].eval_forward(xC, mC)}
        out = server.predict(embs)
        preds["ihm"].append(out["ihm"].squeeze(-1).cpu().numpy())
        preds["decomp"].append(out["decomp"].squeeze(-1).cpu().numpy())
        preds["pheno"].append(out["pheno"].cpu().numpy())
        labels["ihm"].append(yI.numpy())
        labels["decomp"].append(yD.numpy())
        labels["pheno"].append(yP.numpy())

    p_ihm = np.concatenate(preds["ihm"]); y_ihm = np.concatenate(labels["ihm"])
    p_dec = np.concatenate(preds["decomp"]); y_dec = np.concatenate(labels["decomp"])
    p_phn = np.concatenate(preds["pheno"]); y_phn = np.concatenate(labels["pheno"])
    return {**{f"ihm_{k}":    v for k, v in ihm_metrics(y_ihm, p_ihm).items()},
            **{f"decomp_{k}": v for k, v in decomp_metrics(y_dec, p_dec).items()},
            **{f"pheno_{k}":  v for k, v in pheno_metrics(y_phn, p_phn).items()}}


def _infer_vfl_dims(ckpt: dict) -> tuple[int, int]:
    """Infer embed_dim and hidden_dim from client_A encoder weights."""
    proj = ckpt["client_A"]["projection.weight"]   # (embed_dim, hidden_dim)
    lstm = ckpt["client_A"]["lstm.weight_ih_l0"]    # (4*hidden_dim, input_dim)
    return int(proj.shape[0]), int(lstm.shape[0] // 4)


def eval_vfl_mtl(ckpt_dir: Path, loaders: dict,
                 embed_dim: int = 64, device: str = "cpu") -> list[dict]:
    rows = []
    for model_name in ["VFL-MTL", "ST-IHM", "ST-Decomp", "ST-Pheno"]:
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name} not found")
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            ed, hd = _infer_vfl_dims(ckpt)
            dev = torch.device(device)
            clients = {
                s: VFLClient(input_dim=d, hidden_dim=hd, embed_dim=ed,
                             lr=1e-3, device=dev)
                for s, d in [("A", 7), ("B", 4), ("C", 3)]
            }
            server = VFLServer(embed_dim=ed, device=dev)
            clients["A"].encoder.load_state_dict(ckpt["client_A"])
            clients["B"].encoder.load_state_dict(ckpt["client_B"])
            clients["C"].encoder.load_state_dict(ckpt["client_C"])
            server.model.load_state_dict(ckpt["server"])
            m = _eval_vfl(clients, server, loaders)
            rows.append({"model": model_name, "seed": seed, **m})
            print(f"  {model_name} seed={seed}: IHM={m['ihm_auc_roc']:.4f} "
                  f"Decomp={m['decomp_auc_roc']:.4f} Pheno={m['pheno_macro_auc']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Local-only evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_local(ckpt_dir: Path, loaders: dict, device: str = "cpu") -> list[dict]:
    rows = []
    dev = torch.device(device)
    for site in ["A", "B", "C"]:
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_local_{site}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name} not found")
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            cfg = _SITE_CFG[site]
            embed_dim  = ckpt.get("embed_dim",  192)
            hidden_dim = ckpt.get("hidden_dim", 128)
            encoder = SiteEncoder(cfg["input_dim"], hidden_dim, embed_dim=embed_dim).to(dev)
            head    = _LocalHead(embed_dim, cfg["task_type"]).to(dev)
            encoder.load_state_dict(ckpt["encoder"])
            head.load_state_dict(ckpt["head"])
            encoder.eval(); head.eval()

            all_p, all_y = [], []
            for x, mask, y in loaders[site]:
                emb  = encoder(x.to(dev), mask.to(dev))
                pred = head(emb).cpu()
                all_p.append(pred); all_y.append(y)
            p = torch.cat(all_p); y = torch.cat(all_y)

            if site == "A":
                m = ihm_metrics(y.numpy(), p.squeeze(-1).numpy())
                row = {"model": f"local_{site}", "seed": seed,
                       "ihm_auc_roc": m["auc_roc"], "ihm_auc_pr": m["auc_pr"],
                       "decomp_auc_roc": float("nan"), "decomp_auc_pr": float("nan"),
                       "pheno_macro_auc": float("nan"), "pheno_micro_auc": float("nan")}
            elif site == "B":
                m = decomp_metrics(y.numpy(), p.squeeze(-1).numpy())
                row = {"model": f"local_{site}", "seed": seed,
                       "ihm_auc_roc": float("nan"), "ihm_auc_pr": float("nan"),
                       "decomp_auc_roc": m["auc_roc"], "decomp_auc_pr": m["auc_pr"],
                       "pheno_macro_auc": float("nan"), "pheno_micro_auc": float("nan")}
            else:
                m = pheno_metrics(y.numpy(), p.numpy())
                row = {"model": f"local_{site}", "seed": seed,
                       "ihm_auc_roc": float("nan"), "ihm_auc_pr": float("nan"),
                       "decomp_auc_roc": float("nan"), "decomp_auc_pr": float("nan"),
                       "pheno_macro_auc": m["macro_auc"], "pheno_micro_auc": m["micro_auc"]}

            rows.append(row)
            print(f"  local_{site} seed={seed}: {m}")
    return rows


# ---------------------------------------------------------------------------
# Centralized evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_centralized(ckpt_dir: Path, root: str, batch_size: int,
                     use_synthetic: bool, device: str = "cpu") -> list[dict]:
    dev = torch.device(device)
    rows = []

    if use_synthetic:
        def _loader(seed):
            N, T = 64, 48
            g = torch.Generator(); g.manual_seed(seed + 777)
            ds = TensorDataset(
                torch.randn(N, T, 14), torch.ones(N, T),
                torch.randint(0, 2, (N,),    generator=g).float(),
                torch.randint(0, 2, (N,),    generator=g).float(),
                torch.randint(0, 2, (N, 25), generator=g).float())
            return DataLoader(ds, batch_size=batch_size)
    else:
        _ds = CentralizedDataset(root, "test")
        _shared = DataLoader(_ds, batch_size=batch_size,
                             collate_fn=_collate, shuffle=False)
        _loader = lambda seed: _shared  # noqa: E731

    for seed in SEEDS:
        ckpt_path = ckpt_dir / f"best_centralized_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path.name} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        hidden_dim = ckpt.get("hidden_dim", 128)
        encoder = CentralizedEncoder(hidden_dim=hidden_dim).to(dev)
        mmoe    = MMoEServer(input_dim=_EMBED_DIM).to(dev)
        encoder.load_state_dict(ckpt["encoder"])
        mmoe.load_state_dict(ckpt["mmoe"])
        encoder.eval(); mmoe.eval()

        ihm_p, ihm_l = [], []
        dec_p, dec_l = [], []
        phn_p, phn_l = [], []
        for x, mask, yi, yd, yp in _loader(seed):
            emb = encoder(x.to(dev), mask.to(dev))
            out = mmoe(emb)
            ihm_p.append(out["ihm"].squeeze(-1).cpu()); ihm_l.append(yi)
            dec_p.append(out["decomp"].squeeze(-1).cpu()); dec_l.append(yd)
            phn_p.append(out["pheno"].cpu()); phn_l.append(yp)

        m_ihm  = ihm_metrics(torch.cat(ihm_l).numpy(),  torch.cat(ihm_p).numpy())
        m_dec  = decomp_metrics(torch.cat(dec_l).numpy(), torch.cat(dec_p).numpy())
        m_phn  = pheno_metrics(torch.cat(phn_l).numpy(),  torch.cat(phn_p).numpy())
        row = {"model": "centralized_oracle", "seed": seed,
               "ihm_auc_roc": m_ihm["auc_roc"], "ihm_auc_pr": m_ihm["auc_pr"],
               "decomp_auc_roc": m_dec["auc_roc"], "decomp_auc_pr": m_dec["auc_pr"],
               "pheno_macro_auc": m_phn["macro_auc"], "pheno_micro_auc": m_phn["micro_auc"]}
        rows.append(row)
        print(f"  centralized seed={seed}: IHM={m_ihm['auc_roc']:.4f} "
              f"Decomp={m_dec['auc_roc']:.4f} Pheno={m_phn['macro_auc']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = ["ihm_auc_roc", "decomp_auc_roc", "pheno_macro_auc"]
    header = f"{'Model':<25}" + "".join(f"{m:>20}" for m in metrics)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for model, grp in df.groupby("model", sort=False):
        vals = []
        for m in metrics:
            col = grp[m].dropna()
            vals.append(f"{col.mean():.4f}±{col.std():.4f}" if not col.empty else "—")
        print(f"{model:<25}" + "".join(f"{v:>20}" for v in vals))
    print("─" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root",         default=".")
    p.add_argument("--ckpt_dir",     default="checkpoints")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--embed_dim",    type=int, default=64,
                   help="VFL-MTL per-site embed dim (must match training)")
    p.add_argument("--device",       default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--output",       default="results/test_results.csv")
    p.add_argument("--use_synthetic",action="store_true",
                   help="Use synthetic test data (smoke test only)")
    args = p.parse_args()

    ckpt_dir = Path(args.root) / args.ckpt_dir

    print("=" * 60)
    print("FINAL TEST-SET EVALUATION")
    print("Run this only once, after all design decisions are final.")
    print("=" * 60)

    # ── Test loaders ──────────────────────────────────────────────────────────
    if args.use_synthetic:
        site_loaders = _synthetic_test_loaders(args.batch_size, seed=42)
    else:
        site_loaders = _real_test_loaders(args.root, args.batch_size)

    all_rows: list[dict] = []

    print("\n── VFL-MTL & ST variants ──")
    all_rows.extend(eval_vfl_mtl(ckpt_dir, site_loaders, args.embed_dim, args.device))

    print("\n── Local-only ──")
    all_rows.extend(eval_local(ckpt_dir, site_loaders, args.device))

    print("\n── Centralized oracle ──")
    all_rows.extend(eval_centralized(
        ckpt_dir, args.root, args.batch_size, args.use_synthetic, args.device))

    if not all_rows:
        print("\n[WARNING] No checkpoints found — run training scripts first.")
        return

    _print_summary(all_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "ihm_auc_roc", "ihm_auc_pr",
              "decomp_auc_roc", "decomp_auc_pr",
              "pheno_macro_auc", "pheno_micro_auc"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nTest results → {args.output}")


if __name__ == "__main__":
    main()
