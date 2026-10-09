"""
experiments/calibration_predictions.py — Save per-sample predicted
probabilities for calibration curve plots.

Runs inference on the test set and writes raw (y_true, y_prob) pairs per
(model/seed/task) to a CSV.  No metrics are computed here — plot_brier.py
reads these CSVs and builds calibration curves.

Outputs:
  results/predictions_nodp.csv   (--mode nodp)
  results/predictions_dp.csv     (--mode dp)

Column schema — nodp:
  model, seed, task, y_true, y_prob

Column schema — dp:
  epsilon_level, mode, seed, task, y_true, y_prob

Usage:
    python experiments/calibration_predictions.py --mode nodp --root .
    python experiments/calibration_predictions.py --mode dp   --root .
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import build_site_loaders
from fl.client import VFLClient
from fl.server import VFLServer
from model.encoder import SiteEncoder
from baselines.local_only import _SITE_CFG, _LocalHead
from baselines.centralized import CentralizedEncoder, CentralizedDataset, _collate, _EMBED_DIM
from model.mmoe import MMoEServer

SEEDS = [42, 123, 7]

NODP_IHM_MODELS   = ["local_A", "ST-IHM",    "VFL-MTL", "centralized_oracle"]
NODP_DECOMP_MODELS= ["local_B", "ST-Decomp",  "VFL-MTL", "centralized_oracle"]

DP_EPS    = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
DP_MODES  = ["uniform", "stratified"]


# ── shared inference helpers ─────────────────────────────────────────────────

def _load_vfl_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    proj = ckpt["client_A"]["projection.weight"]
    lstm = ckpt["client_A"]["lstm.weight_ih_l0"]
    ed   = int(proj.shape[0])
    hd   = int(lstm.shape[0] // 4)
    dev  = torch.device(device)
    clients = {s: VFLClient(input_dim=d, hidden_dim=hd, embed_dim=ed, lr=1e-3, device=dev)
               for s, d in [("A", 7), ("B", 4), ("C", 3)]}
    server = VFLServer(embed_dim=ed, device=dev)
    clients["A"].encoder.load_state_dict(ckpt["client_A"])
    clients["B"].encoder.load_state_dict(ckpt["client_B"])
    clients["C"].encoder.load_state_dict(ckpt["client_C"])
    server.model.load_state_dict(ckpt["server"])
    for c in clients.values(): c.encoder.eval()
    server.model.eval()
    return clients, server


@torch.no_grad()
def _run_vfl(clients, server, loaders) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {task: (y_true, y_prob)} for IHM and Decomp."""
    pred_ihm, lbl_ihm = [], []
    pred_dec, lbl_dec = [], []
    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA; xB, mB, yD = bB; xC, mC, _ = bC
        embs = {"A": clients["A"].eval_forward(xA, mA),
                "B": clients["B"].eval_forward(xB, mB),
                "C": clients["C"].eval_forward(xC, mC)}
        out = server.predict(embs)
        pred_ihm.append(out["ihm"].squeeze(-1).cpu().numpy())
        pred_dec.append(out["decomp"].squeeze(-1).cpu().numpy())
        lbl_ihm.append(yI.numpy())
        lbl_dec.append(yD.numpy())
    return {
        "ihm":   (np.concatenate(lbl_ihm),   np.concatenate(pred_ihm)),
        "decomp":(np.concatenate(lbl_dec),   np.concatenate(pred_dec)),
    }


@torch.no_grad()
def _run_local(ckpt_path: Path, site: str, loaders, device: str
               ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = _SITE_CFG[site]
    ed   = ckpt.get("embed_dim", 192)
    hd   = ckpt.get("hidden_dim", 128)
    dev  = torch.device(device)
    enc  = SiteEncoder(cfg["input_dim"], hd, embed_dim=ed).to(dev)
    head = _LocalHead(ed, cfg["task_type"]).to(dev)
    enc.load_state_dict(ckpt["encoder"]); head.load_state_dict(ckpt["head"])
    enc.eval(); head.eval()
    all_p, all_y = [], []
    for x, mask, y in loaders[site]:
        emb = enc(x.to(dev), mask.to(dev))
        all_p.append(head(emb).cpu()); all_y.append(y)
    p = torch.cat(all_p).squeeze(-1).numpy()
    y = torch.cat(all_y).numpy()
    task = "ihm" if site == "A" else "decomp"
    return {task: (y, p)}


@torch.no_grad()
def _run_centralized(ckpt_path: Path, root: str, batch_size: int, device: str
                     ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    dev  = torch.device(device)
    hd   = ckpt.get("hidden_dim", 128)
    enc  = CentralizedEncoder(hidden_dim=hd).to(dev)
    mmoe = MMoEServer(input_dim=_EMBED_DIM).to(dev)
    enc.load_state_dict(ckpt["encoder"]); mmoe.load_state_dict(ckpt["mmoe"])
    enc.eval(); mmoe.eval()
    ds = CentralizedDataset(root, "test")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         collate_fn=_collate, shuffle=False)
    ihm_p, ihm_l, dec_p, dec_l = [], [], [], []
    for x, mask, yi, yd, _ in loader:
        emb = enc(x.to(dev), mask.to(dev))
        out = mmoe(emb)
        ihm_p.append(out["ihm"].squeeze(-1).cpu()); ihm_l.append(yi)
        dec_p.append(out["decomp"].squeeze(-1).cpu()); dec_l.append(yd)
    return {
        "ihm":    (torch.cat(ihm_l).numpy(), torch.cat(ihm_p).numpy()),
        "decomp": (torch.cat(dec_l).numpy(), torch.cat(dec_p).numpy()),
    }


# ── DP checkpoint name helper ─────────────────────────────────────────────────

def _dp_ckpt_name(eps: float, mode: str, seed: int) -> str:
    if not np.isfinite(eps):
        return f"best_VFL-MTL_seed{seed}.pt"
    # uniform checkpoints saved as eps1.0, eps2.0 ... ; stratified as eps5 (no decimal)
    eps_str = str(int(eps)) if mode == "stratified" else str(float(eps))
    return f"best_DP-{mode}-eps{eps_str}-seed{seed}_seed{seed}.pt"


# ── writers ───────────────────────────────────────────────────────────────────

def _write_rows_nodp(writer, model: str, seed: int,
                     task_preds: dict[str, tuple[np.ndarray, np.ndarray]]):
    for task, (y_true, y_prob) in task_preds.items():
        for yt, yp in zip(y_true.tolist(), y_prob.tolist()):
            writer.writerow({"model": model, "seed": seed,
                             "task": task, "y_true": yt, "y_prob": yp})


def _write_rows_dp(writer, eps: float, mode: str, seed: int,
                   task_preds: dict[str, tuple[np.ndarray, np.ndarray]]):
    for task, (y_true, y_prob) in task_preds.items():
        for yt, yp in zip(y_true.tolist(), y_prob.tolist()):
            writer.writerow({"epsilon_level": eps, "mode": mode, "seed": seed,
                             "task": task, "y_true": yt, "y_prob": yp})


# ── main modes ────────────────────────────────────────────────────────────────

def run_nodp(args):
    root     = Path(args.root)
    ckpt_dir = root / args.ckpt_dir
    loaders  = build_site_loaders(root, "test", args.batch_size, 0, 48, align_stays=args.align_stays)
    out      = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["model", "seed", "task", "y_true", "y_prob"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model_name in ["VFL-MTL", "ST-IHM", "ST-Decomp"]:
            for seed in SEEDS:
                ckpt = ckpt_dir / f"best_{model_name}_seed{seed}.pt"
                if not ckpt.exists():
                    print(f"  [SKIP] {ckpt.name}"); continue
                clients, server = _load_vfl_ckpt(ckpt, args.device)
                preds = _run_vfl(clients, server, loaders)
                # Keep only the task each model is trained on (IHM or Decomp);
                # VFL-MTL exposes both
                if model_name == "ST-IHM":
                    preds = {"ihm": preds["ihm"]}
                elif model_name == "ST-Decomp":
                    preds = {"decomp": preds["decomp"]}
                _write_rows_nodp(writer, model_name, seed, preds)
                print(f"  saved {model_name} seed={seed}")

        for site in ["A", "B"]:
            for seed in SEEDS:
                ckpt = ckpt_dir / f"best_local_{site}_seed{seed}.pt"
                if not ckpt.exists():
                    print(f"  [SKIP] {ckpt.name}"); continue
                preds = _run_local(ckpt, site, loaders, args.device)
                _write_rows_nodp(writer, f"local_{site}", seed, preds)
                print(f"  saved local_{site} seed={seed}")

        for seed in SEEDS:
            ckpt = ckpt_dir / f"best_centralized_seed{seed}.pt"
            if not ckpt.exists():
                print(f"  [SKIP] {ckpt.name}"); continue
            preds = _run_centralized(ckpt, str(root), args.batch_size, args.device)
            _write_rows_nodp(writer, "centralized_oracle", seed, preds)
            print(f"  saved centralized seed={seed}")

    print(f"\nPredictions → {out}")


def run_dp(args):
    root     = Path(args.root)
    ckpt_dir = root / args.ckpt_dir
    loaders  = build_site_loaders(root, "test", args.batch_size, 0, 48, align_stays=args.align_stays)
    out      = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["epsilon_level", "mode", "seed", "task", "y_true", "y_prob"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for eps in DP_EPS:
            for mode in DP_MODES:
                if not np.isfinite(eps) and mode == "stratified":
                    continue   # ε=∞ has no stratified variant
                for seed in SEEDS:
                    ckpt = ckpt_dir / _dp_ckpt_name(eps, mode, seed)
                    if not ckpt.exists():
                        print(f"  [SKIP] {ckpt.name}"); continue
                    clients, server = _load_vfl_ckpt(ckpt, args.device)
                    preds = _run_vfl(clients, server, loaders)
                    _write_rows_dp(writer, eps, mode, seed, preds)
                    print(f"  saved eps={eps} {mode} seed={seed}")

    print(f"\nPredictions → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["nodp", "dp"], required=True)
    p.add_argument("--root",       default=".")
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--align_stays", action="store_true",
                   help="same ICU stay at the same row across sites (use for models trained with it)")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = f"results/predictions_{args.mode}.csv"

    if args.mode == "nodp":
        run_nodp(args)
    else:
        run_dp(args)


if __name__ == "__main__":
    main()
