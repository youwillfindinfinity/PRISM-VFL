"""
baselines/local_only.py — Local-only single-task baselines (no FL, no server).

Each site trains an independent LSTM encoder + task head on its own features
with zero cross-site communication. Lower bound: measures what each site
achieves without VFL embedding exchange or multi-task learning.

Difference from run_exp1.py ST-* configs: those still route embeddings through
the VFL server (cross-site exchange present). Here there is no server at all —
the encoder output goes directly to a local task head.

Sites / tasks:
  Site A (7 vitals)    → IHM binary classification
  Site B (4 labs)      → Decompensation binary classification
  Site C (3 composite) → Phenotyping 25-label classification

Usage:
    # Smoke test (synthetic data, no MIMIC required):
    python baselines/local_only.py --site A --use_synthetic --n_epochs 3

    # Full run (Snellius):
    python baselines/local_only.py --site A \
        --root /home/asoare/vfl_mlt --n_epochs 50 \
        --seeds 42 123 7 --output results/local_only_A.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import (
    VFLSiteDataset,
    collate_fn,
    SITE_A_FEATURES,
    SITE_B_FEATURES,
    SITE_C_FEATURES,
    PHENO_LABEL_COLS,
)
from model.encoder import SiteEncoder
from experiments.metrics import ihm_metrics, decomp_metrics, pheno_metrics


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------

_SITE_CFG = {
    "A": {
        "input_dim":   len(SITE_A_FEATURES),
        "feature_cols": SITE_A_FEATURES,
        "label_col":   "y_ihm",
        "task_type":   "binary",
        "site_csv":    "site_A_vitals.csv",
        "ts_subdir":   "in-hospital-mortality",
    },
    "B": {
        "input_dim":   len(SITE_B_FEATURES),
        "feature_cols": SITE_B_FEATURES,
        "label_col":   "y_decomp",
        "task_type":   "binary",
        "site_csv":    "site_B_labs.csv",
        "ts_subdir":   "decompensation",
    },
    "C": {
        "input_dim":   len(SITE_C_FEATURES),
        "feature_cols": SITE_C_FEATURES,
        "label_col":   PHENO_LABEL_COLS,
        "task_type":   "multilabel",
        "site_csv":    "site_C_composite.csv",
        "ts_subdir":   "phenotyping",
    },
}


# ---------------------------------------------------------------------------
# Local task heads
# ---------------------------------------------------------------------------

class _LocalHead(nn.Module):
    def __init__(self, embed_dim: int, task_type: str):
        super().__init__()
        if task_type == "binary":
            self.head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
        elif task_type == "los_bins":
            self.head = nn.Linear(embed_dim, 10)      # logits
        else:                                           # multilabel
            self.head = nn.Sequential(nn.Linear(embed_dim, 25), nn.Sigmoid())

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _loss_fn(task_type: str, pos_weight: float = 1.0):
    if task_type == "binary" and pos_weight != 1.0:
        # pos_weight upweights positive samples: weight = pos_weight * y + (1 - y)
        # Equivalent to BCEWithLogitsLoss(pos_weight=...) but works on Sigmoid outputs.
        pw = pos_weight
        return lambda pred, target: nn.functional.binary_cross_entropy(
            pred, target, weight=pw * target + (1.0 - target)
        )
    return {
        "binary":     nn.BCELoss(),
        "multilabel": nn.BCELoss(),
    }[task_type]


# ---------------------------------------------------------------------------
# Synthetic loaders (smoke-test, no MIMIC required)
# ---------------------------------------------------------------------------

def _synthetic_loaders(site: str, batch_size: int, seed: int) -> dict[str, DataLoader]:
    cfg = _SITE_CFG[site]
    n_train, n_val = 256, 64
    rng = torch.Generator(); rng.manual_seed(seed)

    def _make(n):
        x    = torch.randn(n, 48, cfg["input_dim"])
        mask = torch.ones(n, 48)
        if cfg["task_type"] == "binary":
            y = torch.randint(0, 2, (n,), generator=rng).float()
        elif cfg["task_type"] == "multilabel":
            y = torch.randint(0, 2, (n, 25), generator=rng).float()
        from torch.utils.data import TensorDataset
        return DataLoader(TensorDataset(x, mask, y), batch_size=batch_size)

    return {"train": _make(n_train), "val": _make(n_val)}


# ---------------------------------------------------------------------------
# Real data loaders
# ---------------------------------------------------------------------------

def _real_loaders(site: str, root: str, batch_size: int,
                  num_workers: int) -> dict[str, DataLoader]:
    cfg        = _SITE_CFG[site]
    root_p     = Path(root)
    splits_dir = root_p / "data" / "vertical_splits"
    bench_dir  = root_p / "data" / "mimic3-benchmarks" / "data"
    aligned    = splits_dir / "aligned_patient_ids.csv"
    ts_root    = bench_dir / cfg["ts_subdir"]

    loaders = {}
    for split in ("train", "val"):
        ds = VFLSiteDataset(
            site_csv        = splits_dir / cfg["site_csv"],
            feature_cols    = cfg["feature_cols"],
            label_col       = cfg["label_col"],
            split           = split,
            aligned_ids_csv = aligned,
            timeseries_root = ts_root,
            task_type       = cfg["task_type"],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            collate_fn  = collate_fn,
            num_workers = num_workers,
            pin_memory  = True,
        )
    return loaders


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_local(
    site:             str,
    root:             str,
    n_epochs:         int,
    lr:               float,
    batch_size:       int,
    seed:             int,
    use_synthetic:    bool,
    num_workers:      int = 0,
    embed_dim:        int = 192,
    hidden_dim:       int = 128,
    prebuilt_loaders: dict = None,
    patience:         int = 10,
    ckpt_dir:         str | None = None,
    decomp_pos_weight: float = 0.0,  # 0.0 = auto-compute from CSV (Site B only)
) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg       = _SITE_CFG[site]
    task_type = cfg["task_type"]

    # Determine pos_weight for Site B decompensation
    pw = 1.0
    if site == "B" and not use_synthetic:
        if decomp_pos_weight > 0.0:
            pw = decomp_pos_weight
        else:
            splits_dir = Path(root) / "data" / "vertical_splits"
            _b = pd.read_csv(splits_dir / "site_B_labs.csv", usecols=["y_decomp", "split"])
            pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
            pw = (1.0 - pos_rate) / pos_rate
            print(f"[local_B] decomp pos_weight={pw:.1f}  (pos_rate={pos_rate:.3%})")

    encoder = SiteEncoder(
        input_dim  = cfg["input_dim"],
        hidden_dim = hidden_dim,
        embed_dim  = embed_dim,   # 192 — matches VFL-MTL concat dimension
    ).to(device)

    head    = _LocalHead(embed_dim, task_type).to(device)
    loss_fn = _loss_fn(task_type, pos_weight=pw)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    opt     = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr
    )

    if prebuilt_loaders is not None:
        loaders = prebuilt_loaders
    elif use_synthetic:
        loaders = _synthetic_loaders(site, batch_size, seed)
    else:
        loaders = _real_loaders(site, root, batch_size, num_workers)

    # early stopping state
    primary = "auc_roc" if task_type == "binary" else "macro_auc"
    best_score, no_improve, best_state = -1.0, 0, None

    rows = []
    for epoch in range(1, n_epochs + 1):
        encoder.train(); head.train()
        t0 = time.perf_counter()
        total_loss, n_batches = 0.0, 0

        for x, mask, y in loaders["train"]:
            x    = x.to(device)
            mask = mask.to(device)
            y    = y.to(device)

            emb  = encoder(x, mask)
            pred = head(emb)

            if task_type == "binary":
                loss = loss_fn(pred.squeeze(-1), y)
            else:
                loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1

        # ── validation ──────────────────────────────────────────────────
        encoder.eval(); head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, mask, y in loaders["val"]:
                emb  = encoder(x.to(device), mask.to(device))
                pred = head(emb).cpu()
                all_preds.append(pred)
                all_labels.append(y)

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        if task_type == "binary" and site == "A":
            metrics = ihm_metrics(labels.numpy(), preds.squeeze(-1).numpy())
        elif task_type == "binary" and site == "B":
            metrics = decomp_metrics(labels.numpy(), preds.squeeze(-1).numpy())
        else:
            metrics = pheno_metrics(labels.numpy(), preds.numpy())

        score = metrics[primary]
        if score > best_score:
            best_score = score
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in
                          {**dict(encoder.named_parameters()),
                           **dict(head.named_parameters())}.items()}
            if ckpt_dir is not None and not use_synthetic:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"encoder": encoder.state_dict(),
                     "head":    head.state_dict(),
                     "site": site, "seed": seed,
                     "embed_dim": embed_dim, "hidden_dim": hidden_dim,
                     "best_val_score": float(best_score)},
                    Path(ckpt_dir) / f"best_local_{site}_seed{seed}.pt",
                )
        else:
            no_improve += 1

        rows.append({
            "model":      f"local_{site}",
            "site":       site,
            "task":       task_type,
            "epoch":      epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "elapsed_s":  time.perf_counter() - t0,
            "seed":       seed,
            **{f"val_{k}": v for k, v in metrics.items()},
        })

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (best {primary}={best_score:.4f})")
            break

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",          required=True, choices=["A", "B", "C"])
    parser.add_argument("--root",          default=".")
    parser.add_argument("--n_epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--num_workers",   type=int,   default=0)
    parser.add_argument("--output",        default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    args = parser.parse_args()

    out = args.output or f"results/local_only_{args.site}.csv"
    # Build real dataset once — preload is expensive, reuse across seeds
    prebuilt = (None if args.use_synthetic
                else _real_loaders(args.site, args.root, args.batch_size, args.num_workers))

    all_rows = []
    for seed in args.seeds:
        rows = train_local(
            site             = args.site,
            root             = args.root,
            n_epochs         = args.n_epochs,
            lr               = args.lr,
            batch_size       = args.batch_size,
            seed             = seed,
            use_synthetic    = args.use_synthetic,
            num_workers      = args.num_workers,
            prebuilt_loaders = prebuilt,
            patience         = args.patience,
            ckpt_dir         = args.ckpt_dir,
        )
        all_rows.extend(rows)
        print(f"site={args.site} seed={seed}: last val metrics = "
              + str({k: round(v, 4) for k, v in rows[-1].items()
                     if k.startswith("val_")}))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
