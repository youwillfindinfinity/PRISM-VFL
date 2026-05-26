"""
attacks/feature_reconstruction.py — Feature reconstruction attack on Site A cut-layer embeddings.

Threat model (passive party adversary):
  A passive party (e.g. Site B or the honest-but-curious server) observes Site A's
  64-dimensional cut-layer embedding vectors z_A. It trains an MLP decoder to
  reconstruct Site A's 7 raw input features from those embeddings alone.

Attack:
  - Decoder: MLP(64 → 128 → 7) trained on train-set (embedding, raw_feature) pairs
  - Evaluated on test-set embeddings
  - Compared against a mean-baseline (predict feature mean on every sample)

Metrics per feature:
  - MSE (reconstruction mean squared error)
  - Baseline MSE (predict mean of each feature)
  - R² score (coefficient of determination; R² > 0 means attack beats mean baseline)

For each epsilon_level:
  1. Load the Site A encoder checkpoint
  2. Extract train + test embeddings via VFLClient.eval_forward()
  3. Train MLP decoder on train embeddings → raw features (50 epochs, Adam lr=1e-3)
  4. Evaluate decoder on test embeddings; record per-feature metrics

Output: results/feature_reconstruction.csv
  columns: epsilon_level, seed, feature_name, mse, baseline_mse, r2_score

Usage
-----
  python attacks/feature_reconstruction.py --splits_dir data/vertical_splits
  python attacks/feature_reconstruction.py --ckpt_dir checkpoints --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.client import VFLClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITE_A_FEATURES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "spo2",
    "resp_rate",
    "gcs_total",
]

EMBED_DIM    = 64
SITE_A_INPUT = 7

DEFAULT_EPSILON_LEVELS = ["inf", "10", "5", "2", "1", "0.5"]


# ---------------------------------------------------------------------------
# MLP decoder
# ---------------------------------------------------------------------------

class FeatureDecoder(nn.Module):
    """
    Simple MLP: 64 → 128 → 7 (Site A features).
    """

    def __init__(self, embed_dim: int = 64, n_features: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Checkpoint helpers  (mirrors attacks/label_inference.py)
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: str, eps_label: str, seed: int) -> Path:
    """
    Resolve checkpoint path for a given epsilon level and seed.

    Rules (consistent with label_inference.py):
      - eps="inf"    → best_VFL-MTL_seed{seed}.pt
      - eps=other    → best_DP-uniform-eps{eps}-seed{seed}.pt
    """
    base = Path(ckpt_dir)
    if str(eps_label) == "inf":
        return base / f"best_VFL-MTL_seed{seed}.pt"
    return base / f"best_DP-uniform-eps{eps_label}-seed{seed}.pt"


def _find_ckpt(ckpt_dir: str, eps_label: str, seed: int) -> Path | None:
    """
    Try the canonical path first; fall back to any file matching *seed{seed}*.pt.
    Returns None if nothing is found.
    """
    preferred = _ckpt_path(ckpt_dir, eps_label, seed)
    if preferred.exists():
        return preferred

    # Glob fallback: any checkpoint that contains seed{seed} in its name
    candidates = sorted(Path(ckpt_dir).glob(f"*seed{seed}*.pt"))
    if candidates:
        # Prefer a file that also has the eps label in the name
        eps_str = str(eps_label).replace(".", "")
        for c in candidates:
            if eps_str in c.name:
                return c
        # Otherwise return the most recently modified one
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def _load_site_a_encoder(ckpt_path: Path, device: torch.device) -> VFLClient:
    """Build a VFLClient for Site A and load weights from checkpoint."""
    client = VFLClient(input_dim=SITE_A_INPUT, embed_dim=EMBED_DIM, device=device)
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if "client_A" in ckpt:
        client.encoder.load_state_dict(ckpt["client_A"])
    else:
        print(f"  [feat_recon] WARNING: 'client_A' key missing in {ckpt_path.name}")
    return client


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_embeddings_and_features(
    client: VFLClient,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Site A encoder in eval mode over all batches.

    Parameters
    ----------
    loader : DataLoader yielding (x, mask, y) where x is (B, T, 7)

    Returns
    -------
    embeddings : (N, 64)  float32
    features   : (N, 7)   float32  — mean of non-zero timesteps per feature
    """
    emb_list  = []
    feat_list = []

    for x, mask, _ in loader:
        # x: (B, T, 7); mask: (B, T)
        z = client.eval_forward(x.to(device), mask.to(device)).cpu().numpy()
        emb_list.append(z)

        # Aggregate raw features: mean over valid (non-padded) timesteps
        # mask: (B, T) → (B, T, 1) broadcast
        mask_np = mask.numpy()[:, :, None]  # (B, T, 1)
        x_np    = x.numpy()                 # (B, T, 7)
        lengths = mask_np.sum(axis=1).clip(min=1)  # (B, 1)
        feat_mean = (x_np * mask_np).sum(axis=1) / lengths  # (B, 7)
        feat_list.append(feat_mean.astype(np.float32))

    return np.concatenate(emb_list, axis=0), np.concatenate(feat_list, axis=0)


# ---------------------------------------------------------------------------
# Decoder training
# ---------------------------------------------------------------------------

def _train_decoder(
    z_train: np.ndarray,
    x_train: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> FeatureDecoder:
    """Train MLP decoder z → x on training-set pairs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    decoder = FeatureDecoder(embed_dim=EMBED_DIM, n_features=SITE_A_INPUT).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    z_t = torch.from_numpy(z_train)
    x_t = torch.from_numpy(x_train)
    ds  = TensorDataset(z_t, x_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    decoder.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for z_batch, x_batch in dl:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            pred = decoder(z_batch)
            loss = criterion(pred, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            n_batches = max(1, len(dl))
            print(f"    [decoder] epoch {epoch + 1}/{n_epochs} "
                  f"train_mse={epoch_loss / n_batches:.4f}")

    return decoder


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_decoder(
    decoder: FeatureDecoder,
    z_test: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
) -> list[dict]:
    """
    Compute per-feature MSE, baseline MSE, and R² on the test set.

    Returns
    -------
    List of dicts — one per feature: {feature_name, mse, baseline_mse, r2_score}
    """
    decoder.eval()
    z_t   = torch.from_numpy(z_test).to(device)
    preds = decoder(z_t).cpu().numpy()  # (N, 7)

    rows = []
    for i, feat_name in enumerate(SITE_A_FEATURES):
        y_true  = x_test[:, i]
        y_pred  = preds[:, i]

        mse      = float(np.mean((y_true - y_pred) ** 2))
        baseline = float(np.mean((y_true - y_true.mean()) ** 2))  # predict mean

        ss_res   = np.sum((y_true - y_pred) ** 2)
        ss_tot   = np.sum((y_true - y_true.mean()) ** 2)
        r2       = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

        rows.append({
            "feature_name": feat_name,
            "mse":           mse,
            "baseline_mse":  baseline,
            "r2_score":      r2,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-epsilon runner
# ---------------------------------------------------------------------------

def _run_one_epsilon(
    eps_label: str,
    seed: int,
    ckpt_dir: str,
    splits_dir: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[dict] | None:
    """
    Full pipeline for one (epsilon, seed) pair.
    Returns list of per-feature result dicts, or None if checkpoint is missing.
    """
    print(f"\n[feat_recon] ε={eps_label} | seed={seed}")

    # ---- Load checkpoint ----
    ckpt = _find_ckpt(ckpt_dir, eps_label, seed)
    if ckpt is None:
        print(f"  WARNING: no checkpoint found for ε={eps_label} seed={seed} — skipping.")
        return None
    print(f"  Loading: {ckpt.name}")
    client = _load_site_a_encoder(ckpt, device)

    # ---- Load data ----
    try:
        from data_prep.dataset import build_site_loaders
        project_root = Path(splits_dir).parents[1]  # …/fl-thesis from data/vertical_splits
        train_loaders = build_site_loaders(project_root, "train", batch_size=batch_size)
        test_loaders  = build_site_loaders(project_root, "test",  batch_size=batch_size)
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    loader_A_train = train_loaders["A"]
    loader_A_test  = test_loaders["A"]

    # ---- Extract embeddings + aggregated features ----
    print("  Extracting train embeddings …")
    z_train, x_train = _extract_embeddings_and_features(client, loader_A_train, device)
    print(f"  Train: z={z_train.shape}, x={x_train.shape}")

    print("  Extracting test embeddings …")
    z_test, x_test = _extract_embeddings_and_features(client, loader_A_test, device)
    print(f"  Test:  z={z_test.shape}, x={x_test.shape}")

    # ---- Train decoder ----
    print(f"  Training MLP decoder ({n_epochs} epochs) …")
    decoder = _train_decoder(
        z_train, x_train,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        device=device, seed=seed,
    )

    # ---- Evaluate ----
    feat_rows = _evaluate_decoder(decoder, z_test, x_test, device)

    # Attach metadata
    for r in feat_rows:
        r["epsilon_level"] = eps_label
        r["seed"]          = seed

    return feat_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature reconstruction attack: Site A embedding → raw features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints",
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--splits_dir",
        default="data/vertical_splits",
        help="Directory containing site_A_vitals.csv etc.",
    )
    parser.add_argument(
        "--epsilon_levels",
        nargs="+",
        default=DEFAULT_EPSILON_LEVELS,
        help="Epsilon levels to evaluate (use 'inf' for no-DP baseline).",
    )
    parser.add_argument(
        "--output",
        default="results/feature_reconstruction.csv",
        help="Path for output CSV.",
    )
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--n_epochs",   type=int,   default=50,
                        help="Training epochs for the MLP decoder.")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Adam learning rate for the decoder.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[feat_recon] device={device} | seed={args.seed}")
    print(f"[feat_recon] epsilon_levels={args.epsilon_levels}")

    all_rows: list[dict] = []

    for eps in args.epsilon_levels:
        rows = _run_one_epsilon(
            eps_label  = str(eps),
            seed       = args.seed,
            ckpt_dir   = args.ckpt_dir,
            splits_dir = args.splits_dir,
            n_epochs   = args.n_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            device     = device,
        )
        if rows is not None:
            all_rows.extend(rows)

    if not all_rows:
        print("[feat_recon] No results collected — all checkpoints missing?")
        return

    # ---- Write CSV ----
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epsilon_level", "seed", "feature_name", "mse", "baseline_mse", "r2_score"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[feat_recon] Done. {len(all_rows)} rows → {out}")

    # ---- Summary table ----
    print("\n=== Feature Reconstruction Attack Summary ===")
    print(f"{'epsilon':>8}  {'feature':>18}  {'mse':>10}  {'baseline_mse':>14}  {'r2':>7}")
    print("-" * 65)
    for r in all_rows:
        print(
            f"{str(r['epsilon_level']):>8}  "
            f"{r['feature_name']:>18}  "
            f"{r['mse']:>10.4f}  "
            f"{r['baseline_mse']:>14.4f}  "
            f"{r['r2_score']:>7.4f}"
        )


if __name__ == "__main__":
    main()
