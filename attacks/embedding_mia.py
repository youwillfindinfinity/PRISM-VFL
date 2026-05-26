"""
attacks/embedding_mia.py — Membership inference attack on VFL cut-layer embeddings.

Threat model (honest-but-curious server):
  The server has seen training-set embeddings repeatedly during training.
  A binary classifier is trained to distinguish member (training-set)
  from non-member (validation-set) patients based on their cut-layer embeddings.

Attack (consistent with Weng et al. 2021, Luo et al. CCS 2021):
  - Members:     80% of training-set embeddings (train the attack classifier)
  - Non-members: 80% of test-set embeddings (balanced sample)
  - Hold-out 20% of each for evaluation
  - Binary logistic regression: input = z_concat (B, 3 × embed_dim)
  - Target under sufficient DP: attack AUC ≈ 0.50

Note: gradient-norm MIA (Carlini et al. 2022) not used — requires white-box
gradient access; in VFL the server observes cut-layer embeddings, not raw
parameter gradients.

Output: results/embedding_mia.csv
  columns: epsilon_level, mode, seed, attack_auc, attack_accuracy

Usage
-----
  python attacks/embedding_mia.py --use_synthetic --n_rounds 3
  python attacks/embedding_mia.py --splits_dir data/vertical_splits
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.client import VFLClient
from fl.server import VFLServer
from train import make_synthetic_loaders

# ---------------------------------------------------------------------------
# Constants matching privacy_utility_curves.py
# ---------------------------------------------------------------------------

SEEDS           = [42, 123, 7]
EPSILON_LEVELS  = ["inf", "10.0", "5.0", "2.0", "1.0", "0.5"]
EMBED_DIM       = 64
SITE_INPUT_DIMS = {"A": 7, "B": 4, "C": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: str, eps_label: str, mode: str, seed: int) -> Path:
    if str(eps_label) == "inf":
        return Path(ckpt_dir) / f"best_VFL-MTL_seed{seed}.pt"
    if mode == "stratified":
        model_name = f"DP-stratified-eps5-seed{seed}"
    else:
        model_name = f"DP-uniform-eps{eps_label}-seed{seed}"
    return Path(ckpt_dir) / f"best_{model_name}_seed{seed}.pt"


def _build_clients_and_server(device: torch.device) -> tuple[dict, VFLServer]:
    clients = {
        s: VFLClient(input_dim=SITE_INPUT_DIMS[s], embed_dim=EMBED_DIM, device=device)
        for s in ("A", "B", "C")
    }
    server = VFLServer(embed_dim=EMBED_DIM, device=device)
    return clients, server


def _load_weights(path: Path, clients: dict, server: VFLServer) -> None:
    """Load only model weights — skips optimizer state (not needed for inference)."""
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    for site in ("A", "B", "C"):
        if f"client_{site}" in ckpt:
            clients[site].encoder.load_state_dict(ckpt[f"client_{site}"])
    if "server" in ckpt:
        server.model.load_state_dict(ckpt["server"])


@torch.no_grad()
def _extract_concat_embeddings(
    clients: dict,
    loaders: dict,
    device: torch.device,
) -> np.ndarray:
    """
    Extract concatenated cut-layer embeddings [z_A | z_B | z_C] for all batches.
    Returns (N, 3 * embed_dim).
    """
    emb_parts: list[np.ndarray] = []

    for batch_A, batch_B, batch_C in zip(loaders["A"], loaders["B"], loaders["C"]):
        x_A, mask_A, _ = batch_A
        x_B, mask_B, _ = batch_B
        x_C, mask_C, _ = batch_C

        z_A = clients["A"].eval_forward(x_A, mask_A).cpu().numpy()
        z_B = clients["B"].eval_forward(x_B, mask_B).cpu().numpy()
        z_C = clients["C"].eval_forward(x_C, mask_C).cpu().numpy()

        emb_parts.append(np.concatenate([z_A, z_B, z_C], axis=-1))  # (B, 192)

    return np.concatenate(emb_parts, axis=0)  # (N, 192)


def run_mia(
    z_members: np.ndarray,
    z_nonmembers: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Train a binary MIA classifier and return attack metrics.

    Parameters
    ----------
    z_members    : (N_m, D) — training-set (member) embeddings
    z_nonmembers : (N_nm, D) — validation-set (non-member) embeddings
    test_size    : fraction held out for evaluation
    random_state : random seed for train/test split

    Returns
    -------
    {'attack_auc': float, 'attack_accuracy': float}
    Target under sufficient DP: AUC ≈ 0.50, accuracy ≈ 0.50
    """
    n_m  = len(z_members)
    n_nm = len(z_nonmembers)
    n    = min(n_m, n_nm)  # balanced attack dataset

    # Balance classes
    rng = np.random.default_rng(random_state)
    idx_m  = rng.choice(n_m,  n, replace=False)
    idx_nm = rng.choice(n_nm, n, replace=False)

    X = np.concatenate([z_members[idx_m], z_nonmembers[idx_nm]], axis=0)
    y = np.array([1] * n + [0] * n)

    if n < 2:
        return {"attack_auc": float("nan"), "attack_accuracy": float("nan")}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    try:
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, y_proba)) if len(np.unique(y_te)) > 1 else float("nan")
        acc = float(accuracy_score(y_te, y_pred))
    except Exception:
        auc, acc = float("nan"), float("nan")

    return {"attack_auc": auc, "attack_accuracy": acc}


# ---------------------------------------------------------------------------
# Per-experiment runner
# ---------------------------------------------------------------------------

def _run_one(
    eps_label: str,
    mode: str,
    seed: int,
    ckpt_dir: str,
    use_synthetic: bool,
    n_synthetic: int,
    batch_size: int,
    max_seq_len: int,
    splits_dir: str,
    device: torch.device,
    split: str = "test",
) -> dict:
    torch.manual_seed(seed)
    clients, server = _build_clients_and_server(device)

    ckpt = _ckpt_path(ckpt_dir, eps_label, mode, seed)
    if ckpt.exists():
        _load_weights(ckpt, clients, server)
    elif not use_synthetic:
        print(f"  [embedding_mia] WARNING: checkpoint not found: {ckpt}")

    if use_synthetic:
        n_tr = max(1, n_synthetic // batch_size)
        n_te = max(1, n_tr // 4)
        train_loaders = make_synthetic_loaders(batch_size, max_seq_len, n_tr)
        test_loaders  = make_synthetic_loaders(batch_size, max_seq_len, n_te)
    else:
        from data_prep.dataset import build_site_loaders
        project_root = Path(splits_dir).parents[1]
        train_loaders = build_site_loaders(project_root, "train", batch_size)
        test_loaders  = build_site_loaders(project_root, split,   batch_size)

    z_members    = _extract_concat_embeddings(clients, train_loaders, device)
    z_nonmembers = _extract_concat_embeddings(clients, test_loaders,  device)

    metrics = run_mia(z_members, z_nonmembers, random_state=seed)
    metrics.update({"epsilon_level": eps_label, "mode": mode, "seed": seed})
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    parser.add_argument("--privacy_csv",   default="results/privacy_utility_combined.csv")
    parser.add_argument("--output",        default="results/embedding_mia.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--max_seq_len",   type=int, default=48)
    parser.add_argument("--split",         default="test", choices=["val", "test"],
                        help="Which held-out split to use as non-members.")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    combos: list[tuple[str, str]] = []
    privacy_csv = Path(args.privacy_csv)
    if privacy_csv.exists():
        import pandas as pd
        df = pd.read_csv(privacy_csv)
        combos = list(df[["epsilon_level", "mode"]].drop_duplicates().itertuples(index=False, name=None))
        print(f"[embedding_mia] {len(combos)} (ε, mode) combos from {privacy_csv}")
    else:
        combos = [(eps, "uniform") for eps in EPSILON_LEVELS] + [("5.0", "stratified")]
        print(f"[embedding_mia] privacy_utility_combined.csv not found; using defaults ({len(combos)} combos)")

    all_rows: list[dict] = []

    for eps_label, mode in combos:
        for seed in SEEDS:
            print(f"  ε={eps_label} | mode={mode} | seed={seed}")
            try:
                row = _run_one(
                    eps_label=str(eps_label),
                    mode=mode,
                    seed=seed,
                    ckpt_dir=args.ckpt_dir,
                    use_synthetic=args.use_synthetic,
                    n_synthetic=args.n_synthetic,
                    batch_size=args.batch_size,
                    max_seq_len=args.max_seq_len,
                    splits_dir=args.splits_dir,
                    device=device,
                    split=args.split,
                )
                all_rows.append(row)
            except Exception as e:
                print(f"  ERROR: {e}")

    if not all_rows:
        print("[embedding_mia] No results collected.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[embedding_mia] Done. {len(all_rows)} rows → {out}")

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["mode", "epsilon_level"])[["attack_auc", "attack_accuracy"]].mean()
    print("\n── MIA Summary (mean across seeds) ──")
    print(summary.to_string())


if __name__ == "__main__":
    main()
