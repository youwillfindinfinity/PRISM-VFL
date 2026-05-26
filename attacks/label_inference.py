"""
attacks/label_inference.py — Label inference attack on VFL cut-layer embeddings.

Threat model (honest-but-curious server):
  The server observes all cut-layer embeddings z_A, z_B, z_C.
  It trains a linear probe to infer each active party's task labels
  from those embeddings alone (without access to raw features).

Attack:
  - Probe 1: z_A (Site A encoder output) → y_ihm  (binary, Site A label)
  - Probe 2: z_C (Site C encoder output) → y_pheno (25 phenotype labels, Site C)
  Train probe on training-set embeddings; evaluate on validation-set embeddings.

Metrics:
  - IHM: accuracy + AUC-ROC  (target under sufficient DP: accuracy ≈ prevalence ~10%)
  - Pheno: macro-AUC across 25 labels (target: ≈ 0.50)

For each (epsilon_level, mode, seed) in results/privacy_utility_combined.csv:
  1. Reconstruct model from TrainConfig defaults
  2. Load checkpoint from checkpoints/ (falls back to fresh model if absent)
  3. Extract train + test embeddings via VFLClient.eval_forward()
  4. Fit LogisticRegression probe; report metrics on test set

Output: results/label_inference.csv
  columns: epsilon_level, mode, seed, task, accuracy, auroc

Usage
-----
  python attacks/label_inference.py --use_synthetic --n_rounds 3
  python attacks/label_inference.py --splits_dir data/vertical_splits
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.client import VFLClient
from fl.server import VFLServer
from train import make_synthetic_loaders

# ---------------------------------------------------------------------------
# Constants matching privacy_utility_curves.py
# ---------------------------------------------------------------------------

SEEDS         = [42, 123, 7]
EPSILON_LEVELS = ["inf", "10.0", "5.0", "2.0", "1.0", "0.5"]
EMBED_DIM     = 64
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
    """Load only model weights from checkpoint — skips optimizer state (not needed for inference)."""
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    for site in ("A", "B", "C"):
        if f"client_{site}" in ckpt:
            clients[site].encoder.load_state_dict(ckpt[f"client_{site}"])
    if "server" in ckpt:
        server.model.load_state_dict(ckpt["server"])


@torch.no_grad()
def _extract_embeddings(
    clients: dict,
    loaders: dict,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Run eval_forward over all batches in loaders.

    Returns
    -------
    embeddings : {'A': (N, 64), 'B': (N, 64), 'C': (N, 64)}
    labels     : {'ihm': (N,), 'pheno': (N, 25)}
    """
    emb_lists:   dict[str, list] = {"A": [], "B": [], "C": []}
    label_lists: dict[str, list] = {"ihm": [], "pheno": []}

    for batch_A, batch_B, batch_C in zip(loaders["A"], loaders["B"], loaders["C"]):
        x_A, mask_A, y_ihm   = batch_A
        x_B, mask_B, _       = batch_B
        x_C, mask_C, y_pheno = batch_C

        emb_lists["A"].append(clients["A"].eval_forward(x_A, mask_A).cpu().numpy())
        emb_lists["B"].append(clients["B"].eval_forward(x_B, mask_B).cpu().numpy())
        emb_lists["C"].append(clients["C"].eval_forward(x_C, mask_C).cpu().numpy())
        label_lists["ihm"].append(y_ihm.numpy())
        label_lists["pheno"].append(y_pheno.numpy())

    embeddings = {s: np.concatenate(emb_lists[s])   for s in ("A", "B", "C")}
    labels     = {t: np.concatenate(label_lists[t]) for t in ("ihm", "pheno")}
    return embeddings, labels


def run_label_inference(
    z_train: dict[str, np.ndarray],
    y_train: dict[str, np.ndarray],
    z_val:   dict[str, np.ndarray],
    y_val:   dict[str, np.ndarray],
) -> list[dict]:
    """
    Train LR probes and return per-task attack metrics.

    Parameters
    ----------
    z_train / z_val : {'A': (N, 64), ...}
    y_train / y_val : {'ihm': (N,), 'pheno': (N, 25)}

    Returns
    -------
    List of dicts, one per attacked task: {task, accuracy, auroc}
    """
    results = []

    # ---- IHM: binary probe on z_A ----
    z_A_tr, y_ihm_tr = z_train["A"], y_train["ihm"]
    z_A_val, y_ihm_val = z_val["A"], y_val["ihm"]

    clf_ihm = LogisticRegression(max_iter=500, solver="lbfgs")
    try:
        clf_ihm.fit(z_A_tr, y_ihm_tr)
        ihm_preds  = clf_ihm.predict(z_A_val)
        ihm_proba  = clf_ihm.predict_proba(z_A_val)[:, 1]
        ihm_acc    = float((ihm_preds == y_ihm_val).mean())
        ihm_auroc  = float(roc_auc_score(y_ihm_val, ihm_proba)) \
            if len(np.unique(y_ihm_val)) > 1 else float("nan")
    except Exception:
        ihm_acc, ihm_auroc = float("nan"), float("nan")

    results.append({"task": "ihm", "accuracy": ihm_acc, "auroc": ihm_auroc})

    # ---- Pheno: one-vs-rest probe on z_C ----
    z_C_tr, y_pheno_tr = z_train["C"], y_train["pheno"]
    z_C_val, y_pheno_val = z_val["C"], y_val["pheno"]

    clf_pheno = OneVsRestClassifier(LogisticRegression(max_iter=500, solver="lbfgs"))
    pheno_aucs = []
    try:
        clf_pheno.fit(z_C_tr, y_pheno_tr)
        pheno_proba = clf_pheno.predict_proba(z_C_val)   # (N, 25)
        pheno_preds = clf_pheno.predict(z_C_val)         # (N, 25)
        pheno_acc   = float((pheno_preds == y_pheno_val).all(axis=1).mean())
        for i in range(y_pheno_val.shape[1]):
            if y_pheno_val[:, i].sum() > 0:
                pheno_aucs.append(roc_auc_score(y_pheno_val[:, i], pheno_proba[:, i]))
    except Exception:
        pheno_acc = float("nan")
    pheno_auroc = float(np.mean(pheno_aucs)) if pheno_aucs else float("nan")

    results.append({"task": "pheno", "accuracy": pheno_acc, "auroc": pheno_auroc})

    return results


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
    n_batches_val: int,
    max_seq_len: int,
    splits_dir: str,
    device: torch.device,
    split: str = "test",
) -> list[dict]:
    """
    Extract embeddings for one (eps_label, mode, seed) config and run attack.
    Returns list of result rows (one per task).
    """
    torch.manual_seed(seed)
    clients, server = _build_clients_and_server(device)

    # Load checkpoint when available (weights only — no optimizer state needed)
    ckpt = _ckpt_path(ckpt_dir, eps_label, mode, seed)
    if ckpt.exists():
        _load_weights(ckpt, clients, server)
    elif not use_synthetic:
        print(f"  [label_inference] WARNING: checkpoint not found: {ckpt}")

    # Data
    if use_synthetic:
        n_tr = max(1, n_synthetic // batch_size)
        n_te = max(1, n_tr // 4)
        train_loaders = make_synthetic_loaders(batch_size, max_seq_len, n_tr)
        eval_loaders  = make_synthetic_loaders(batch_size, max_seq_len, n_te)
    else:
        from data_prep.dataset import build_site_loaders
        project_root = Path(splits_dir).parents[1]
        train_loaders = build_site_loaders(project_root, "train", batch_size)
        eval_loaders  = build_site_loaders(project_root, split,   batch_size)

    z_train, y_train = _extract_embeddings(clients, train_loaders, device)
    z_eval,  y_eval  = _extract_embeddings(clients, eval_loaders,  device)

    task_rows = run_label_inference(z_train, y_train, z_eval, y_eval)
    for r in task_rows:
        r.update({"epsilon_level": eps_label, "mode": mode, "seed": seed})
    return task_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    parser.add_argument("--privacy_csv",   default="results/privacy_utility_combined.csv",
                        help="CSV from privacy_utility_curves.py — used to discover (ε, mode) combos")
    parser.add_argument("--output",        default="results/label_inference.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--n_rounds",      type=int, default=50,
                        help="Used only for checkpoint path reconstruction")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--max_seq_len",   type=int, default=48)
    parser.add_argument("--split",         default="test", choices=["val", "test"],
                        help="Which held-out split to evaluate the probe on.")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Determine (epsilon_level, mode) combos from CSV or defaults
    combos: list[tuple[str, str]] = []
    privacy_csv = Path(args.privacy_csv)
    if privacy_csv.exists():
        import pandas as pd
        df = pd.read_csv(privacy_csv)
        combos = list(df[["epsilon_level", "mode"]].drop_duplicates().itertuples(index=False, name=None))
        print(f"[label_inference] {len(combos)} (ε, mode) combos from {privacy_csv}")
    else:
        combos = [(eps, "uniform") for eps in EPSILON_LEVELS] + [("5.0", "stratified")]
        print(f"[label_inference] privacy_utility_combined.csv not found; using defaults ({len(combos)} combos)")

    all_rows: list[dict] = []
    n_val_batches = max(1, args.n_synthetic // args.batch_size // 4)

    for eps_label, mode in combos:
        for seed in SEEDS:
            print(f"  ε={eps_label} | mode={mode} | seed={seed}")
            try:
                rows = _run_one(
                    eps_label=str(eps_label),
                    mode=mode,
                    seed=seed,
                    ckpt_dir=args.ckpt_dir,
                    use_synthetic=args.use_synthetic,
                    n_synthetic=args.n_synthetic,
                    batch_size=args.batch_size,
                    n_batches_val=n_val_batches,
                    max_seq_len=args.max_seq_len,
                    splits_dir=args.splits_dir,
                    device=device,
                    split=args.split,
                )
                all_rows.extend(rows)
            except Exception as e:
                print(f"  ERROR: {e}")

    if not all_rows:
        print("[label_inference] No results collected.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[label_inference] Done. {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
