"""
experiments/evaluate_test_ablations_dp.py — Test-set evaluation for DP ablation models.

Mirrors evaluate_ablations.py for the base architecture ablations, but covers the DP ablation checkpoints
that were trained by ablations_dp.py (Abl 2 and Abl 3). Those scripts logged last-round
val metrics during training; this script provides the proper test-set inference step.

Abl 2 checkpoints (2-task configs at ε=5, embed_dim=64):
    best_abl2_ihm_decomp_seed{N}.pt  — active tasks: IHM + Decomp
    best_abl2_ihm_pheno_seed{N}.pt   — active tasks: IHM + Pheno

Abl 3 checkpoints (embed_dim × ε grid, all 3 tasks):
    best_abl3_embed{D}_eps{E}_seed{N}.pt  — D ∈ {32,64,128}, E ∈ {inf,1,5}

Output: results/test_ablations_dp.csv
  columns: ablation, config, embed_dim, epsilon_level, seed,
           ihm_auroc, ihm_auprc, decomp_auroc, decomp_auprc, pheno_macro_auroc,
           label_inf_ihm_auroc, label_inf_ihm_accuracy

Smoke test (--use_synthetic) writes to results/test_ablations_dp_smoketest.csv
so it never touches real results.

Usage
-----
  # Smoke test (local):
  python experiments/evaluate_test_ablations_dp.py --use_synthetic

  # Full run (Snellius):
  python experiments/evaluate_test_ablations_dp.py \\
      --root /home/asoare/vfl_mlt \\
      --ckpt_dir checkpoints \\
      --device cpu
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import build_site_loaders
from experiments.metrics import ihm_metrics, decomp_metrics, pheno_metrics
from fl.client import VFLClient
from fl.server import VFLServer
from train import make_synthetic_loaders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 7]
SITE_INPUT_DIMS = {"A": 7, "B": 4, "C": 3}

ABL2_CONFIGS = {
    "ihm_decomp": {"active": ("ihm", "decomp")},
    "ihm_pheno":  {"active": ("ihm", "pheno")},
}

ABL3_EMBED_DIMS  = [32, 64, 128]
ABL3_EPS_LABELS  = ["inf", "1", "5"]

OUTPUT_COLS = [
    "ablation", "config", "embed_dim", "epsilon_level", "seed",
    "ihm_auroc", "ihm_auprc",
    "decomp_auroc", "decomp_auprc",
    "pheno_macro_auroc",
    "label_inf_ihm_auroc", "label_inf_ihm_accuracy",
]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _infer_embed_dim(ckpt: dict) -> int:
    return int(ckpt["client_A"]["projection.weight"].shape[0])


def _build_model(ckpt: dict, device: torch.device) -> tuple[dict, VFLServer]:
    embed_dim  = _infer_embed_dim(ckpt)
    hidden_dim = int(ckpt["client_A"]["lstm.weight_ih_l0"].shape[0] // 4)
    clients = {
        s: VFLClient(
            input_dim=SITE_INPUT_DIMS[s],
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            lr=1e-3,
            device=device,
        )
        for s in ("A", "B", "C")
    }
    server = VFLServer(embed_dim=embed_dim, device=device)
    for s in ("A", "B", "C"):
        clients[s].encoder.load_state_dict(ckpt[f"client_{s}"])
    server.model.load_state_dict(ckpt["server"])
    return clients, server


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_tasks(
    clients: dict,
    server: VFLServer,
    loaders: dict,
    active_tasks: tuple[str, ...],
) -> dict[str, float]:
    for c in clients.values():
        c.encoder.eval()
    server.model.eval()

    preds:  dict[str, list] = {t: [] for t in ("ihm", "decomp", "pheno")}
    labels: dict[str, list] = {t: [] for t in ("ihm", "decomp", "pheno")}

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

    results: dict[str, float] = {
        "ihm_auroc": float("nan"), "ihm_auprc": float("nan"),
        "decomp_auroc": float("nan"), "decomp_auprc": float("nan"),
        "pheno_macro_auroc": float("nan"),
    }

    if "ihm" in active_tasks:
        m = ihm_metrics(np.concatenate(labels["ihm"]), np.concatenate(preds["ihm"]))
        results["ihm_auroc"] = m["auc_roc"]
        results["ihm_auprc"] = m["auc_pr"]

    if "decomp" in active_tasks:
        m = decomp_metrics(np.concatenate(labels["decomp"]), np.concatenate(preds["decomp"]))
        results["decomp_auroc"] = m["auc_roc"]
        results["decomp_auprc"] = m["auc_pr"]

    if "pheno" in active_tasks:
        m = pheno_metrics(np.concatenate(labels["pheno"]), np.concatenate(preds["pheno"]))
        results["pheno_macro_auroc"] = m["macro_auc"]

    return results


@torch.no_grad()
def _extract_zA(clients: dict, loaders: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract Site A embeddings and IHM labels for label inference."""
    clients["A"].encoder.eval()
    zA_list, y_list = [], []
    for bA, _, _ in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA
        z = clients["A"].eval_forward(xA, mA).cpu().numpy()
        zA_list.append(z)
        y_list.append(yI.numpy())
    return np.concatenate(zA_list), np.concatenate(y_list)


def _label_inference_ihm(
    z_train: np.ndarray, y_train: np.ndarray,
    z_test:  np.ndarray, y_test:  np.ndarray,
    seed: int,
) -> dict[str, float]:
    """LR probe: fit on train embeddings, evaluate on test embeddings."""
    try:
        clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
        clf.fit(z_train, y_train.astype(int))
        proba = clf.predict_proba(z_test)[:, 1]
        preds = clf.predict(z_test)
        auroc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
        acc   = float(accuracy_score(y_test, preds))
    except Exception:
        auroc, acc = float("nan"), float("nan")
    return {"label_inf_ihm_auroc": auroc, "label_inf_ihm_accuracy": acc}


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def _eval_checkpoint(
    ckpt_path: Path,
    ablation: str,
    config: str,
    embed_dim_override: int | None,
    epsilon_level: str,
    seed: int,
    active_tasks: tuple[str, ...],
    train_loaders: dict,
    test_loaders: dict,
    device: torch.device,
    run_label_inf: bool,
) -> dict:
    row: dict = {
        "ablation":      ablation,
        "config":        config,
        "embed_dim":     embed_dim_override,
        "epsilon_level": epsilon_level,
        "seed":          seed,
        "ihm_auroc": float("nan"), "ihm_auprc": float("nan"),
        "decomp_auroc": float("nan"), "decomp_auprc": float("nan"),
        "pheno_macro_auroc": float("nan"),
        "label_inf_ihm_auroc": float("nan"),
        "label_inf_ihm_accuracy": float("nan"),
    }

    if not ckpt_path.exists():
        print(f"  [SKIP] {ckpt_path.name} not found")
        return row

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    clients, server = _build_model(ckpt, device)
    if embed_dim_override is None:
        row["embed_dim"] = _infer_embed_dim(ckpt)

    task_m = _eval_tasks(clients, server, test_loaders, active_tasks)
    row.update(task_m)

    if run_label_inf and "ihm" in active_tasks:
        z_tr, y_tr = _extract_zA(clients, train_loaders)
        z_te, y_te = _extract_zA(clients, test_loaders)
        li_m = _label_inference_ihm(z_tr, y_tr, z_te, y_te, seed=seed)
        row.update(li_m)

    ihm_str   = f"IHM={row['ihm_auroc']:.4f}" if row['ihm_auroc'] == row['ihm_auroc'] else "IHM=N/A"
    decomp_str = f"Decomp={row['decomp_auroc']:.4f}" if row['decomp_auroc'] == row['decomp_auroc'] else "Decomp=N/A"
    pheno_str  = f"Pheno={row['pheno_macro_auroc']:.4f}" if row['pheno_macro_auroc'] == row['pheno_macro_auroc'] else "Pheno=N/A"
    print(f"  {config} seed={seed}: {ihm_str} {decomp_str} {pheno_str}")

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root",          default=".")
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    parser.add_argument("--output",        default="results/test_ablations_dp.csv")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    # Smoke test writes to a separate file — never overwrites real results
    if args.use_synthetic and args.output == "results/test_ablations_dp.csv":
        args.output = "results/test_ablations_dp_smoketest.csv"

    device   = torch.device(args.device)
    ckpt_dir = Path(args.root) / args.ckpt_dir

    print("=" * 60)
    print("DP ABLATION TEST-SET EVALUATION (Abl 2 + Abl 3)")
    print(f"Output: {args.output}")
    print("=" * 60)

    if args.use_synthetic:
        print("[WARNING] Synthetic mode — no real checkpoints, random data.")
        n_tr = max(1, 256 // args.batch_size)
        n_te = max(1, n_tr // 4)
        train_loaders = make_synthetic_loaders(args.batch_size, 48, n_tr)
        test_loaders  = make_synthetic_loaders(args.batch_size, 48, n_te)
    else:
        project_root = Path(args.root)
        print("[evaluate_test_ablations_dp] Loading train and test data loaders...")
        train_loaders = build_site_loaders(project_root, "train", args.batch_size)
        test_loaders  = build_site_loaders(project_root, "test",  args.batch_size)

    all_rows: list[dict] = []

    # ---- Abl 2 ----
    print("\n--- Ablation 2: task coupling (ihm_decomp vs ihm_pheno at ε=5) ---")
    for config_name, cfg in ABL2_CONFIGS.items():
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_abl2_{config_name}_seed{seed}.pt"
            row = _eval_checkpoint(
                ckpt_path     = ckpt_path,
                ablation      = "abl2",
                config        = config_name,
                embed_dim_override = 64,
                epsilon_level = "5",
                seed          = seed,
                active_tasks  = cfg["active"],
                train_loaders = train_loaders,
                test_loaders  = test_loaders,
                device        = device,
                run_label_inf = True,
            )
            all_rows.append(row)

    # ---- Abl 3 ----
    print("\n--- Ablation 3: embed_dim × ε interaction ---")
    for embed_dim in ABL3_EMBED_DIMS:
        for eps_label in ABL3_EPS_LABELS:
            for seed in SEEDS:
                ckpt_path = ckpt_dir / f"best_abl3_embed{embed_dim}_eps{eps_label}_seed{seed}.pt"
                row = _eval_checkpoint(
                    ckpt_path     = ckpt_path,
                    ablation      = "abl3",
                    config        = f"embed{embed_dim}_eps{eps_label}",
                    embed_dim_override = embed_dim,
                    epsilon_level = eps_label,
                    seed          = seed,
                    active_tasks  = ("ihm", "decomp", "pheno"),
                    train_loaders = train_loaders,
                    test_loaders  = test_loaders,
                    device        = device,
                    run_label_inf = False,
                )
                all_rows.append(row)

    if not all_rows:
        print("[WARNING] No rows collected.")
        return

    out = Path(args.root) / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
