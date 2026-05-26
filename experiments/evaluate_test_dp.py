"""
experiments/evaluate_test_dp.py — Final test-set evaluation for DP models.

Run ONCE after all design decisions are final. Evaluates every (ε_level, mode, seed)
checkpoint on the held-out TEST split — never validation.

For each checkpoint:
  1. Task AUC on test set (IHM, Decomp, Pheno)
  2. Label inference attack: LR probe fitted on TRAIN embeddings, tested on TEST
  3. MIA: binary classifier trained on TRAIN (members) vs TEST (non-members)

Checkpoints follow naming convention:
  uniform    → checkpoints/best_DP-uniform-eps{label}-seed{N}.pt
  stratified → checkpoints/best_DP-stratified-eps5-seed{N}.pt

(ε_level, mode) combinations are read from results/privacy_utility_combined.csv.
If that file does not exist, defaults to all EPSILON_LEVELS × ["uniform"] + ("5.0", "stratified").

Output: results/test_results_dp.csv
  columns: epsilon_level, mode, seed,
           ihm_auroc, ihm_auprc, decomp_auroc, decomp_auprc, pheno_macro_auroc,
           label_inf_ihm_auroc, label_inf_pheno_macro_auroc,
           mia_auc, mia_accuracy

Usage
-----
  # Smoke test:
  python experiments/evaluate_test_dp.py --use_synthetic --n_rounds 3

  # Full run (Snellius):
  python experiments/evaluate_test_dp.py \\
      --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \\
      --device cuda --num_workers 4
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
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import build_site_loaders
from experiments.metrics import ihm_metrics, decomp_metrics, pheno_metrics
from fl.client import VFLClient
from fl.server import VFLServer
from train import make_synthetic_loaders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS          = [42, 123, 7]
EPSILON_LEVELS = ["inf", "10.0", "5.0", "2.0", "1.0", "0.5"]
EMBED_DIM      = 64
SITE_INPUT_DIMS: dict[str, int] = {"A": 7, "B": 4, "C": 3}


# ---------------------------------------------------------------------------
# Checkpoint path
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: Path, eps_label: str, mode: str, seed: int) -> Path:
    if str(eps_label) == "inf":
        return ckpt_dir / f"best_VFL-MTL_seed{seed}.pt"
    if mode == "stratified":
        name = f"DP-stratified-eps5-seed{seed}"
    else:
        name = f"DP-uniform-eps{eps_label}-seed{seed}"
    return ckpt_dir / f"best_{name}_seed{seed}.pt"


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_model(device: torch.device) -> tuple[dict, VFLServer]:
    clients = {
        s: VFLClient(input_dim=SITE_INPUT_DIMS[s], embed_dim=EMBED_DIM, device=device)
        for s in ("A", "B", "C")
    }
    server = VFLServer(embed_dim=EMBED_DIM, device=device)
    return clients, server


def _load_weights(path: Path, clients: dict, server: VFLServer, device: torch.device) -> bool:
    if not path.exists():
        return False
    ckpt = torch.load(path, weights_only=True, map_location=device)
    for site in ("A", "B", "C"):
        if f"client_{site}" in ckpt:
            clients[site].encoder.load_state_dict(ckpt[f"client_{site}"])
    if "server" in ckpt:
        server.model.load_state_dict(ckpt["server"])
    return True


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_embeddings(
    clients: dict,
    loaders: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns z_A, z_C, y_ihm, y_pheno, z_concat (for MIA).
    z_A : (N, embed_dim), z_C : (N, embed_dim), z_concat : (N, 3*embed_dim)
    """
    for c in clients.values():
        c.encoder.eval()

    zA_list, zB_list, zC_list = [], [], []
    y_ihm_list, y_pheno_list = [], []

    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA
        xB, mB, _  = bB
        xC, mC, yP = bC

        zA = clients["A"].eval_forward(xA.to(device), mA.to(device)).cpu().numpy()
        zB = clients["B"].eval_forward(xB.to(device), mB.to(device)).cpu().numpy()
        zC = clients["C"].eval_forward(xC.to(device), mC.to(device)).cpu().numpy()

        zA_list.append(zA); zB_list.append(zB); zC_list.append(zC)
        y_ihm_list.append(yI.numpy()); y_pheno_list.append(yP.numpy())

    z_A     = np.concatenate(zA_list,     axis=0)
    z_B     = np.concatenate(zB_list,     axis=0)
    z_C     = np.concatenate(zC_list,     axis=0)
    y_ihm   = np.concatenate(y_ihm_list,  axis=0)
    y_pheno = np.concatenate(y_pheno_list, axis=0)
    z_concat = np.concatenate([z_A, z_B, z_C], axis=-1)

    return z_A, z_C, y_ihm, y_pheno, z_concat


# ---------------------------------------------------------------------------
# Task-AUC evaluation on test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_task_metrics(
    clients: dict,
    server: VFLServer,
    loaders: dict,
    device: torch.device,
) -> dict[str, float]:
    for c in clients.values():
        c.encoder.eval()
    server.model.eval()

    ihm_p, ihm_l = [], []
    dec_p, dec_l = [], []
    phn_p, phn_l = [], []

    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA; xB, mB, yD = bB; xC, mC, yP = bC
        embs = {
            "A": clients["A"].eval_forward(xA.to(device), mA.to(device)),
            "B": clients["B"].eval_forward(xB.to(device), mB.to(device)),
            "C": clients["C"].eval_forward(xC.to(device), mC.to(device)),
        }
        out = server.predict(embs)
        ihm_p.append(out["ihm"].squeeze(-1).cpu().numpy()); ihm_l.append(yI.numpy())
        dec_p.append(out["decomp"].squeeze(-1).cpu().numpy()); dec_l.append(yD.numpy())
        phn_p.append(out["pheno"].cpu().numpy()); phn_l.append(yP.numpy())

    p_ihm  = np.concatenate(ihm_p);  y_ihm  = np.concatenate(ihm_l)
    p_dec  = np.concatenate(dec_p);  y_dec  = np.concatenate(dec_l)
    p_phn  = np.concatenate(phn_p);  y_phn  = np.concatenate(phn_l)

    m_ihm  = ihm_metrics(y_ihm, p_ihm)
    m_dec  = decomp_metrics(y_dec, p_dec)
    m_phn  = pheno_metrics(y_phn, p_phn)

    return {
        "ihm_auroc":          m_ihm["auc_roc"],
        "ihm_auprc":          m_ihm["auc_pr"],
        "decomp_auroc":       m_dec["auc_roc"],
        "decomp_auprc":       m_dec["auc_pr"],
        "pheno_macro_auroc":  m_phn["macro_auc"],
    }


# ---------------------------------------------------------------------------
# Label inference attack
# ---------------------------------------------------------------------------

def _label_inference(
    z_A_train: np.ndarray, y_ihm_train: np.ndarray,
    z_C_train: np.ndarray, y_pheno_train: np.ndarray,
    z_A_test:  np.ndarray, y_ihm_test:  np.ndarray,
    z_C_test:  np.ndarray, y_pheno_test: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """
    Fit LR probe on training embeddings; evaluate on test embeddings.
    Probe 1: z_A → y_ihm  (binary)
    Probe 2: z_C → y_pheno (25 phenotype labels, macro-AUC)
    """
    results: dict[str, float] = {}

    # IHM probe (binary)
    try:
        clf_ihm = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
        clf_ihm.fit(z_A_train, y_ihm_train.astype(int))
        proba = clf_ihm.predict_proba(z_A_test)[:, 1]
        results["label_inf_ihm_auroc"] = float(roc_auc_score(y_ihm_test, proba))
    except Exception:
        results["label_inf_ihm_auroc"] = float("nan")

    # Pheno probe (25 binary labels → macro-AUC)
    try:
        clf_pheno = OneVsRestClassifier(
            LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
        )
        clf_pheno.fit(z_C_train, y_pheno_train.astype(int))
        proba = clf_pheno.predict_proba(z_C_test)
        aucs = []
        for col in range(proba.shape[1]):
            if len(np.unique(y_pheno_test[:, col])) > 1:
                aucs.append(roc_auc_score(y_pheno_test[:, col], proba[:, col]))
        results["label_inf_pheno_macro_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    except Exception:
        results["label_inf_pheno_macro_auroc"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Membership inference attack
# ---------------------------------------------------------------------------

def _mia(
    z_members: np.ndarray,
    z_nonmembers: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """
    Binary LR classifier: TRAIN embeddings (members) vs TEST embeddings (non-members).
    Target under sufficient DP: AUC ≈ 0.50.
    """
    n = min(len(z_members), len(z_nonmembers))
    rng = np.random.default_rng(seed)
    idx_m  = rng.choice(len(z_members),    n, replace=False)
    idx_nm = rng.choice(len(z_nonmembers), n, replace=False)

    X = np.concatenate([z_members[idx_m], z_nonmembers[idx_nm]], axis=0)
    y = np.array([1] * n + [0] * n)

    if n < 4:
        return {"mia_auc": float("nan"), "mia_accuracy": float("nan")}

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else float("nan")
        acc = float(accuracy_score(y_te, clf.predict(X_te)))
    except Exception:
        auc, acc = float("nan"), float("nan")

    return {"mia_auc": auc, "mia_accuracy": acc}


# ---------------------------------------------------------------------------
# Per-combination runner
# ---------------------------------------------------------------------------

def _run_one(
    eps_label: str,
    mode: str,
    seed: int,
    ckpt_dir: Path,
    train_loaders: dict,
    test_loaders: dict,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    clients, server = _build_model(device)
    ckpt = _ckpt_path(ckpt_dir, eps_label, mode, seed)
    found = _load_weights(ckpt, clients, server, device)
    if not found:
        print(f"    [WARNING] checkpoint not found: {ckpt.name}")

    # Task metrics on test set
    task_m = _eval_task_metrics(clients, server, test_loaders, device)

    # Embeddings: train = members/probe-train, test = non-members/probe-test
    zA_tr, zC_tr, y_ihm_tr, y_phn_tr, z_cat_tr = _extract_embeddings(
        clients, train_loaders, device)
    zA_te, zC_te, y_ihm_te, y_phn_te, z_cat_te = _extract_embeddings(
        clients, test_loaders, device)

    # Label inference (probe fitted on train, evaluated on test)
    li_m = _label_inference(
        zA_tr, y_ihm_tr, zC_tr, y_phn_tr,
        zA_te, y_ihm_te, zC_te, y_phn_te,
        seed=seed,
    )

    # MIA (train = members, test = non-members — using concatenated embeddings)
    mia_m = _mia(z_cat_tr, z_cat_te, seed=seed)

    return {
        "epsilon_level": eps_label,
        "mode":          mode,
        "seed":          seed,
        **task_m,
        **li_m,
        **mia_m,
    }


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
                        help="Used to discover (ε_level, mode) combinations. Falls back to defaults.")
    parser.add_argument("--output",        default="results/test_results_dp.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--max_seq_len",   type=int, default=48)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("FINAL TEST-SET EVALUATION — PAPER 2 DP MODELS")
    print("Run only once, after all DP training runs are complete.")
    print("=" * 60)

    device   = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)

    # (ε, mode) combinations to evaluate
    combos: list[tuple[str, str]] = []
    pcsv = Path(args.privacy_csv)
    if pcsv.exists():
        import pandas as pd
        df = pd.read_csv(pcsv)
        combos = list(
            df[["epsilon_level", "mode"]].drop_duplicates().itertuples(index=False, name=None)
        )
        print(f"[evaluate_test_dp] {len(combos)} (ε, mode) combos from {pcsv}")
    else:
        combos = [(eps, "uniform") for eps in EPSILON_LEVELS] + [("5.0", "stratified")]
        print(f"[evaluate_test_dp] privacy_utility_combined.csv not found; using {len(combos)} defaults")

    # Data loaders — built once, reused across all (ε, mode, seed) combos
    if args.use_synthetic:
        n_tr  = max(1, args.n_synthetic // args.batch_size)
        n_te  = max(1, n_tr // 4)
        train_loaders = make_synthetic_loaders(args.batch_size, args.max_seq_len, n_tr)
        test_loaders  = make_synthetic_loaders(args.batch_size, args.max_seq_len, n_te)
    else:
        project_root  = Path(args.splits_dir).parents[1]
        print("[evaluate_test_dp] Loading train and test data loaders...")
        train_loaders = build_site_loaders(
            project_root, "train", args.batch_size,
            num_workers=args.num_workers, max_seq_len=args.max_seq_len,
        )
        test_loaders  = build_site_loaders(
            project_root, "test", args.batch_size,
            num_workers=args.num_workers, max_seq_len=args.max_seq_len,
        )

    # Evaluate
    all_rows: list[dict] = []
    for eps_label, mode in combos:
        for seed in SEEDS:
            print(f"  ε={eps_label} | mode={mode} | seed={seed}", end=" ... ", flush=True)
            try:
                row = _run_one(
                    eps_label=str(eps_label),
                    mode=str(mode),
                    seed=seed,
                    ckpt_dir=ckpt_dir,
                    train_loaders=train_loaders,
                    test_loaders=test_loaders,
                    device=device,
                )
                all_rows.append(row)
                print(f"IHM={row['ihm_auroc']:.4f} "
                      f"Decomp={row['decomp_auroc']:.4f} "
                      f"Pheno={row['pheno_macro_auroc']:.4f} "
                      f"MIA={row['mia_auc']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

    if not all_rows:
        print("[evaluate_test_dp] No results collected — run DP training scripts first.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon_level", "mode", "seed",
        "ihm_auroc", "ihm_auprc",
        "decomp_auroc", "decomp_auprc",
        "pheno_macro_auroc",
        "label_inf_ihm_auroc", "label_inf_pheno_macro_auroc",
        "mia_auc", "mia_accuracy",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[evaluate_test_dp] Done. {len(all_rows)} rows → {out}")
    _print_summary(all_rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["ihm_auroc", "decomp_auroc", "pheno_macro_auroc", "mia_auc"]
    header = f"{'mode':12s} {'ε':>8s} │" + "".join(f" {c:>22s}" for c in cols)
    print("\n── Test-Set Summary (mean across seeds) ──")
    print(header)
    print("─" * len(header))
    for (mode, eps), grp in df.groupby(["mode", "epsilon_level"]):
        vals = []
        for c in cols:
            col = grp[c].dropna()
            vals.append(f"{col.mean():.4f}±{col.std():.4f}" if not col.empty else "—")
        print(f"{mode:12s} {str(eps):>8s} │" + "".join(f" {v:>22s}" for v in vals))
    print("─" * len(header))


if __name__ == "__main__":
    main()
