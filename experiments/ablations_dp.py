"""
experiments/ablations_dp.py — DP ablations.

Abl 1 — uniform vs. task-stratified σ at ε=5; reads privacy_utility_combined.csv (no new training).
Abl 2 — related (IHM+Decomp) vs. unrelated (IHM+Pheno) task pair at ε=5;
         trains two 2-task configs and runs label inference to test coupling amplification.
Abl 3 — embed_dim ∈ {32, 64, 128} × ε ∈ {1, 5, ∞};
         larger embed_dim degrades SNR under DP (SNR = 1/(embed_dim × σ²)),
         so ε* is embed_dim-dependent. Validates ε* at embed_dim=64.

Output: results/dp_ablations.csv
  columns: ablation, config, embed_dim, epsilon_level, seed, val_ihm_auroc,
           val_decomp_auroc, val_pheno_macro_auroc, rho,
           inference_auroc_ihm, inference_accuracy_ihm

Usage
-----
  python experiments/ablations_dp.py --use_synthetic --n_rounds 3
  python experiments/ablations_dp.py --splits_dir data/vertical_splits --n_rounds 100
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_training, TrainConfig, make_synthetic_loaders
from attacks.label_inference import (
    _build_clients_and_server,
    _load_weights,
    _extract_embeddings,
    run_label_inference,
    _ckpt_path,
)

SEEDS          = [42, 123, 7]
DELTA          = 1e-5
MAX_GRAD_NORM  = 1.0
ABL2_EPS       = 5.0
EMBED_DIM      = 64

ABL2_CONFIGS = {
    "ihm_decomp": {"task_weights": {"ihm": 1.0, "decomp": 1.0, "pheno": 0.0}},
    "ihm_pheno":  {"task_weights": {"ihm": 1.0, "decomp": 0.0, "pheno": 1.0}},
}

ABL3_EMBED_DIMS = [32, 64, 128]
ABL3_EPS_LEVELS = [1.0, 5.0, float("inf")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sigma(eps: float, sample_rate: float, n_rounds: int) -> float | None:
    if not math.isfinite(eps):
        return None
    from opacus.accountants.utils import get_noise_multiplier
    return float(get_noise_multiplier(
        target_epsilon=eps,
        target_delta=DELTA,
        sample_rate=sample_rate,
        epochs=n_rounds,
    ))


def _last_row_metrics(
    df: pd.DataFrame, config_col: str, config_val: str, seed: int
) -> dict:
    """Return last-round val metrics for a given (config, seed) from a per-round CSV."""
    mask = (df[config_col] == config_val) & (df["seed"] == seed)
    sub = df[mask]
    if sub.empty:
        return {}
    return sub.sort_values("round").iloc[-1].to_dict()


def _extract_rho(results: list[dict]) -> dict[str, float]:
    """Return mean grad_sim values from training results (last logged, not NaN)."""
    rho: dict[str, float] = {}
    for key in ("grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno"):
        vals = [r[key] for r in results if key in r and r[key] == r[key]]
        rho[key] = float(np.mean(vals)) if vals else float("nan")
    return rho


# ---------------------------------------------------------------------------
# Ablation 1 — read test-set results from test_results_dp.csv
# ---------------------------------------------------------------------------

def run_abl1(test_dp_csv: Path) -> list[dict]:
    """Read uniform vs stratified AUC at ε=5 from the test-set evaluation CSV."""
    if not test_dp_csv.exists():
        print(f"[abl1] {test_dp_csv} not found — skipping Abl 1.")
        return []

    df = pd.read_csv(test_dp_csv)
    df_eps5 = df[df["epsilon_level"].astype(str) == "5.0"]

    rows = []
    for mode in ("uniform", "stratified"):
        df_mode = df_eps5[df_eps5["mode"] == mode]
        if df_mode.empty:
            print(f"[abl1] No rows for mode={mode} at ε=5.0 — skipping.")
            continue
        for _, row in df_mode.iterrows():
            seed = int(row["seed"])
            rows.append({
                "ablation":               "abl1",
                "config":                 f"{mode}_eps5",
                "seed":                   seed,
                "val_ihm_auroc":          row.get("ihm_auroc",            float("nan")),
                "val_decomp_auroc":       row.get("decomp_auroc",         float("nan")),
                "val_pheno_macro_auroc":  row.get("pheno_macro_auroc",    float("nan")),
                "rho":                    float("nan"),
                "inference_auroc_ihm":    float("nan"),
                "inference_accuracy_ihm": float("nan"),
            })
    return rows


# ---------------------------------------------------------------------------
# Ablation 2 — train 2-task configs + label inference
# ---------------------------------------------------------------------------

def run_abl2(
    args: argparse.Namespace,
    sample_rate: float,
    prebuilt: dict | None,
    decomp_pos_weight: float,
) -> list[dict]:
    sigma = _compute_sigma(ABL2_EPS, sample_rate, args.n_rounds)
    if sigma is None:
        print("[abl2] Could not compute σ for ε=5 — skipping Abl 2.")
        return []

    privacy_cfg = {
        "mode":          "uniform",
        "sigma":         sigma,
        "max_grad_norm": MAX_GRAD_NORM,
        "delta":         DELTA,
    }
    print(f"[abl2] ε={ABL2_EPS} → σ={sigma:.4f}")

    device = torch.device(args.device)
    rows: list[dict] = []

    for config_name, config_kwargs in ABL2_CONFIGS.items():
        for seed in SEEDS:
            model_name = f"abl2_{config_name}"
            print(f"\n=== Abl2 | {config_name} | seed={seed} ===")

            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                use_fedavg=True,
                fedavg_every=5,
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
                patience=args.patience,
                decomp_pos_weight=decomp_pos_weight,
                model_name=model_name,
                uncertainty_weighting=False,  # single σ per task; no log_var params needed
                grad_sim_every=5,             # capture ρ every 5 rounds
                privacy_config=privacy_cfg,
                **config_kwargs,
            )
            results = run_training(cfg, prebuilt_loaders=prebuilt)

            # Last-round val metrics
            last = results[-1] if results else {}
            rho_dict = _extract_rho(results)

            # Determine the dominant ρ for this config
            if config_name == "ihm_decomp":
                rho = rho_dict.get("grad_sim_ihm_decomp", float("nan"))
            else:
                rho = rho_dict.get("grad_sim_ihm_pheno", float("nan"))

            # Label inference attack on the trained checkpoint
            inf_auroc, inf_acc = float("nan"), float("nan")
            try:
                ckpt = Path(args.ckpt_dir) / f"best_{model_name}_seed{seed}.pt"
                clients, server = _build_clients_and_server(device)
                if ckpt.exists():
                    _load_weights(ckpt, clients, server)

                if args.use_synthetic:
                    n_tr = max(1, args.n_synthetic // args.batch_size)
                    n_te = max(1, n_tr // 4)
                    train_loaders = make_synthetic_loaders(args.batch_size, 48, n_tr)
                    test_loaders  = make_synthetic_loaders(args.batch_size, 48, n_te)
                else:
                    from data_prep.dataset import build_site_loaders
                    project_root  = Path(args.splits_dir).parents[1]
                    train_loaders = build_site_loaders(project_root, "train",      args.batch_size)
                    test_loaders  = build_site_loaders(project_root, args.split,   args.batch_size)

                z_train, y_train = _extract_embeddings(clients, train_loaders, device)
                z_test,  y_test  = _extract_embeddings(clients, test_loaders,  device)
                attack_rows = run_label_inference(z_train, y_train, z_test, y_test)

                ihm_row = next((r for r in attack_rows if r["task"] == "ihm"), {})
                inf_auroc = ihm_row.get("auroc",    float("nan"))
                inf_acc   = ihm_row.get("accuracy", float("nan"))
            except Exception as e:
                print(f"  [abl2] label inference failed: {e}")

            rows.append({
                "ablation":               "abl2",
                "config":                 config_name,
                "seed":                   seed,
                "rho":                    round(rho, 6),
                "inference_auroc_ihm":    round(inf_auroc, 6) if inf_auroc == inf_auroc else float("nan"),
                "inference_accuracy_ihm": round(inf_acc, 6)   if inf_acc   == inf_acc   else float("nan"),
            })

    return rows


# ---------------------------------------------------------------------------
# Ablation 3 — embed_dim × DP interaction
# ---------------------------------------------------------------------------

def run_abl3(
    args: argparse.Namespace,
    sample_rate: float,
    prebuilt: dict | None,
    decomp_pos_weight: float,
) -> list[dict]:
    rows: list[dict] = []

    for embed_dim in ABL3_EMBED_DIMS:
        for eps in ABL3_EPS_LEVELS:
            sigma = _compute_sigma(eps, sample_rate, args.n_rounds)
            privacy_cfg = (
                None if sigma is None
                else {"mode": "uniform", "sigma": sigma,
                      "max_grad_norm": MAX_GRAD_NORM, "delta": DELTA}
            )
            eps_label = "inf" if not math.isfinite(eps) else f"{eps:g}"
            print(f"\n── Abl3 | embed_dim={embed_dim} | ε={eps_label} ──")

            for seed in SEEDS:
                model_name = f"abl3_embed{embed_dim}_eps{eps_label}"
                print(f"  seed={seed}")

                cfg = TrainConfig(
                    splits_dir=args.splits_dir,
                    n_rounds=args.n_rounds,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed,
                    use_fedavg=True,
                    fedavg_every=5,
                    use_synthetic=args.use_synthetic,
                    n_synthetic=args.n_synthetic,
                    patience=args.patience,
                    decomp_pos_weight=decomp_pos_weight,
                    model_name=model_name,
                    embed_dim=embed_dim,
                    privacy_config=privacy_cfg,
                )
                results = run_training(cfg, prebuilt_loaders=prebuilt)
                last = results[-1] if results else {}

                rows.append({
                    "ablation":               "abl3",
                    "config":                 f"embed{embed_dim}_eps{eps_label}",
                    "embed_dim":              embed_dim,
                    "epsilon_level":          eps,
                    "seed":                   seed,
                    "val_ihm_auroc":          last.get("val_ihm_auroc",         float("nan")),
                    "val_decomp_auroc":       last.get("val_decomp_auroc",      float("nan")),
                    "val_pheno_macro_auroc":  last.get("val_pheno_macro_auroc", float("nan")),
                    "rho":                    float("nan"),
                    "inference_auroc_ihm":    float("nan"),
                    "inference_accuracy_ihm": float("nan"),
                })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    parser.add_argument("--test_dp_csv",   default="results/test_results_dp.csv",
                        help="Test-set DP evaluation CSV for Abl 1 (uniform vs stratified at ε=5).")
    parser.add_argument("--output",        default="results/dp_ablations.csv")
    parser.add_argument("--n_rounds",      type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--patience",      type=int, default=15)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    parser.add_argument("--split",         default="test", choices=["val", "test"],
                        help="Held-out split for label inference probe evaluation in Abl 2.")
    args = parser.parse_args()

    # ---- Data ----
    decomp_pos_weight = 1.0
    prebuilt = None

    if not args.use_synthetic:
        print("[ablations_dp] Pre-loading data loaders...")
        project_root = Path(args.splits_dir).parents[1]
        site_b_csv   = Path(args.splits_dir) / "site_B_labs.csv"
        _b = pd.read_csv(site_b_csv, usecols=["y_decomp", "split"])
        pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
        decomp_pos_weight = (1.0 - pos_rate) / pos_rate
        prebuilt = {
            "train": __import__("data_prep.dataset", fromlist=["build_site_loaders"])
                     .build_site_loaders(project_root, "train", args.batch_size),
            "val":   __import__("data_prep.dataset", fromlist=["build_site_loaders"])
                     .build_site_loaders(project_root, "val",   args.batch_size),
            "decomp_pos_weight": decomp_pos_weight,
        }

    # Sample rate for σ computation
    if args.use_synthetic:
        n_batches   = max(1, args.n_synthetic // args.batch_size)
        sample_rate = 1.0 / n_batches
    else:
        assert prebuilt is not None
        sample_rate = 1.0 / max(len(prebuilt["train"]["A"]), 1)

    print(f"[ablations_dp] sample_rate={sample_rate:.5f}  n_rounds={args.n_rounds}")

    # ---- Run ablations ----
    all_rows: list[dict] = []

    print("\n── Abl 1: uniform vs. stratified at ε=5 ──")
    all_rows.extend(run_abl1(Path(args.test_dp_csv)))

    print("\n── Abl 2: related vs. unrelated task pair under DP ──")
    all_rows.extend(run_abl2(args, sample_rate, prebuilt, decomp_pos_weight))

    print("\n── Abl 3: embed_dim × DP interaction ──")
    all_rows.extend(run_abl3(args, sample_rate, prebuilt, decomp_pos_weight))

    if not all_rows:
        print("[ablations_dp] No results — exiting.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # union of all row keys to handle different column sets across ablations
    fieldnames: list[str] = []
    for row in all_rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                restval=float("nan"))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[ablations_dp] Done. {len(all_rows)} rows → {out}")
    _print_summary(all_rows)


def _print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n── Abl 1: uniform vs. stratified at ε=5 (mean across seeds) ──")
    abl1 = df[df["ablation"] == "abl1"]
    if not abl1.empty:
        for config, grp in abl1.groupby("config"):
            ihm  = grp["val_ihm_auroc"].mean()
            dec  = grp["val_decomp_auroc"].mean()
            phen = grp["val_pheno_macro_auroc"].mean()
            print(f"  {config:25s} IHM={ihm:.4f}  Decomp={dec:.4f}  Pheno={phen:.4f}")

    print("\n── Abl 2: coupling amplification (mean across seeds) ──")
    abl2 = df[df["ablation"] == "abl2"]
    if not abl2.empty:
        for config, grp in abl2.groupby("config"):
            rho = grp["rho"].mean()
            inf = grp["inference_auroc_ihm"].mean()
            print(f"  {config:12s}  ρ={rho:.4f}  IHM inference AUC={inf:.4f}")

    print("\n── Abl 3: embed_dim × DP (mean across seeds) ──")
    abl3 = df[df["ablation"] == "abl3"]
    if not abl3.empty:
        for config, grp in abl3.groupby("config"):
            ihm  = grp["val_ihm_auroc"].mean()
            dec  = grp["val_decomp_auroc"].mean()
            phen = grp["val_pheno_macro_auroc"].mean()
            print(f"  {config:25s} IHM={ihm:.4f}  Decomp={dec:.4f}  Pheno={phen:.4f}")


if __name__ == "__main__":
    main()
