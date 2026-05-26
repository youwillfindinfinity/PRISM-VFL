"""
experiments/run_baselines.py — Primary results table: VFL-MTL vs. all baselines.

Answers the main RQ: does VFL-MTL achieve competitive performance vs. the
centralised oracle while operating under privacy/communication constraints?

Compares (per task, per seed):
  centralized_oracle  : all 14 features, no VFL, no privacy — upper bound
  local_A / local_B / local_C : each site independent, no FL, no MTL — lower bound
  VFL-MTL             : full system (loaded from exp1.csv after run_exp1.py)

Seeds: [42, 123, 7]

Output: results/baselines_comparison.csv
  columns: model, seed, ihm_auroc, ihm_auprc, decomp_auroc, decomp_auprc,
           pheno_macro_auroc

Usage:
    # Smoke test (synthetic, no MIMIC required):
    python experiments/run_baselines.py --use_synthetic --n_epochs 3

    # Full run on Snellius (re-runs baselines from scratch):
    python experiments/run_baselines.py \
        --root /home/asoare/vfl_mlt \
        --n_epochs 50 \
        --vfl_results results/exp1.csv

    # Use pre-computed baseline CSVs (skip rerun, just aggregate):
    python experiments/run_baselines.py \
        --skip_rerun \
        --local_a  results/local_only_A.csv \
        --local_b  results/local_only_B.csv \
        --local_c  results/local_only_C.csv \
        --central  results/centralized.csv \
        --vfl_results results/exp1.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.local_only import train_local
from baselines.centralized import train_centralized

SEEDS = [42, 123, 7]


# ---------------------------------------------------------------------------
# Helpers to extract final-epoch metrics from each baseline's CSV format
# ---------------------------------------------------------------------------

def _last_epoch_mean_std(df: pd.DataFrame, seed_col: str, metric_col: str) -> tuple[float, float]:
    """Return mean and std of `metric_col` at the last epoch, across seeds."""
    last = df.groupby(seed_col)[metric_col].last()
    return float(last.mean()), float(last.std(skipna=True))


def _extract_local(path: str, site: str) -> dict[str, list[dict]]:
    """
    Load local_only_{site}.csv and return per-seed final-epoch metrics.
    Returns list of dicts keyed by seed with the relevant metric columns.
    """
    df = pd.read_csv(path)
    rows = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("epoch").iloc[-1]
        row = {"model": f"local_{site}", "seed": int(seed)}
        if site == "A":
            row["ihm_auroc"]   = float(last.get("val_auc_roc", float("nan")))
            row["ihm_auprc"]   = float(last.get("val_auc_pr",  float("nan")))
            row["decomp_auroc"]        = float("nan")
            row["decomp_auprc"]        = float("nan")
            row["pheno_macro_auroc"]   = float("nan")
        elif site == "B":
            row["ihm_auroc"]           = float("nan")
            row["ihm_auprc"]           = float("nan")
            row["decomp_auroc"]        = float(last.get("val_auc_roc", float("nan")))
            row["decomp_auprc"]        = float(last.get("val_auc_pr",  float("nan")))
            row["pheno_macro_auroc"]   = float("nan")
        elif site == "C":
            row["ihm_auroc"]           = float("nan")
            row["ihm_auprc"]           = float("nan")
            row["decomp_auroc"]        = float("nan")
            row["decomp_auprc"]        = float("nan")
            row["pheno_macro_auroc"]   = float(last.get("val_macro_auc", float("nan")))
        rows.append(row)
    return rows


def _extract_centralized(path: str) -> list[dict]:
    """Load centralized.csv and return per-seed final-epoch metrics."""
    df = pd.read_csv(path)
    rows = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("epoch").iloc[-1]
        row = {
            "model":              "centralized_oracle",
            "seed":               int(seed),
            "ihm_auroc":          float(last.get("val_ihm_auc_roc",    float("nan"))),
            "ihm_auprc":          float(last.get("val_ihm_auc_pr",     float("nan"))),
            "decomp_auroc":       float(last.get("val_decomp_auc_roc", float("nan"))),
            "decomp_auprc":       float(last.get("val_decomp_auc_pr",  float("nan"))),
            "pheno_macro_auroc":  float(last.get("val_pheno_macro_auc",float("nan"))),
        }
        rows.append(row)
    return rows


def _extract_vfl_mtl(path: str) -> list[dict]:
    """
    Load exp1.csv, filter VFL-MTL rows, return per-seed final-round metrics.
    exp1.csv columns: model, seed, round, val_ihm_auroc, val_ihm_auprc,
                      val_decomp_auroc, val_decomp_auprc, val_pheno_macro_auroc
    """
    df = pd.read_csv(path)
    df = df[df["model"] == "VFL-MTL"].copy()
    if df.empty:
        raise ValueError(
            f"No VFL-MTL rows found in {path}. "
            "Run experiments/run_exp1.py first."
        )
    rows = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("round").iloc[-1]
        row = {
            "model":             "VFL-MTL",
            "seed":              int(seed),
            "ihm_auroc":         float(last.get("val_ihm_auroc",         float("nan"))),
            "ihm_auprc":         float(last.get("val_ihm_auprc",         float("nan"))),
            "decomp_auroc":      float(last.get("val_decomp_auroc",      float("nan"))),
            "decomp_auprc":      float(last.get("val_decomp_auprc",      float("nan"))),
            "pheno_macro_auroc": float(last.get("val_pheno_macro_auroc", float("nan"))),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run baselines with synthetic data
# ---------------------------------------------------------------------------

def _run_baselines_synthetic(n_epochs: int, batch_size: int) -> tuple[list[dict], ...]:
    """
    Run local_only and centralized with synthetic data (smoke test).
    Returns (local_A_rows, local_B_rows, local_C_rows, central_rows).
    """
    local_a, local_b, local_c, central = [], [], [], []
    for seed in SEEDS:
        local_a.extend(train_local("A", root=".", n_epochs=n_epochs, lr=1e-3,
                                   batch_size=batch_size, seed=seed,
                                   use_synthetic=True))
        local_b.extend(train_local("B", root=".", n_epochs=n_epochs, lr=1e-3,
                                   batch_size=batch_size, seed=seed,
                                   use_synthetic=True))
        local_c.extend(train_local("C", root=".", n_epochs=n_epochs, lr=1e-3,
                                   batch_size=batch_size, seed=seed,
                                   use_synthetic=True))
        central.extend(train_centralized(root=".", n_epochs=n_epochs, lr=1e-3,
                                         batch_size=batch_size, seed=seed,
                                         use_synthetic=True))
    return local_a, local_b, local_c, central


def _rows_to_final_per_seed(rows: list[dict], model_name: str, site: str) -> list[dict]:
    """Convert raw training rows (from train_local/train_centralized) to final-epoch summary."""
    df = pd.DataFrame(rows)
    epoch_col = "epoch"
    out = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values(epoch_col).iloc[-1]
        row = {"model": model_name, "seed": int(seed)}
        if site == "A":
            row["ihm_auroc"]         = float(last.get("val_auc_roc",  float("nan")))
            row["ihm_auprc"]         = float(last.get("val_auc_pr",   float("nan")))
            row["decomp_auroc"]      = float("nan")
            row["decomp_auprc"]      = float("nan")
            row["pheno_macro_auroc"] = float("nan")
        elif site == "B":
            row["ihm_auroc"]         = float("nan")
            row["ihm_auprc"]         = float("nan")
            row["decomp_auroc"]      = float(last.get("val_auc_roc",  float("nan")))
            row["decomp_auprc"]      = float(last.get("val_auc_pr",   float("nan")))
            row["pheno_macro_auroc"] = float("nan")
        elif site == "C":
            row["ihm_auroc"]         = float("nan")
            row["ihm_auprc"]         = float("nan")
            row["decomp_auroc"]      = float("nan")
            row["decomp_auprc"]      = float("nan")
            row["pheno_macro_auroc"] = float(last.get("val_macro_auc", float("nan")))
        elif site == "central":
            row["ihm_auroc"]         = float(last.get("val_ihm_auc_roc",     float("nan")))
            row["ihm_auprc"]         = float(last.get("val_ihm_auc_pr",      float("nan")))
            row["decomp_auroc"]      = float(last.get("val_decomp_auc_roc",  float("nan")))
            row["decomp_auprc"]      = float(last.get("val_decomp_auc_pr",   float("nan")))
            row["pheno_macro_auroc"] = float(last.get("val_pheno_macro_auc", float("nan")))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_rows: list[dict]) -> None:
    """Print mean ± std comparison table to stdout."""
    df = pd.DataFrame(all_rows)
    metrics = ["ihm_auroc", "decomp_auroc", "pheno_macro_auroc"]
    col_widths = [22] + [18] * len(metrics)
    header = f"{'Model':<22}" + "".join(f"{m:>18}" for m in metrics)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for model, grp in df.groupby("model", sort=False):
        vals = []
        for m in metrics:
            col = grp[m].dropna()
            if col.empty:
                vals.append("     —     ")
            else:
                vals.append(f"{col.mean():.4f}±{col.std():.4f}")
        print(f"{model:<22}" + "".join(f"{v:>18}" for v in vals))
    print("─" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Primary results table: VFL-MTL vs. baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",         default=".",      help="Project root")
    p.add_argument("--n_epochs",     type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--use_synthetic",action="store_true",
                   help="Use synthetic data (smoke test, no MIMIC required)")
    p.add_argument("--skip_rerun",   action="store_true",
                   help="Load pre-computed baseline CSVs instead of rerunning")
    p.add_argument("--local_a",      default="results/local_only_A.csv")
    p.add_argument("--local_b",      default="results/local_only_B.csv")
    p.add_argument("--local_c",      default="results/local_only_C.csv")
    p.add_argument("--central",      default="results/centralized.csv")
    p.add_argument("--vfl_results",  default="results/exp1.csv",
                   help="Path to exp1.csv produced by run_exp1.py")
    p.add_argument("--output",       default="results/baselines_comparison.csv")
    args = p.parse_args()

    all_rows: list[dict] = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    if args.skip_rerun:
        print("Loading pre-computed baseline results...")
        all_rows.extend(_extract_local(args.local_a, "A"))
        all_rows.extend(_extract_local(args.local_b, "B"))
        all_rows.extend(_extract_local(args.local_c, "C"))
        all_rows.extend(_extract_centralized(args.central))
    elif args.use_synthetic:
        print("Running baselines with synthetic data (smoke test)...")
        la, lb, lc, cent = _run_baselines_synthetic(args.n_epochs, args.batch_size)
        all_rows.extend(_rows_to_final_per_seed(la,   "local_A",            "A"))
        all_rows.extend(_rows_to_final_per_seed(lb,   "local_B",            "B"))
        all_rows.extend(_rows_to_final_per_seed(lc,   "local_C",            "C"))
        all_rows.extend(_rows_to_final_per_seed(cent, "centralized_oracle", "central"))
    else:
        print("Running baselines with real data...")
        # Build real datasets once; reuse across seeds for efficiency
        la, lb, lc, cent = [], [], [], []
        for site, store in [("A", la), ("B", lb), ("C", lc)]:
            for seed in SEEDS:
                store.extend(train_local(site, root=args.root, n_epochs=args.n_epochs,
                                         lr=1e-3, batch_size=args.batch_size,
                                         seed=seed, use_synthetic=False))
        for seed in SEEDS:
            cent.extend(train_centralized(root=args.root, n_epochs=args.n_epochs,
                                          lr=1e-3, batch_size=args.batch_size,
                                          seed=seed, use_synthetic=False))
        all_rows.extend(_rows_to_final_per_seed(la,   "local_A",            "A"))
        all_rows.extend(_rows_to_final_per_seed(lb,   "local_B",            "B"))
        all_rows.extend(_rows_to_final_per_seed(lc,   "local_C",            "C"))
        all_rows.extend(_rows_to_final_per_seed(cent, "centralized_oracle", "central"))

    # ── VFL-MTL ───────────────────────────────────────────────────────────────
    vfl_path = Path(args.vfl_results)
    if vfl_path.exists():
        print(f"Loading VFL-MTL results from {vfl_path}...")
        all_rows.extend(_extract_vfl_mtl(str(vfl_path)))
    else:
        print(f"[WARNING] {vfl_path} not found — run experiments/run_exp1.py first. "
              "VFL-MTL rows will be absent from the output.")

    # ── Output ────────────────────────────────────────────────────────────────
    _print_summary(all_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "ihm_auroc", "ihm_auprc",
              "decomp_auroc", "decomp_auprc", "pheno_macro_auroc"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nBaselines comparison → {args.output}")


if __name__ == "__main__":
    main()
