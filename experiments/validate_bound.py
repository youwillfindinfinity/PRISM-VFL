"""
experiments/validate_bound.py — Multi-task label inference bound validation.

Derives and validates a multi-task label inference AUC upper bound for
VFL under Gaussian DP noise, extending the single-task Gaussian mechanism
bound (Abadi et al., 2016) to the heterogeneous VFL-MTL setting via task
gradient correlation ρ.

Theoretical bound (novel; derived from Gaussian mechanism + ρ extension):
  g(σ, ρ) = Φ(C · √(1 + ρ) / σ)

  where Φ = standard normal CDF, C = max_grad_norm (clipping bound),
  σ = noise multiplier, ρ = max pairwise task gradient cosine similarity.

  - At σ → ∞: g → 0.5  (heavy noise → chance-level inference)
  - At σ → 0: g → 1.0  (no noise → perfect inference possible)
  - ρ ↑: g ↑            (higher task coupling → inflated bound)

  The bound is on label inference AUC (scale consistent with roc_auc_score).

Output: results/bound_validation.csv
  columns: rho, epsilon_level, sigma, bound_ihm, bound_pheno,
           empirical_ihm_auroc_mean, empirical_pheno_auroc_mean

Usage
-----
  python experiments/validate_bound.py --use_synthetic --n_rounds 3
  python experiments/validate_bound.py --splits_dir data/vertical_splits --n_rounds 50
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

# ρ values to sweep (task gradient correlation proxy)
RHO_VALUES     = [0.1, 0.3, 0.5, 0.7, 0.9]
EPSILON_LEVELS = ["inf", "10.0", "5.0", "2.0", "1.0", "0.5"]
DELTA          = 1e-5
MAX_GRAD_NORM  = 1.0


# ---------------------------------------------------------------------------
# Bound function
# ---------------------------------------------------------------------------

def g_bound(sigma: float | None, rho: float, C: float = MAX_GRAD_NORM) -> float:
    """
    Multi-task label inference AUC upper bound.

    Parameters
    ----------
    sigma : noise multiplier (None = no DP → bound = 1.0)
    rho   : max pairwise task gradient cosine similarity in [0, 1]
    C     : gradient clipping norm (sensitivity)

    Returns
    -------
    float in [0.5, 1.0] — upper bound on label inference AUC
    """
    if sigma is None or not math.isfinite(sigma) or sigma <= 0.0:
        return 1.0
    return float(norm.cdf(C * math.sqrt(1.0 + rho) / sigma))


# ---------------------------------------------------------------------------
# σ lookup (mirrors privacy_utility_curves._compute_sigma)
# ---------------------------------------------------------------------------

def _sigma_for_eps(eps_str: str, sample_rate: float, n_rounds: int) -> float | None:
    if eps_str == "inf":
        return None
    eps = float(eps_str)
    try:
        from opacus.accountants.utils import get_noise_multiplier
        return float(get_noise_multiplier(
            target_epsilon=eps,
            target_delta=DELTA,
            sample_rate=sample_rate,
            epochs=n_rounds,
        ))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Empirical accuracy loader
# ---------------------------------------------------------------------------

def _load_empirical(label_inference_csv: Path) -> dict[tuple[str, str, str], float]:
    """
    Read results/label_inference.csv.

    Returns dict keyed by (epsilon_level, mode, task) → mean auroc across seeds.
    """
    if not label_inference_csv.exists():
        return {}

    import pandas as pd
    df = pd.read_csv(label_inference_csv)

    result: dict[tuple[str, str, str], float] = {}
    for (eps, mode, task), grp in df.groupby(["epsilon_level", "mode", "task"]):
        result[(str(eps), str(mode), str(task))] = float(grp["auroc"].mean())
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--label_inference_csv", default="results/label_inference.csv")
    parser.add_argument("--output",              default="results/bound_validation.csv")
    parser.add_argument("--n_rounds",            type=int, default=100)
    parser.add_argument("--batch_size",          type=int, default=64)
    parser.add_argument("--use_synthetic",       action="store_true")
    parser.add_argument("--n_synthetic",         type=int, default=256)
    parser.add_argument("--splits_dir",          default="data/vertical_splits")
    args = parser.parse_args()

    # Sample rate — must match privacy_utility_curves.py
    if args.use_synthetic:
        n_batches   = max(1, args.n_synthetic // args.batch_size)
        sample_rate = 1.0 / n_batches
    else:
        try:
            from data_prep.dataset import build_site_loaders
            project_root = Path(args.splits_dir).parents[1]
            loaders      = build_site_loaders(project_root, "train", args.batch_size)
            sample_rate  = 1.0 / max(len(loaders["A"]), 1)
        except Exception:
            sample_rate = 1.0 / 100   # fallback: assume ~100 batches
            print(f"[validate_bound] Could not load loaders; using sample_rate={sample_rate}")

    print(f"[validate_bound] sample_rate={sample_rate:.5f}  n_rounds={args.n_rounds}")

    # σ map
    sigma_map: dict[str, float | None] = {
        eps: _sigma_for_eps(eps, sample_rate, args.n_rounds)
        for eps in EPSILON_LEVELS
    }
    for eps, sigma in sigma_map.items():
        s = f"{sigma:.4f}" if sigma is not None else "∞"
        print(f"  ε={eps:>6s} → σ={s}")

    # Empirical accuracy
    empirical = _load_empirical(Path(args.label_inference_csv))
    if not empirical:
        print(f"[validate_bound] WARNING: {args.label_inference_csv} not found or empty — "
              f"empirical columns will be NaN")

    # Build output rows: one row per (rho, epsilon_level)
    all_rows: list[dict] = []

    # Use uniform mode as reference (matches the ε sweep in privacy_utility_curves.py)
    mode_ref = "uniform"

    for rho in RHO_VALUES:
        for eps_str in EPSILON_LEVELS:
            sigma = sigma_map[eps_str]

            bound_ihm   = g_bound(sigma, rho)
            bound_pheno = g_bound(sigma, rho)   # task-agnostic upper bound; empirical points per task overlaid in figures

            emp_ihm   = empirical.get((eps_str, mode_ref, "ihm"),   float("nan"))
            emp_pheno = empirical.get((eps_str, mode_ref, "pheno"), float("nan"))

            all_rows.append({
                "rho":                       rho,
                "epsilon_level":             eps_str,
                "sigma":                     sigma if sigma is not None else float("inf"),
                "bound_ihm":                 round(bound_ihm,   6),
                "bound_pheno":               round(bound_pheno, 6),
                "empirical_ihm_auroc_mean":   round(emp_ihm,    6),
                "empirical_pheno_auroc_mean": round(emp_pheno,  6),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[validate_bound] Done. {len(all_rows)} rows → {out}")

    # Brief summary: bound at ρ=0.5 for each ε level
    print("\n── Bound g(σ, ρ=0.5) across ε levels ──")
    print(f"{'ε':>6s}  {'σ':>8s}  {'bound':>8s}")
    for eps_str in EPSILON_LEVELS:
        sigma  = sigma_map[eps_str]
        b      = g_bound(sigma, rho=0.5)
        s_str  = f"{sigma:.4f}" if sigma is not None else "  no DP"
        print(f"{eps_str:>6s}  {s_str:>8s}  {b:.4f}")


if __name__ == "__main__":
    main()
