"""
revision_scripts/analyse_prism_privacy_sweep.py — statistics over
results_revision/prism_privacy_sweep.csv (pure pandas/scipy/numpy, no GPU).

Computes, per task (ihm/decomp/pheno):
  - per-seed best-round mean, SD, bootstrap 95% CI
  - paired PRISM(equal, eps=inf) - matching ST baseline, per seed: Wilcoxon + sign test
  - reproducible/directional verdict (CI excludes 0 AND sign p<0.05 AND >=8/10 seeds agree)
  - negative transfer rate (fraction of seeds where PRISM < ST)
  - DP utility loss: paired PRISM(equal, eps) - PRISM(equal, inf), per seed
  - PCMU via compute_pcmu.evaluate_sweep() (compute_pcmu.py itself untouched)

Usage:
    python revision_scripts/analyse_prism_privacy_sweep.py
    python revision_scripts/analyse_prism_privacy_sweep.py \
        --input smoke_tests/revision1/sweep.csv --output smoke_tests/revision1/stats.csv \
        --summary_md smoke_tests/revision1/summary.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.compute_pcmu import evaluate_sweep

TASKS = {"ihm": "val_ihm_auroc", "decomp": "val_decomp_auroc", "pheno": "val_pheno_macro_auroc"}
ST_NAME = {"ihm": "ST-IHM", "decomp": "ST-Decomp", "pheno": "ST-Pheno"}
DP_EPS_LEVELS = [0.5, 1.0, 2.0, 5.0, 10.0]


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_seed_best(df: pd.DataFrame, task_col: str) -> pd.DataFrame:
    """One row per (arm, weighting, epsilon_level, seed): task_col at that run's best_round."""
    sel = df[df["round"] == df["best_round"]]
    return sel[["arm", "weighting", "epsilon_level", "seed", task_col]].rename(columns={task_col: "value"})


def summary_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for task, col in TASKS.items():
        best = per_seed_best(df, col)
        for (arm, weighting, eps), g in best.groupby(["arm", "weighting", "epsilon_level"]):
            vals = g["value"].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals)
            rows.append(dict(
                metric="summary", task=task, arm=arm, weighting=weighting, epsilon_level=eps,
                mean=float(np.nanmean(vals)),
                sd=float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                ci_lo=lo, ci_hi=hi, n_seeds=len(vals),
            ))
    return rows


def paired_rows(df: pd.DataFrame) -> list[dict]:
    """PRISM(equal, eps=inf) - matching ST baseline, per seed, per task."""
    rows = []
    prism_inf = df[(df.arm == "PRISM") & (df.weighting == "equal") & (df.epsilon_level == float("inf"))]
    for task, col in TASKS.items():
        prism = per_seed_best(prism_inf, col)
        st = per_seed_best(df[df.arm == ST_NAME[task]], col)
        merged = prism.merge(st, on="seed", suffixes=("_prism", "_st"))
        deltas = (merged["value_prism"] - merged["value_st"]).to_numpy(dtype=float)
        if len(deltas) == 0:
            continue
        lo, hi = bootstrap_ci(deltas)
        nonzero = deltas[deltas != 0]
        w_p = float(wilcoxon(nonzero).pvalue) if len(nonzero) > 0 else float("nan")
        n_pos = int(np.sum(deltas > 0))
        sign_p = float(binomtest(n_pos, len(deltas), p=0.5).pvalue) if len(deltas) > 0 else float("nan")
        agree = max(n_pos, len(deltas) - n_pos)
        reproducible = bool((lo > 0 or hi < 0) and w_p < 0.05 and sign_p < 0.05 and agree >= 8)
        rows.append(dict(
            metric="paired_vs_st", task=task, arm="PRISM", weighting="equal", epsilon_level=float("inf"),
            mean=float(np.mean(deltas)), sd=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
            ci_lo=lo, ci_hi=hi, n_seeds=len(deltas), wilcoxon_p=w_p, sign_p=sign_p,
            n_positive=n_pos, reproducible=reproducible,
            negative_transfer_rate=float(np.mean(deltas < 0)),
        ))
    return rows


def dp_utility_loss_rows(df: pd.DataFrame) -> list[dict]:
    """PRISM(equal, eps) - PRISM(equal, eps=inf), per seed, per task, per finite eps level."""
    rows = []
    prism_equal = df[(df.arm == "PRISM") & (df.weighting == "equal")]
    for task, col in TASKS.items():
        nodp = per_seed_best(prism_equal[prism_equal.epsilon_level == float("inf")], col)
        for eps in DP_EPS_LEVELS:
            dp = per_seed_best(prism_equal[prism_equal.epsilon_level == eps], col)
            if dp.empty:
                continue
            merged = dp.merge(nodp, on="seed", suffixes=("_dp", "_nodp"))
            deltas = (merged["value_dp"] - merged["value_nodp"]).to_numpy(dtype=float)
            if len(deltas) == 0:
                continue
            lo, hi = bootstrap_ci(deltas)
            rows.append(dict(
                metric="dp_utility_loss", task=task, arm="PRISM", weighting="equal", epsilon_level=eps,
                mean=float(np.mean(deltas)), sd=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
                ci_lo=lo, ci_hi=hi, n_seeds=len(deltas),
            ))
    return rows


def pcmu_rows(df: pd.DataFrame) -> list[dict]:
    """PCMU via compute_pcmu.evaluate_sweep(); adapts our schema (arm/weighting) into
    the (model/mode) schema that function expects, without touching compute_pcmu.py."""
    privacy_df = df[(df.arm == "PRISM") & (df.weighting == "equal")].copy()
    if privacy_df.empty:
        return []
    privacy_df["mode"] = "uniform"
    exp1_df = df[df.arm.isin(ST_NAME.values())].rename(columns={"arm": "model"})
    pcmu_df = evaluate_sweep(privacy_df, exp1_df, mode="uniform")
    rows = []
    for eps, g in pcmu_df.groupby("epsilon_level"):
        rows.append(dict(
            metric="pcmu", task="all", arm="PRISM", weighting="equal", epsilon_level=eps,
            mean=float(g["pcmu"].mean()),
            sd=float(g["pcmu"].std(ddof=1)) if len(g) > 1 else float("nan"),
            n_seeds=len(g),
        ))
    return rows


def write_summary_md(path: Path, paired: list[dict]) -> None:
    lines = ["# PRISM privacy-utility sweep — results summary", ""]
    for r in paired:
        verdict = "reproducible" if r["reproducible"] else "directional"
        lines.append(
            f"- **{r['task']}**: PRISM - ST Δ = {r['mean']:.4f} "
            f"(95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}], "
            f"sign-test p={r['sign_p']:.4f}, {r['n_positive']}/{r['n_seeds']} seeds positive) "
            f"-> **{verdict}**. Negative transfer rate: {r['negative_transfer_rate']:.0%}."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results_revision/prism_privacy_sweep.csv")
    parser.add_argument("--output", default="results_revision/prism_privacy_sweep_stats.csv")
    parser.add_argument("--summary_md", default="revision_docs/prism_privacy_sweep_summary.md")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert df["epsilon_level"].dtype != object, (
        "epsilon_level did not parse as numeric — check the input CSV's epsilon_level column."
    )

    rows = summary_rows(df) + paired_rows(df) + dp_utility_loss_rows(df) + pcmu_rows(df)
    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[analyse] {len(out)} rows -> {out_path}")

    paired = [r for r in rows if r["metric"] == "paired_vs_st"]
    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_md(md_path, paired)
    print(f"[analyse] summary -> {md_path}")


if __name__ == "__main__":
    main()
