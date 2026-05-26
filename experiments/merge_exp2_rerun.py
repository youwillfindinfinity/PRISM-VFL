"""
experiments/merge_exp2_rerun.py — Merge corrected ihm_decomp rows into exp2.csv.

The server.py uncertainty_weighting bug corrupted only the ihm_decomp config in exp2.csv
(zero-weight tasks were still trained). All other configs are clean.

This script:
  1. Loads results/exp2.csv — drops all ihm_decomp rows
  2. Loads results/exp2_ihm_decomp.csv — the fixed rerun
  3. Concatenates and writes back to results/exp2.csv

Run after run_exp3_ihm_decomp.sh completes and results are synced locally.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent / "results"
EXP3_PATH       = ROOT / "exp2.csv"
IHM_DECOMP_PATH = ROOT / "exp2_ihm_decomp.csv"

def main():
    exp3 = pd.read_csv(EXP3_PATH)
    print(f"Loaded exp2.csv: {len(exp3)} rows")
    print(f"  configs present: {sorted(exp3['task_config'].unique())}")

    ihm_decomp_rows = pd.read_csv(IHM_DECOMP_PATH)
    print(f"Loaded exp2_ihm_decomp.csv: {len(ihm_decomp_rows)} rows")

    clean = exp3[exp3["task_config"] != "ihm_decomp"]
    print(f"Kept {len(clean)} clean rows (dropped {len(exp3) - len(clean)} ihm_decomp rows)")

    merged = pd.concat([clean, ihm_decomp_rows], ignore_index=True)

    # Sort to match original order: config order × seed × round
    config_order = ["all_tasks", "ihm_only", "ihm_decomp", "ihm_pheno"]
    merged["_config_rank"] = merged["task_config"].map({c: i for i, c in enumerate(config_order)})
    merged = merged.sort_values(["_config_rank", "seed", "round"]).drop(columns="_config_rank")
    merged = merged.reset_index(drop=True)

    merged.to_csv(EXP3_PATH, index=False)
    print(f"\nMerged exp2.csv written: {len(merged)} rows")
    print(f"  configs present: {sorted(merged['task_config'].unique())}")

if __name__ == "__main__":
    main()
