"""
experiments/merge_exp3_rerun.py — Merge corrected n_sites=2 rows into exp3.csv.

The server.py uncertainty_weighting bug corrupted only the n_sites=2 config in exp3.csv
(pheno task had weight=0 but was still trained). n_sites=3 rows are clean.

This script:
  1. Loads results/exp3.csv — drops all n_sites=2 rows
  2. Loads results/exp3_n_sites_2.csv — the fixed rerun
  3. Concatenates and writes back to results/exp3.csv

Run after run_exp4_n_sites_2.sh completes and results are synced locally.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent / "results"
EXP4_PATH      = ROOT / "exp3.csv"
N_SITES_2_PATH = ROOT / "exp3_n_sites_2.csv"

def main():
    exp4 = pd.read_csv(EXP4_PATH)
    print(f"Loaded exp3.csv: {len(exp4)} rows")
    print(f"  n_sites present: {sorted(exp4['n_sites'].unique())}")

    n_sites_2_rows = pd.read_csv(N_SITES_2_PATH)
    print(f"Loaded exp3_n_sites_2.csv: {len(n_sites_2_rows)} rows")

    clean = exp4[exp4["n_sites"] != 2]
    print(f"Kept {len(clean)} clean rows (dropped {len(exp4) - len(clean)} n_sites=2 rows)")

    merged = pd.concat([n_sites_2_rows, clean], ignore_index=True)
    merged = merged.sort_values(["n_sites", "seed", "round"]).reset_index(drop=True)

    merged.to_csv(EXP4_PATH, index=False)
    print(f"\nMerged exp3.csv written: {len(merged)} rows")
    print(f"  n_sites present: {sorted(merged['n_sites'].unique())}")

if __name__ == "__main__":
    main()
