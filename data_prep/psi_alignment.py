"""
data_prep/psi_alignment.py — Simulated Private Set Intersection (PSI) for patient alignment.

Simulates a privacy-preserving PSI protocol:
  1. Each site hashes its patient IDs with SHA-256 (+ shared salt).
  2. The server computes the intersection of hashed ID sets.
  3. Returns the aligned patient index (original IDs) for training.

In production VFL the raw IDs never leave each site — only hashed tokens are
compared. This module simulates that protocol locally for benchmarking purposes.

Usage (module):
    from data_prep.psi_alignment import compute_psi_alignment, write_aligned_ids

Usage (CLI):
    python data_prep/psi_alignment.py \\
        --site_a data/vertical_splits/site_A_vitals.csv \\
        --site_b data/vertical_splits/site_B_labs.csv \\
        --site_c data/vertical_splits/site_C_composite.csv \\
        --output data/vertical_splits/aligned_patient_ids.csv \\
        [--salt my_shared_salt]
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

DEFAULT_SALT = "vfl_mtl_mimic3"


# ── Hashing ────────────────────────────────────────────────────────────────

def hash_id(patient_id: int, salt: str = DEFAULT_SALT) -> str:
    """SHA-256 hash of <salt>||<patient_id>."""
    token = f"{salt}:{patient_id}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def hash_id_set(patient_ids: Set[int], salt: str = DEFAULT_SALT) -> Dict[str, int]:
    """Return {hash → original_id} mapping for a set of patient IDs."""
    return {hash_id(pid, salt): pid for pid in patient_ids}


# ── PSI computation ────────────────────────────────────────────────────────

def compute_psi_alignment(
    site_dfs: list[pd.DataFrame],
    splits: list[str] = ("train", "val", "test"),
    salt: str = DEFAULT_SALT,
    id_col: str = "subject_id",
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Simulate PSI across N site DataFrames and return aligned patient index.

    Parameters
    ----------
    site_dfs  : list of DataFrames, each with at least `id_col` and `split_col`
    splits    : which split labels to process
    salt      : shared salt for SHA-256 hashing
    id_col    : name of the patient ID column
    split_col : name of the split column

    Returns
    -------
    DataFrame with columns [id_col, split_col] — one row per aligned patient per split
    """
    rows = []

    for split in splits:
        # Step 1 — each site hashes its patient IDs (only hashes are "shared")
        hashed_sets: list[Dict[str, int]] = []
        for df in site_dfs:
            subset = df.loc[df[split_col] == split, id_col]
            hashed_sets.append(hash_id_set(set(subset), salt))

        # Step 2 — server computes intersection of hashed token sets
        common_hashes: Set[str] = set(hashed_sets[0].keys())
        for hs in hashed_sets[1:]:
            common_hashes &= set(hs.keys())

        # Step 3 — recover original IDs from any site's reverse map (all identical for matching IDs)
        aligned_ids = sorted(hashed_sets[0][h] for h in common_hashes)

        for pid in aligned_ids:
            rows.append({id_col: pid, split_col: split})

        print(f"  PSI [{split}]: {len(aligned_ids):,} aligned patients "
              f"({', '.join(f'site{i+1}={len(hs):,}' for i, hs in enumerate(hashed_sets))})")

    return pd.DataFrame(rows)


def write_aligned_ids(aligned: pd.DataFrame, output_path: Path) -> None:
    """Write the aligned patient index to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    print(f"  → {output_path}  ({len(aligned):,} rows)")


# ── Label balance check & stratified re-assignment ─────────────────────────

def check_label_balance(
    aligned_ids: pd.DataFrame,
    site_a: pd.DataFrame,
    site_b: pd.DataFrame,
    site_c: pd.DataFrame,
    pheno_cols: list[str],
    tol: float = 0.03,
) -> bool:
    """
    Returns True if IHM rate, Decompensation label balance, and per-phenotype prevalence
    are all within `tol` of the train distribution in val and test splits.
    """
    a_labels = site_a[["subject_id", "y_ihm"]].drop_duplicates(subset="subject_id")
    b_labels = site_b[["subject_id", "y_decomp"]].drop_duplicates(subset="subject_id")
    c_labels = site_c[["subject_id"] + pheno_cols].drop_duplicates(subset="subject_id")

    merged = (
        aligned_ids
        .merge(a_labels, on="subject_id", how="left")
        .merge(b_labels, on="subject_id", how="left")
        .merge(c_labels, on="subject_id", how="left")
    )

    train_ihm = merged.loc[merged["split"] == "train", "y_ihm"].mean()
    for split in ["val", "test"]:
        split_ihm = merged.loc[merged["split"] == split, "y_ihm"].mean()
        if abs(split_ihm - train_ihm) > tol:
            print(f"  IHM imbalance in {split}: train={train_ihm:.3f}, {split}={split_ihm:.3f}")
            return False

    for col in pheno_cols:
        train_prev = merged.loc[merged["split"] == "train", col].mean()
        for split in ["val", "test"]:
            split_prev = merged.loc[merged["split"] == split, col].mean()
            if abs(split_prev - train_prev) > tol:
                print(f"  Phenotype '{col}' imbalance in {split}: "
                      f"train={train_prev:.3f}, {split}={split_prev:.3f}")
                return False

    return True


def stratify_aligned_cohort(
    aligned_ids: pd.DataFrame,
    site_a: pd.DataFrame,
    site_b: pd.DataFrame,
    site_c: pd.DataFrame,
    pheno_cols: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Re-assigns train/val/test within the aligned cohort using iterative stratification
    across all label signals: IHM (binary), Decompensation (binary), phenotypes (25 binary).
    Returns updated aligned_ids DataFrame with new 'split' column.
    """
    from skmultilearn.model_selection import IterativeStratification

    # Deduplicate per subject_id — site CSVs have one row per stay (multiple stays
    # per subject are possible). Take first occurrence for stratification purposes.
    a_labels = site_a[["subject_id", "y_ihm"]].drop_duplicates(subset="subject_id")
    b_labels = site_b[["subject_id", "y_decomp"]].drop_duplicates(subset="subject_id")
    c_labels = site_c[["subject_id"] + pheno_cols].drop_duplicates(subset="subject_id")

    merged = (
        aligned_ids[["subject_id"]]
        .merge(a_labels, on="subject_id", how="left")
        .merge(b_labels, on="subject_id", how="left")
        .merge(c_labels, on="subject_id", how="left")
    )

    # Combined label matrix: y_ihm | y_decomp (binary) | 25 phenotypes
    label_matrix = pd.concat(
        [merged[["y_ihm", "y_decomp"]], merged[pheno_cols]], axis=1
    ).fillna(0).astype(float).to_numpy()

    ids = merged["subject_id"].to_numpy()

    # First split: (train+val) vs test
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[test_frac, 1 - test_frac],
    )
    # sklearn convention: split() yields (train, test); fold 0 is test (smaller, test_frac)
    trainval_idx, test_idx = next(stratifier.split(ids.reshape(-1, 1), label_matrix))

    # Second split: train vs val within (train+val)
    val_frac_of_trainval = val_frac / (1 - test_frac)
    stratifier2 = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[val_frac_of_trainval, 1 - val_frac_of_trainval],
    )
    train_idx, val_idx = next(
        stratifier2.split(ids[trainval_idx].reshape(-1, 1), label_matrix[trainval_idx])
    )
    val_idx_global   = trainval_idx[val_idx]
    train_idx_global = trainval_idx[train_idx]

    split_col = np.empty(len(ids), dtype=object)
    split_col[train_idx_global] = "train"
    split_col[val_idx_global]   = "val"
    split_col[test_idx]         = "test"

    print(f"  Stratified split: train={( split_col=='train').sum():,}, "
          f"val={(split_col=='val').sum():,}, test={(split_col=='test').sum():,}")
    return pd.DataFrame({"subject_id": ids, "split": split_col})


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--site_a", required=True,
                        help="Path to site_A_vitals.csv")
    parser.add_argument("--site_b", required=True,
                        help="Path to site_B_labs.csv")
    parser.add_argument("--site_c", required=True,
                        help="Path to site_C_composite.csv")
    parser.add_argument("--output", required=True,
                        help="Output path for aligned_patient_ids.csv")
    parser.add_argument("--salt", default=DEFAULT_SALT,
                        help=f"Shared salt for SHA-256 hashing (default: '{DEFAULT_SALT}')")
    args = parser.parse_args()

    paths = {"site_a": args.site_a, "site_b": args.site_b, "site_c": args.site_c}
    site_dfs = {}
    for key, path in paths.items():
        p = Path(path)
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        site_dfs[key] = pd.read_csv(p)

    print("Running PSI alignment ...")
    aligned = compute_psi_alignment(
        [site_dfs["site_a"], site_dfs["site_b"], site_dfs["site_c"]],
        salt=args.salt,
    )

    # Detect phenotype label columns from site_C (everything except metadata + features)
    non_pheno = {"stay", "subject_id", "split", "Height", "Weight", "Mean blood pressure"}
    pheno_cols = [c for c in site_dfs["site_c"].columns if c not in non_pheno]

    print("Checking label balance across splits ...")
    balanced = check_label_balance(
        aligned, site_dfs["site_a"], site_dfs["site_b"], site_dfs["site_c"], pheno_cols
    )
    if not balanced:
        # Stratification would reassign subjects to splits inconsistent with the site CSVs,
        # causing near-zero overlap in VFLSiteDataset (which filters by split in BOTH files).
        # YerevaNN's original splits are already reasonable; minor imbalance is acceptable.
        print("  WARNING: label imbalance exceeds tolerance but splits are NOT re-assigned.")
        print("  Stratification would break site CSV ↔ aligned_patient_ids.csv split consistency.")
        print("  Keeping inherited YerevaNN splits.")
    else:
        print("  Label balance OK — keeping inherited YerevaNN splits.")

    write_aligned_ids(aligned, Path(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
