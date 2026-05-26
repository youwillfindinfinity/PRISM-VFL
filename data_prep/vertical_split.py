#!/usr/bin/env python3
"""
data_prep/vertical_split.py

Splits MIMIC-III YerevaNN benchmark data into three vertical (feature) partitions
representing three simulated hospital sites. See README.md for the full
vertical split protocol and feature assignment rationale.

    Site A (7 vitals)   : HR, SBP, DBP, Temp, SpO2, RespRate, GCS total
                          Task label: in-hospital mortality (binary)

    Site B (4 labs)     : Glucose, pH, FiO2, CapRefill
                          Task label: decompensation (binary, 24h window)

    Site C (3 composite): Height, Weight, MeanBP
                          Task label: phenotyping (25 binary ICD codes)

14 of the 17 YerevaNN "ready" features are used. The three GCS sub-scores
(eye opening, motor response, verbal response) are excluded: retaining them
alongside GCS total would allow cross-site reconstruction (total = eye + motor
+ verbal), violating VFL's feature-privacy guarantee. GCS total in Site A
captures the same clinical signal used in standard severity scores (APACHE, SOFA).
pCO2, pO2, and Bilirubin from the original protocol are absent from the benchmark
(STATUS="verify" in itemid_to_variable_map.csv; never extracted).

Processing per stay:
  1. Load timeseries CSV.
  2. Clip each feature to VALID_LOW/VALID_HIGH from YerevaNN variable_ranges.csv.
     Note: extract_episodes_from_subjects.py accepts --reference_range_file but never
     calls remove_outliers_for_variable(), so artefacts survive to the timeseries CSVs
     (e.g. SpO2=29818%, Weight=3761721 kg). Clipping here is the fix until the raw
     data is regenerated with the upstream wired up.
  3. Forward-fill then backward-fill within-stay missing values.
  4. Aggregate: compute column mean over all time steps.
  5. Mean-impute any feature still NaN (never observed in this stay)
     using the training-set column mean.

Output (written to --output directory):
  site_A_vitals.csv      — stay, subject_id, split, [7 features], y_ihm
  site_B_labs.csv        — stay, subject_id, split, [4 features], y_decomp
  site_C_composite.csv   — stay, subject_id, split, [3 features], [25 pheno labels]

Note: aligned_patient_ids.csv is produced by a separate step:
  python data_prep/psi_alignment.py --site_a ... --site_b ... --site_c ... --output ...

Usage:
    python data_prep/vertical_split.py \\
        --root  data/mimic3-benchmarks/data/ \\
        --output data/vertical_splits/
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# YerevaNN benchmark path — used to load variable_ranges.csv
_BENCHMARK_RESOURCES = Path(__file__).parent.parent / "mimic3-benchmarks" / "mimic3benchmark" / "resources"
_VARIABLE_RANGES_FILE = _BENCHMARK_RESOURCES / "variable_ranges.csv"

# ---------------------------------------------------------------------------
# Feature definitions — exact column names used in YerevaNN timeseries CSVs
# ---------------------------------------------------------------------------

SITE_A_FEATURES = [
    "Heart Rate",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Temperature",
    "Oxygen saturation",
    "Respiratory rate",
    "Glascow coma scale total",           # GCS summary score
]

SITE_B_FEATURES = [
    "Glucose",
    "pH",
    "Fraction inspired oxygen",
    "Capillary refill rate",
]

SITE_C_FEATURES = [
    "Height",
    "Weight",
    "Mean blood pressure",
]

# Sanity-check: no overlap, 14 features total (17 ready - 3 GCS sub-scores excluded)
assert len(SITE_A_FEATURES) == 7
assert len(SITE_B_FEATURES) == 4
assert len(SITE_C_FEATURES) == 3
assert len(set(SITE_A_FEATURES) & set(SITE_B_FEATURES)) == 0
assert len(set(SITE_A_FEATURES) & set(SITE_C_FEATURES)) == 0
assert len(set(SITE_B_FEATURES) & set(SITE_C_FEATURES)) == 0

TASK_DIRS = {
    "ihm":   "in-hospital-mortality",
    "decomp": "decompensation",
    "pheno": "phenotyping",
}

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Clipping — load VALID_LOW / VALID_HIGH from YerevaNN variable_ranges.csv
# ---------------------------------------------------------------------------

def load_clip_bounds(ranges_file: Path = _VARIABLE_RANGES_FILE) -> dict[str, tuple[float, float]]:
    """
    Read VALID_LOW and VALID_HIGH for every feature that has both values defined.
    Returns {feature_name: (lo, hi)}.
    """
    df = pd.read_csv(ranges_file)
    df = df.rename(columns={"LEVEL2": "VARIABLE", "VALID LOW": "VALID_LOW", "VALID HIGH": "VALID_HIGH"})
    df = df[["VARIABLE", "VALID_LOW", "VALID_HIGH"]].dropna(subset=["VALID_LOW", "VALID_HIGH"])
    return {row["VARIABLE"]: (row["VALID_LOW"], row["VALID_HIGH"]) for _, row in df.iterrows()}


def clip_features(ts_feat: pd.DataFrame, feature_cols: list,
                  clip_bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip each feature column to its VALID_LOW/VALID_HIGH range."""
    for col in feature_cols:
        if col in clip_bounds:
            lo, hi = clip_bounds[col]
            ts_feat[col] = ts_feat[col].clip(lower=lo, upper=hi)
    return ts_feat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_subject_id(stay_filename: str) -> int:
    """Extract integer subject_id from '22094_episode1_timeseries.csv'."""
    return int(stay_filename.split("_")[0])


def ts_path(task_dir: Path, split: str, stay_filename: str) -> Path:
    """
    Resolve the timeseries CSV path for a stay.
    Val stays live in train/ (val_listfile.csv references train-partition files).
    """
    subdir = "test" if split == "test" else "train"
    return task_dir / subdir / stay_filename


def aggregate_stays(task_dir: Path, stays: pd.Series, split: str,
                    feature_cols: list, train_means: pd.Series = None,
                    clip_bounds: dict = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load each stay's timeseries, forward/backward-fill, compute column mean.

    Parameters
    ----------
    task_dir    : path to the task directory (contains train/ and test/)
    stays       : Series of stay filenames (one per unique stay)
    split       : 'train', 'val', or 'test'
    feature_cols: list of feature column names to extract
    train_means : pd.Series used for mean-imputation of fully-missing features;
                  if None the returned DataFrame may contain NaN
    verbose     : print a dot every 1000 stays

    Returns
    -------
    DataFrame with columns: stay, subject_id, <feature_cols>
    """
    rows = []
    missing = 0

    for i, stay in enumerate(stays):
        if verbose and i > 0 and i % 1000 == 0:
            print(f"    {i}/{len(stays)} stays processed ...", flush=True)

        path = ts_path(task_dir, split, stay)
        if not path.exists():
            missing += 1
            continue

        ts = pd.read_csv(path)

        # Select features; add all-NaN column for any feature not in this file
        ts_feat = pd.DataFrame(index=ts.index)
        for col in feature_cols:
            ts_feat[col] = ts[col] if col in ts.columns else np.nan

        # Clip to VALID_LOW/VALID_HIGH before imputation so artefact values
        # do not propagate via ffill/bfill or contaminate the stay mean
        if clip_bounds:
            ts_feat = clip_features(ts_feat, feature_cols, clip_bounds)

        # Within-stay imputation: forward-fill then backward-fill
        ts_feat = ts_feat.ffill().bfill()

        # Aggregate: mean over time steps
        feat_means = ts_feat.mean()

        entry = {"stay": stay, "subject_id": parse_subject_id(stay)}
        for col in feature_cols:
            entry[col] = feat_means.get(col, np.nan)
        rows.append(entry)

    if missing:
        print(f"    Warning: {missing}/{len(stays)} timeseries files not found (skipped).")

    df = pd.DataFrame(rows)

    # Mean-impute features that were never observed in this stay
    if train_means is not None:
        for col in feature_cols:
            df[col] = df[col].fillna(train_means.get(col, 0.0))

    return df


# ---------------------------------------------------------------------------
# Per-site builders
# ---------------------------------------------------------------------------

def build_site_a(root: Path, output: Path, clip_bounds: dict = None) -> pd.DataFrame:
    """
    Build site_A_vitals.csv from in-hospital-mortality listfiles.
    Label: y_ihm (binary 0/1, 48-hour in-hospital mortality).
    """
    task_dir = root / TASK_DIRS["ihm"]
    print("Building Site A (vitals → IHM) ...")

    all_frames = []
    train_means = None

    for split in SPLITS:
        print(f"  [{split}]")
        lf = pd.read_csv(task_dir / f"{split}_listfile.csv")
        unique_stays = lf["stay"].drop_duplicates()

        agg = aggregate_stays(task_dir, unique_stays, split, SITE_A_FEATURES,
                              train_means=train_means, clip_bounds=clip_bounds)
        if split == "train":
            train_means = agg[SITE_A_FEATURES].mean()
            for col in SITE_A_FEATURES:
                agg[col] = agg[col].fillna(train_means.get(col, 0.0))

        # Attach label: one label per unique stay
        lf_dedup = lf.drop_duplicates("stay").set_index("stay")
        agg["y_ihm"] = lf_dedup["y_true"].reindex(agg["stay"].values).values
        agg["split"] = split
        all_frames.append(agg)

    out = pd.concat(all_frames, ignore_index=True)
    out_path = output / "site_A_vitals.csv"
    out.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(out)} rows, {out_path.stat().st_size // 1024} KB)")
    return out


def build_site_b(root: Path, output: Path, clip_bounds: dict = None) -> pd.DataFrame:
    """
    Build site_B_labs.csv from decompensation listfiles.

    Decompensation listfiles have multiple rows per stay (one per hourly prediction
    window). Label: y_decomp = max(y_true) across all hourly rows for the stay
    (1 if the patient ever faced an imminent decompensation event, 0 otherwise).
    """
    task_dir = root / TASK_DIRS["decomp"]
    print("Building Site B (labs → Decompensation) ...")

    all_frames = []
    train_means = None

    for split in SPLITS:
        print(f"  [{split}]")
        lf = pd.read_csv(task_dir / f"{split}_listfile.csv")

        # Per-stay decompensation label: 1 if any hourly y_true == 1, else 0
        lf_dedup = lf.groupby("stay")["y_true"].max().rename("y_decomp")
        unique_stays = pd.Series(lf_dedup.index.tolist())

        agg = aggregate_stays(task_dir, unique_stays, split, SITE_B_FEATURES,
                              train_means=train_means, clip_bounds=clip_bounds)
        if split == "train":
            train_means = agg[SITE_B_FEATURES].mean()
            for col in SITE_B_FEATURES:
                agg[col] = agg[col].fillna(train_means.get(col, 0.0))

        agg["y_decomp"] = lf_dedup.reindex(agg["stay"].values).values
        agg["split"] = split
        all_frames.append(agg)

    out = pd.concat(all_frames, ignore_index=True)
    out_path = output / "site_B_labs.csv"
    out.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(out)} rows, {out_path.stat().st_size // 1024} KB)")
    return out


def build_site_c(root: Path, output: Path, clip_bounds: dict = None) -> pd.DataFrame:
    """
    Build site_C_composite.csv from phenotyping listfiles.
    Labels: 25 binary ICD phenotype columns.
    """
    task_dir = root / TASK_DIRS["pheno"]
    print("Building Site C (composite → Phenotyping) ...")

    # Detect phenotype label columns from train listfile header
    lf_sample = pd.read_csv(task_dir / "train_listfile.csv", nrows=0)
    label_cols = [c for c in lf_sample.columns if c not in ("stay", "period_length")]

    all_frames = []
    train_means = None

    for split in SPLITS:
        print(f"  [{split}]")
        lf = pd.read_csv(task_dir / f"{split}_listfile.csv")
        lf_dedup = lf.drop_duplicates("stay").set_index("stay")
        unique_stays = pd.Series(lf_dedup.index.tolist())

        agg = aggregate_stays(task_dir, unique_stays, split, SITE_C_FEATURES,
                              train_means=train_means, clip_bounds=clip_bounds)
        if split == "train":
            train_means = agg[SITE_C_FEATURES].mean()
            for col in SITE_C_FEATURES:
                agg[col] = agg[col].fillna(train_means.get(col, 0.0))

        for lc in label_cols:
            agg[lc] = lf_dedup[lc].reindex(agg["stay"].values).values
        agg["split"] = split
        all_frames.append(agg)

    out = pd.concat(all_frames, ignore_index=True)
    out_path = output / "site_C_composite.csv"
    out.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(out)} rows, {out_path.stat().st_size // 1024} KB)")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", required=True,
        help="Path to mimic3-benchmarks/data/ (contains in-hospital-mortality/, etc.)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for vertical split CSVs",
    )
    args = parser.parse_args()

    root   = Path(args.root).resolve()
    output = Path(args.output).resolve()

    if not root.exists():
        print(f"Error: --root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # Verify expected task directories
    for task_key, task_name in TASK_DIRS.items():
        td = root / task_name
        if not td.exists():
            print(f"Error: task directory not found: {td}", file=sys.stderr)
            sys.exit(1)

    output.mkdir(parents=True, exist_ok=True)

    clip_bounds = load_clip_bounds()
    print(f"Loaded clip bounds for {len(clip_bounds)} features from variable_ranges.csv")

    build_site_a(root, output, clip_bounds=clip_bounds)
    build_site_b(root, output, clip_bounds=clip_bounds)
    build_site_c(root, output, clip_bounds=clip_bounds)

    print("\nVertical split complete.")
    print(f"Output directory: {output}")
    print("Next step: run psi_alignment.py to produce aligned_patient_ids.csv")


if __name__ == "__main__":
    main()
