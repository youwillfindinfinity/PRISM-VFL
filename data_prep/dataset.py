"""
data/dataset.py — VFL-MTL PyTorch Dataset for MIMIC-III vertical splits.

Each VFLSiteDataset represents one hospital site's view:
  - Site A (7 vitals)    → binary IHM label
  - Site B (4 labs)      → Decompensation binary label (0/1)
  - Site C (3 composite) → multi-label phenotyping (25 ICD codes)

The site CSVs (site_A_vitals.csv etc.) act as an index:
  stay       → filename of raw timeseries (loaded on-the-fly)
  subject_id → used to filter to PSI-aligned patients
  split      → train / val / test assignment
  label cols → training targets

Raw timeseries are loaded from the YerevaNN task directory
(in-hospital-mortality/, length-of-stay/, or phenotyping/) and discretized
into 1-hour bins matching how YerevaNN's LSTM baseline preprocesses data.
The raw files have irregular timestamps (e.g. 0.38, 1.38, 5.25, 5.38 ...)
and must be binned at runtime — the YerevaNN pipeline does not pre-discretize.

Usage:
    from data.dataset import VFLSiteDataset, collate_fn, build_site_loaders
    from data.dataset import SITE_A_FEATURES, SITE_B_FEATURES, SITE_C_FEATURES

    ds = VFLSiteDataset(
        site_csv        = "data/vertical_splits/site_A_vitals.csv",
        feature_cols    = SITE_A_FEATURES,
        label_col       = "y_ihm",
        split           = "train",
        aligned_ids_csv = "data/vertical_splits/aligned_patient_ids.csv",
        timeseries_root = "data/mimic3-benchmarks/data/in-hospital-mortality/",
        task_type       = "binary",
    )
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
    x, mask, y = next(iter(loader))
    # x:    (32, 48, 7)   sequences
    # mask: (32, 48)      1=real timestep, 0=padding
    # y:    (32,)         binary label
"""

import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# YerevaNN LOS binning utilities (dead code — kept for los_bins path only)
# ---------------------------------------------------------------------------
try:
    _BENCH = Path(__file__).parent.parent / "mimic3-benchmarks"
    if str(_BENCH) not in sys.path:
        sys.path.insert(0, str(_BENCH))
    from mimic3models.metrics import CustomBins, get_bin_custom  # noqa: E402
except ModuleNotFoundError:
    CustomBins = None  # type: ignore
    get_bin_custom = None  # type: ignore

# ---------------------------------------------------------------------------
# Feature and label column definitions
# (mirrors vertical_split.py — keep in sync if the split protocol changes)
# ---------------------------------------------------------------------------

SITE_A_FEATURES = [
    "Heart Rate",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Temperature",
    "Oxygen saturation",
    "Respiratory rate",
    "Glascow coma scale total",       # typo preserved from YerevaNN source
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

PHENO_LABEL_COLS = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]

assert len(PHENO_LABEL_COLS) == 25


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VFLSiteDataset(Dataset):
    """
    PyTorch Dataset for one hospital site in the VFL-MTL setup.

    Parameters
    ----------
    site_csv        : path to site_X_*.csv (stay index + labels + split)
    feature_cols    : feature column names belonging to this site
    label_col       : str for binary/los_bins; list[str] for multilabel
    split           : 'train', 'val', or 'test'
    aligned_ids_csv : path to aligned_patient_ids.csv (PSI output)
    timeseries_root : YerevaNN task root directory that contains train/ and test/
                        Site A → .../in-hospital-mortality/
                        Site B → .../decompensation/
                        Site C → .../phenotyping/
    max_seq_len     : truncate / pad all sequences to this length (default 48)
    task_type       : 'binary' | 'los_bins' | 'multilabel'
    """

    def __init__(
        self,
        site_csv: Union[str, Path],
        feature_cols: list,
        label_col: Union[str, list],
        split: str,
        aligned_ids_csv: Union[str, Path],
        timeseries_root: Union[str, Path],
        max_seq_len: int = 48,
        task_type: str = "binary",
    ):
        assert split in ("train", "val", "test"), f"Unknown split: '{split}'"
        assert task_type in ("binary", "los_bins", "multilabel"), \
            f"Unknown task_type: '{task_type}'"

        self.feature_cols    = list(feature_cols)
        self.label_col       = label_col
        self.split           = split
        self.timeseries_root = Path(timeseries_root)
        self.max_seq_len     = max_seq_len
        self.task_type       = task_type

        # Load aligned patient IDs for this split
        aligned_df  = pd.read_csv(aligned_ids_csv)
        aligned_ids = set(
            aligned_df.loc[aligned_df["split"] == split, "subject_id"]
        )

        # Load site CSV; filter to this split and aligned patients only
        site_df = pd.read_csv(site_csv)
        site_df = site_df[
            (site_df["split"] == split) &
            (site_df["subject_id"].isin(aligned_ids))
        ].reset_index(drop=True)

        self.stays       = site_df["stay"].tolist()
        self.subject_ids = site_df["subject_id"].tolist()

        # Pre-load all timeseries into memory to avoid per-sample CSV reads
        subdir = "test" if split == "test" else "train"
        self._cache: dict[str, tuple] = {}
        for stay in self.stays:
            if stay not in self._cache:
                self._cache[stay] = self._load_timeseries_from_disk(
                    self.timeseries_root / subdir / stay
                )

        # Pre-compute labels
        if task_type == "multilabel":
            self.labels = site_df[label_col].values.astype(np.float32)  # (N, 25)

        elif task_type == "los_bins":
            los_hours = site_df[label_col].values.astype(float)
            bins = np.array([
                get_bin_custom(float(h), CustomBins.nbins)
                for h in los_hours
            ])
            bad = np.sum(bins == None)  # noqa: E711
            if bad:
                raise ValueError(
                    f"{bad} LOS values could not be mapped to a CustomBin "
                    f"(NaN or out-of-range in '{label_col}')."
                )
            self.labels = bins.astype(np.int64)  # (N,)

        else:  # binary
            self.labels = site_df[label_col].values.astype(np.float32)  # (N,)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.stays)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns
        -------
        x    : Tensor (max_seq_len, n_features)  float32
        mask : Tensor (max_seq_len,)              float32  1=real, 0=padding
        y    : Tensor scalar float32 (binary) | scalar int64 (los_bins)
                     | (25,) float32 (multilabel)
        """
        x_np, mask_np = self._load_timeseries(self.stays[idx])

        x    = torch.from_numpy(x_np)
        mask = torch.from_numpy(mask_np)

        raw = self.labels[idx]
        if self.task_type == "multilabel":
            y = torch.from_numpy(raw.copy())
        elif self.task_type == "los_bins":
            y = torch.tensor(int(raw), dtype=torch.long)
        else:
            y = torch.tensor(float(raw), dtype=torch.float32)

        return x, mask, y

    # ------------------------------------------------------------------

    def _load_timeseries(self, stay_filename: str) -> tuple:
        return self._cache[stay_filename]

    def _load_timeseries_from_disk(self, path) -> tuple:
        """
        Load one raw timeseries CSV, bin to 1-hour intervals, impute, pad.

        Steps
        -----
        1. Read CSV from the correct subdirectory (train/ or test/).
           Val stays are stored in train/ (YerevaNN convention).
        2. Floor each fractional timestamp to its integer hour bin;
           clip to [0, max_seq_len - 1].
        3. Group rows by bin; take the last non-NaN value per feature
           (matches YerevaNN Discretizer behaviour for continuous channels).
        4. Reindex to the full [0 .. max_seq_len-1] range.
        5. Forward-fill then backward-fill within the sequence.
        6. Fill any feature that was never observed in this stay with 0.
        7. Build mask: 1.0 for timesteps up to the last observed bin, 0.0 beyond.

        Returns
        -------
        x    : np.ndarray (max_seq_len, n_features)  float32
        mask : np.ndarray (max_seq_len,)             float32
        """
        df = pd.read_csv(path)

        # Step 2 — bin fractional hours to integers
        df["_bin"] = (
            df["Hours"]
            .astype(float)
            .apply(lambda t: min(int(t), self.max_seq_len - 1))
        )

        # Record actual stay length before reindexing
        max_observed_bin = int(df["_bin"].max()) if len(df) > 0 else 0
        actual_len       = min(max_observed_bin + 1, self.max_seq_len)

        # Step 3 — last non-NaN value per bin per feature
        available = [c for c in self.feature_cols if c in df.columns]
        binned    = df.groupby("_bin")[available].last()

        # Step 4 — reindex to full range
        binned = binned.reindex(range(self.max_seq_len))

        # Add columns for any feature absent from this file
        for col in self.feature_cols:
            if col not in binned.columns:
                binned[col] = np.nan
        binned = binned[self.feature_cols]   # enforce column order

        # Steps 5–6 — impute
        binned = binned.ffill().bfill().fillna(0.0)

        x    = binned.values.astype(np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0

        return x, mask


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> tuple:
    """
    Stack pre-padded samples into a batch.

    All sequences are already padded to max_seq_len in __getitem__,
    so this is a plain stack with no variable-length logic.

    Returns
    -------
    x    : Tensor (B, max_seq_len, n_features)
    mask : Tensor (B, max_seq_len)
    y    : Tensor (B,) or (B, 25)
    """
    xs, masks, ys = zip(*batch)
    return (
        torch.stack(xs),     # (B, T, F)
        torch.stack(masks),  # (B, T)
        torch.stack(ys),     # (B,) or (B, 25)
    )


# ---------------------------------------------------------------------------
# Convenience builder — constructs all three site loaders at once
# ---------------------------------------------------------------------------

def build_site_loaders(
    root: Union[str, Path],
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_seq_len: int = 48,
) -> dict:
    """
    Build DataLoaders for all three sites for a given split.

    Parameters
    ----------
    root        : project root (contains data/vertical_splits/ and
                  data/mimic3-benchmarks/)
    split       : 'train', 'val', or 'test'
    batch_size  : samples per batch
    num_workers : DataLoader worker processes (0 = main process only)
    max_seq_len : sequence length passed to VFLSiteDataset

    Returns
    -------
    dict[str, DataLoader] with keys 'A', 'B', 'C'
    """
    root       = Path(root)
    splits_dir = root / "data" / "vertical_splits"
    bench_dir  = root / "data" / "mimic3-benchmarks" / "data"
    aligned    = splits_dir / "aligned_patient_ids.csv"

    configs = {
        "A": dict(
            site_csv        = splits_dir / "site_A_vitals.csv",
            feature_cols    = SITE_A_FEATURES,
            label_col       = "y_ihm",
            timeseries_root = bench_dir / "in-hospital-mortality",
            task_type       = "binary",
        ),
        "B": dict(
            site_csv        = splits_dir / "site_B_labs.csv",
            feature_cols    = SITE_B_FEATURES,
            label_col       = "y_decomp",
            timeseries_root = bench_dir / "decompensation",
            task_type       = "binary",
        ),
        "C": dict(
            site_csv        = splits_dir / "site_C_composite.csv",
            feature_cols    = SITE_C_FEATURES,
            label_col       = PHENO_LABEL_COLS,
            timeseries_root = bench_dir / "phenotyping",
            task_type       = "multilabel",
        ),
    }

    loaders = {}
    for site_id, cfg in configs.items():
        ds = VFLSiteDataset(
            site_csv        = cfg["site_csv"],
            feature_cols    = cfg["feature_cols"],
            label_col       = cfg["label_col"],
            split           = split,
            aligned_ids_csv = aligned,
            timeseries_root = cfg["timeseries_root"],
            max_seq_len     = max_seq_len,
            task_type       = cfg["task_type"],
        )
        loaders[site_id] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            collate_fn  = collate_fn,
            num_workers = num_workers,
            drop_last   = True,  # ensures all sites produce equal-sized batches for lockstep zip
        )

    return loaders
