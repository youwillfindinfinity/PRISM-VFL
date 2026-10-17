"""
verify_workspace.py — Step 7 workspace readiness check.

Checks:
  ✓ All four task directories exist and are non-empty
  ✓ vertical_splits/ contains all three site CSVs
  ✓ aligned_patient_ids.csv covers >= 18,000 patients
  ✓ Exact VFL stay alignment per split (same stays, same order, same subject,
    same split at every site; no duplicate stays)
  ✓ No feature column overlap between site CSVs
  ✓ Class balance within acceptable range for each task

Usage:
  python data_prep/verify_workspace.py [--root WORKSPACE_ROOT]
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_prep.dataset import align_stays

# ── columns that are not clinical features ─────────────────────────────────
NON_FEATURE_COLS = {
    "stay", "subject_id", "split",
    # task labels
    "y_ihm", "y_decomp",
    # phenotyping labels (25 ICD groups)
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
}

TASKS = [
    "in-hospital-mortality",
    "decompensation",
    "length-of-stay",
    "phenotyping",
]

SITE_CSVS = {
    "site_A_vitals.csv": "y_ihm",
    "site_B_labs.csv": "y_decomp",
    "site_C_composite.csv": None,  # phenotyping — multi-label, checked separately
}

ALIGNED_MIN_PATIENTS = 18_000


def check(cond: bool, msg_ok: str, msg_fail: str, results: list) -> bool:
    if cond:
        print(f"  ✓ {msg_ok}")
        results.append(True)
    else:
        print(f"  ✗ {msg_fail}")
        results.append(False)
    return cond


def verify(root: Path) -> bool:
    results: list[bool] = []
    bench_data = root / "data" / "mimic3-benchmarks" / "data"
    splits_dir = root / "data" / "vertical_splits"

    print("\n══════════════════════════════════════════════════")
    print("  VFL-MTL Workspace Verification")
    print(f"  Root: {root}")
    print("══════════════════════════════════════════════════\n")

    # ── 1. Task directories ────────────────────────────────────────────────
    print("[ 1 ] Task directories")
    for task in TASKS:
        task_dir = bench_data / task
        has_dir = task_dir.is_dir()
        non_empty = has_dir and any(task_dir.iterdir())
        has_train = has_dir and (task_dir / "train_listfile.csv").exists()
        has_test  = has_dir and (task_dir / "test_listfile.csv").exists()
        has_val   = has_dir and (task_dir / "val_listfile.csv").exists()
        ok = non_empty and has_train and has_test and has_val
        check(
            ok,
            f"{task}/  [train_listfile, val_listfile, test_listfile present]",
            f"{task}/  MISSING or incomplete "
            f"(dir={has_dir}, train={has_train}, val={has_val}, test={has_test})",
            results,
        )

    # ── 2. Vertical split CSVs ─────────────────────────────────────────────
    print("\n[ 2 ] Vertical split files")
    for fname in list(SITE_CSVS) + ["aligned_patient_ids.csv"]:
        fpath = splits_dir / fname
        check(
            fpath.exists() and fpath.stat().st_size > 0,
            f"{fname} exists",
            f"{fname} MISSING or empty",
            results,
        )

    # ── 3. Aligned patient count ──────────────────────────────────────────
    print("\n[ 3 ] Aligned patient count")
    aligned_path = splits_dir / "aligned_patient_ids.csv"
    if aligned_path.exists():
        n = len(pd.read_csv(aligned_path))
        check(
            n >= ALIGNED_MIN_PATIENTS,
            f"aligned_patient_ids.csv: {n:,} patients (>= {ALIGNED_MIN_PATIENTS:,})",
            f"aligned_patient_ids.csv: only {n:,} patients (< {ALIGNED_MIN_PATIENTS:,} required)",
            results,
        )
    else:
        print(f"  ✗ aligned_patient_ids.csv not found — skipping count check")
        results.append(False)

    # ── 3b. Exact VFL stay alignment ────────────────────────────────────────
    # Row i must be the same ICU stay at every site; partial overlap is not enough.
    print("\n[ 3b ] VFL stay alignment (exact)")
    site_csv_paths = {
        "A": splits_dir / "site_A_vitals.csv",
        "B": splits_dir / "site_B_labs.csv",
        "C": splits_dir / "site_C_composite.csv",
    }
    if aligned_path.exists() and all(p.exists() for p in site_csv_paths.values()):
        aligned_df = pd.read_csv(aligned_path)
        sites = {s: pd.read_csv(p, usecols=["stay", "subject_id", "split"])
                 for s, p in site_csv_paths.items()}

        for s, df in sites.items():
            n_dup = int(df["stay"].duplicated().sum())
            check(n_dup == 0, f"Site {s}: no duplicate stay rows",
                  f"Site {s}: {n_dup:,} duplicate stay rows", results)

        # Checked on the full CSVs: a conflicting row would otherwise just drop out of the filter.
        for col in ["subject_id", "split"]:
            by_site = pd.concat(
                {s: df.drop_duplicates("stay").set_index("stay")[col] for s, df in sites.items()},
                axis=1,
            )
            n_shared = int((by_site.notna().sum(axis=1) >= 2).sum())
            n_conflict = int((by_site.nunique(axis=1) > 1).sum())
            check(n_conflict == 0,
                  f"Same {col} at every site for all {n_shared:,} stays held by 2+ sites",
                  f"{n_conflict:,} shared stays have a different {col} across sites", results)

        for sp in ["train", "val", "test"]:
            aligned_ids = set(aligned_df.loc[aligned_df["split"] == sp, "subject_id"])
            views = {}
            for s, df in sites.items():  # same filter as VFLSiteDataset
                d = df[(df["split"] == sp) & df["subject_id"].isin(aligned_ids)]
                views[s] = SimpleNamespace(stays=d["stay"].tolist(),
                                           subject_ids=d["subject_id"].tolist(),
                                           labels=np.arange(len(d)))
            try:
                align_stays(views)
            except ValueError as e:
                check(False, "", f"Split '{sp}': {e}", results)
                continue
            stays = [v.stays for v in views.values()]
            subjects = [v.subject_ids for v in views.values()]
            check(stays[0] == stays[1] == stays[2] and stays[0] == sorted(stays[0]),
                  f"Split '{sp}': {len(stays[0]):,} stays, identical set and sorted order at A/B/C",
                  f"Split '{sp}': aligned stays differ across sites", results)
            check(subjects[0] == subjects[1] == subjects[2],
                  f"Split '{sp}': identical stay->subject_id mapping at A/B/C",
                  f"Split '{sp}': stay->subject_id mapping differs across sites", results)
            n_covered = len(set(subjects[0]))
            check(n_covered == len(aligned_ids),
                  f"Split '{sp}': all {len(aligned_ids):,} aligned patients have aligned stays",
                  f"Split '{sp}': only {n_covered:,} of {len(aligned_ids):,} aligned patients "
                  f"have aligned stays — split assignments disagree with aligned_patient_ids.csv",
                  results)
    else:
        print("  ✗ aligned_patient_ids.csv or a site CSV missing — skipping alignment check")
        results.append(False)

    # ── 4. No feature column overlap between site CSVs ────────────────────
    print("\n[ 4 ] Feature column overlap between sites")
    site_feature_sets: dict[str, set] = {}
    for fname in SITE_CSVS:
        fpath = splits_dir / fname
        if fpath.exists():
            cols = set(pd.read_csv(fpath, nrows=0).columns) - NON_FEATURE_COLS
            site_feature_sets[fname] = cols

    if len(site_feature_sets) == len(SITE_CSVS):
        names = list(site_feature_sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = site_feature_sets[a] & site_feature_sets[b]
                check(
                    len(overlap) == 0,
                    f"No overlap between {a} and {b}",
                    f"Overlap between {a} and {b}: {sorted(overlap)}",
                    results,
                )
        # Print feature summary
        print("  Feature counts per site:")
        for fname, feats in site_feature_sets.items():
            print(f"    {fname}: {len(feats)} features — {sorted(feats)}")
    else:
        print("  ✗ Cannot check overlap — one or more site CSVs missing")
        results.append(False)

    # ── 5. Class balance ──────────────────────────────────────────────────
    print("\n[ 5 ] Class balance")

    # 5a. In-hospital mortality (binary, acceptable range 5–40 %)
    site_a = splits_dir / "site_A_vitals.csv"
    if site_a.exists():
        df_a = pd.read_csv(site_a, usecols=["y_ihm", "split"])
        train_a = df_a[df_a["split"] == "train"]
        pos_rate = train_a["y_ihm"].mean()
        check(
            0.05 <= pos_rate <= 0.40,
            f"IHM mortality rate (train): {pos_rate:.1%}  [5%–40% acceptable]",
            f"IHM mortality rate (train): {pos_rate:.1%}  OUT OF RANGE [5%–40%]",
            results,
        )

    # 5b. Decompensation (binary — check positive rate is plausible 5–50 %)
    site_b = splits_dir / "site_B_labs.csv"
    if site_b.exists():
        df_b = pd.read_csv(site_b, usecols=["y_decomp", "split"])
        train_b = df_b[df_b["split"] == "train"]["y_decomp"]
        pos_rate_b = train_b.mean()
        check(
            0.05 <= pos_rate_b <= 0.50,
            f"Decompensation positive rate (train): {pos_rate_b:.1%}  [5%–50% acceptable]",
            f"Decompensation positive rate (train): {pos_rate_b:.1%}  OUT OF RANGE [5%–50%]",
            results,
        )

    # 5c. Phenotyping (multi-label — check no label is near-zero or near-one)
    site_c = splits_dir / "site_C_composite.csv"
    pheno_labels = [
        "Acute and unspecified renal failure", "Acute cerebrovascular disease",
        "Acute myocardial infarction", "Cardiac dysrhythmias",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Complications of surgical procedures or medical care",
        "Conduction disorders", "Congestive heart failure; nonhypertensive",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus with complications",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism", "Essential hypertension",
        "Fluid and electrolyte disorders", "Gastrointestinal hemorrhage",
        "Hypertension with complications and secondary hypertension",
        "Other liver diseases", "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)", "Shock",
    ]
    if site_c.exists():
        present = [c for c in pheno_labels if c in pd.read_csv(site_c, nrows=0).columns]
        df_c = pd.read_csv(site_c, usecols=present + ["split"])
        train_c = df_c[df_c["split"] == "train"][present]
        rates = train_c.mean()
        degenerate = rates[(rates < 0.001) | (rates > 0.999)]
        check(
            len(degenerate) == 0,
            f"Phenotyping: all {len(present)} labels have prevalence in (0.1%, 99.9%)",
            f"Phenotyping: {len(degenerate)} degenerate label(s): {list(degenerate.index)}",
            results,
        )
        # Report prevalence range
        print(f"    Prevalence range: {rates.min():.1%} – {rates.max():.1%}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════")
    passed = sum(results)
    total  = len(results)
    if passed == total:
        print(f"  ALL CHECKS PASSED ({passed}/{total})")
        print("  Workspace is ready for experiments.\n")
        return True
    else:
        print(f"  {passed}/{total} checks passed — {total - passed} FAILED")
        print("  Resolve failures before running experiments.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify VFL-MTL workspace readiness.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the vfl_mlt project root (default: parent of data_prep/)",
    )
    args = parser.parse_args()

    ok = verify(args.root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
