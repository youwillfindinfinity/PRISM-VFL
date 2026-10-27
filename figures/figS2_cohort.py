"""
figures/figS2_cohort.py — S2 Fig: cohort attrition and acuity bias of the aligned cohort.

Unit of analysis is the ICU stay, as in training. Aligned = stays of PSI-aligned patients present
at all three sites (A∩B∩C, 21,135). Excluded = stays of the filtered cohort (union of the site CSVs)
that are not aligned. Decompensation is compared on Site B stays, Phenotyping on Site C stays.

Two steps, because the site CSVs only exist where MIMIC-III is stored:
  1. python figures/figS2_cohort.py --data data/vertical_splits --stats results_revision/cohort_stats.csv
     aggregates the site CSVs into counts and rates (no patient-level rows).
  2. python figures/figS2_cohort.py
     plots Manuscript/figures_revision/S2_Cohort.png from that CSV.

Panels:
  A. Unique stays and patients per site and in the aligned cohort
  B. Decompensation positive rate, aligned vs. excluded stays
  C. Prevalence of the 25 phenotypes, excluded vs. aligned stays, with y = x
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SITE_FILES = {"A": "site_A_vitals.csv", "B": "site_B_labs.csv", "C": "site_C_composite.csv"}
NON_LABEL_C = {"stay", "subject_id", "split", "Height", "Weight", "Mean blood pressure"}

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]


def compute(splits_dir: Path) -> pd.DataFrame:
    sites = {s: pd.read_csv(splits_dir / f) for s, f in SITE_FILES.items()}
    aligned_ids = set(pd.read_csv(splits_dir / "aligned_patient_ids.csv")["subject_id"])
    aligned = set.intersection(*(set(df.loc[df["subject_id"].isin(aligned_ids), "stay"]) for df in sites.values()))
    union = pd.concat([df[["stay", "subject_id"]] for df in sites.values()]).drop_duplicates("stay")

    rows = []
    def add(metric, group, value, n, label=""):
        rows.append(dict(metric=metric, group=group, label=label, value=value, n=n))

    for s, df in sites.items():
        add("n_stays", f"Site {s}", df["stay"].nunique(), df["stay"].nunique())
        add("n_patients", f"Site {s}", df["subject_id"].nunique(), df["subject_id"].nunique())
    is_al = union["stay"].isin(aligned)
    add("n_stays", "Filtered cohort", len(union), len(union))
    add("n_patients", "Filtered cohort", union["subject_id"].nunique(), union["subject_id"].nunique())
    add("n_stays", "Aligned", int(is_al.sum()), int(is_al.sum()))
    add("n_patients", "Aligned", union.loc[is_al, "subject_id"].nunique(), union.loc[is_al, "subject_id"].nunique())

    b = sites["B"].assign(aligned=sites["B"]["stay"].isin(aligned))
    for grp, g in (("Aligned", b[b.aligned]), ("Excluded", b[~b.aligned])):
        add("decomp_rate", grp, g["y_decomp"].mean(), len(g))
    tr = b[b.aligned & (b["split"] == "train")]
    add("decomp_rate", "Aligned train", tr["y_decomp"].mean(), len(tr))

    c = sites["C"].assign(aligned=sites["C"]["stay"].isin(aligned))
    labels = [col for col in c.columns if col not in NON_LABEL_C | {"aligned"}]
    for grp, g in (("Aligned", c[c.aligned]), ("Excluded", c[~c.aligned])):
        for lab in labels:
            add("pheno_prev", grp, g[lab].mean(), len(g), lab)
        add("pheno_prev_mean", grp, g[labels].mean().mean(), len(g))

    al_subj = union.loc[is_al, ["stay", "subject_id"]].merge(sites["A"][["stay", "split"]], on="stay")
    add("patients_in_multiple_splits", "Aligned", int((al_subj.groupby("subject_id")["split"].nunique() > 1).sum()),
        al_subj["subject_id"].nunique())
    return pd.DataFrame(rows)


def plot(stats: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11, "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 12, "axes.titleweight": "normal", "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })
    get = lambda m, g: stats[(stats.metric == m) & (stats.group == g)].iloc[0]

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], hspace=0.3, wspace=0.35)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # A — attrition
    groups = ["Site A", "Site B", "Site C", "Aligned"]
    names = ["Site A (IHM)", "Site B (Decomp)", "Site C (Phenotyping)", "A ∩ B ∩ C (aligned)"]
    total = get("n_stays", "Filtered cohort").value
    stays = [get("n_stays", g).value for g in groups]
    pats = [get("n_patients", g).value for g in groups]
    bars = ax_a.barh(names[::-1], stays[::-1], color=[_C[5], _C[2], _C[1], _C[0]])
    for bar, s, p in zip(bars, stays[::-1], pats[::-1]):
        ax_a.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                  f"{s:,.0f} stays ({s / total:.0%}) · {p:,.0f} patients", va="center", fontsize=8.5)
    ax_a.set_xlim(0, total * 1.6)
    ax_a.set_xlabel(f"Unique ICU stays (after task-level exclusions, {total:,.0f})")

    # B — Decompensation positive rate
    al, ex = get("decomp_rate", "Aligned"), get("decomp_rate", "Excluded")
    bars = ax_b.bar([f"Aligned\n(n={al.n:,.0f})", f"Excluded\n(n={ex.n:,.0f})"],
                    [100 * al.value, 100 * ex.value], color=[_C[0], _C[3]], width=0.55)
    for bar, v in zip(bars, (al.value, ex.value)):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{100 * v:.1f}%",
                  ha="center", va="bottom", fontsize=9)
    ax_b.set_ylabel("Decompensation positive rate (%)")
    ax_b.set_ylim(0, 100 * max(al.value, ex.value) * 1.25)

    # C — phenotype prevalence
    prev = stats[stats.metric == "pheno_prev"].pivot(index="label", columns="group", values="value") * 100
    ax_c.scatter(prev["Aligned"], prev["Excluded"], color=_C[4], s=40, zorder=3)
    offsets = {"Disorders of lipid metabolism": (-5, 0, "right"),
               "Complications of surgical procedures or medical care": (-5, 0, "right"),
               "Acute and unspecified renal failure": (5, -8, "left")}
    for lab, r in prev[prev["Aligned"] > 15].iterrows():
        name = lab.split(";")[0].split(" (")[0]
        dx, dy, ha = offsets.get(lab, (5, -3, "left"))
        ax_c.annotate(name if len(name) <= 30 else name[:28] + "…", (r["Aligned"], r["Excluded"]),
                      xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha, va="center")
    lim = max(prev.max().max(), 1) * 1.1
    ax_c.plot([0, lim], [0, lim], "--", color="#888888", linewidth=0.9, label="y = x")
    ax_c.set_xlim(0, lim); ax_c.set_ylim(0, lim)
    ax_c.set_xlabel("Prevalence — aligned stays (%)")
    ax_c.set_ylabel("Prevalence — excluded stays (%)")
    ax_c.legend(loc="upper left")

    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.text(-0.08 if ax is not ax_c else -0.04, 1.04, letter, transform=ax.transAxes,
                fontsize=16, fontweight="bold")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=800, bbox_inches="tight")
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=None, help="data/vertical_splits; set to compute the stats CSV")
    parser.add_argument("--stats",  default="results_revision/cohort_stats.csv", help="'-' writes to stdout")
    parser.add_argument("--output", default="Manuscript/figures_revision/S2_Cohort.png")
    args = parser.parse_args()

    if args.data:
        stats = compute(Path(args.data))
        stats.to_csv(sys.stdout if args.stats == "-" else args.stats, index=False)
    else:
        plot(pd.read_csv(args.stats), Path(args.output))


if __name__ == "__main__":
    main()
