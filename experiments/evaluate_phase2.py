"""
experiments/evaluate_phase2.py — PCMU Phase 2: Cross-factor independence gate (additive form).

Loads results/pcmu_phase2_factorial.csv (108 cells: 3 embed_dims × 4 ε × 3 task_configs × 3 seeds).
Computes raw PCMU components per cell using the TRUE centralized oracle (results/centralized.csv)
as the η_priv ceiling — not VFL-MTL at ε=∞. Then z-scores components over the factorial pool,
applies additive aggregation, and shifts so PCMU=1.0 = centralized (non-federated, no DP) baseline.

Phase 2 test battery for additive PCMU:
  1. Variance decomposition: does Δ_m drive more PCMU variance than η_comm?
     Confirms weights work as intended despite component variance asymmetry.
  2. embed_dim ANOVA: interaction η² < 0.05 on PCMU_additive.
     Confirms metric is consistent across technical parameters.
  3. ε × task_count: reported as documented substantive finding, not a gate failure.

Partial correlations between components are also reported as documented ε-coupling findings
(they motivated the switch from geometric to additive aggregation).

Outputs results/pcmu_phase2_gate.csv.

Usage:
    python experiments/evaluate_phase2.py
    python experiments/evaluate_phase2.py --factorial results/pcmu_phase2_factorial.csv
                                          --centralized results/centralized.csv
                                          --exp1 results/exp1.csv
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

W = {"ihm": 0.5, "decomp": 0.3, "pheno": 0.2}
PCMU_WEIGHTS = {"delta_m": 0.70, "eta_priv": 0.20, "eta_comm": 0.10}

ETA2_THRESHOLD = 0.05
PVAL_THRESHOLD = 0.05
PARTIAL_R_THRESHOLD = 0.30

_TASK_COLS = {
    "ihm":    "val_ihm_auroc",
    "decomp": "val_decomp_auroc",
    "pheno":  "val_pheno_macro_auroc",
}
_CEN_COLS = {
    "ihm":    "val_ihm_auc_roc",
    "decomp": "val_decomp_auc_roc",
    "pheno":  "val_pheno_macro_auc",
}
_ST_MODELS = {
    "ihm":    "ST-IHM",
    "decomp": "ST-Decomp",
    "pheno":  "ST-Pheno",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_centralized_aurocs(cen_df: pd.DataFrame) -> dict[str, float]:
    """Peak AUROC per task from centralized oracle CSV."""
    out = {}
    for task, col in _CEN_COLS.items():
        if col in cen_df.columns:
            out[task] = float(cen_df[col].dropna().max())
        else:
            out[task] = float("nan")
    return out


def load_st_aurocs(exp1_df: pd.DataFrame) -> dict[str, float]:
    """Mean-across-seeds peak AUROC from VFL single-task baselines in exp1.csv."""
    out = {}
    for task, model_name in _ST_MODELS.items():
        col = _TASK_COLS[task]
        rows = exp1_df[exp1_df["model"] == model_name]
        if rows.empty or col not in rows.columns:
            out[task] = float("nan")
        else:
            out[task] = float(rows.groupby("seed")[col].max().mean())
    return out


# ---------------------------------------------------------------------------
# Component computation
# ---------------------------------------------------------------------------

def compute_components(
    df: pd.DataFrame,
    centralized_aurocs: dict[str, float],
    st_aurocs: dict[str, float],
) -> tuple[pd.DataFrame, float]:
    """
    Add raw Δ_m, η_priv, η_comm to each factorial row.

    η_priv: plain utility ratio Σ w_t·(M_t/M_t^cen) — no log factor (Option A).
    Δ_m:    aggregate across all active tasks using ST baselines from exp1.csv.
            Σ w_t·(M_t^MTL − M_t^ST)/M_t^ST / T  (same as multitask_gain()).
    R_ST:   convergence_round of ihm_only at ε=∞ per (embed_dim, seed),
            used as R_ref in η_comm = log(1 + R_ST / R_MTL) (Option A).
    """
    inf_rows = df[df["epsilon_level"] == float("inf")].copy()

    # R_ref = single-task convergence (ihm_only), not all_tasks (Option A)
    r_st_base = inf_rows[inf_rows["task_config"] == "ihm_only"].set_index(
        ["embed_dim", "seed"]
    )["convergence_round"]
    R_ref_global = float(r_st_base.median())
    print(f"R_ST (ihm_only, ε=∞, median across embed_dims/seeds) = {R_ref_global:.0f} rounds")

    rows = []
    for _, r in df.iterrows():
        ed         = int(r["embed_dim"])
        eps        = float(r["epsilon_level"])
        task_count = int(r["task_count"])
        seed       = int(r["seed"])

        # Active tasks for this row
        active = {
            "ihm":    True,
            "decomp": float(r["val_decomp_auroc"]) > 0 and task_count > 1,
            "pheno":  float(r["val_pheno_macro_auroc"]) > 0 and task_count > 2,
        }
        total_w = sum(W[t] for t in W if active.get(t, False))

        # η_priv — plain utility ratio, no log factor (Option A)
        if total_w == 0:
            eta_priv = float("nan")
        else:
            eta_priv = 0.0
            for t, col in _TASK_COLS.items():
                if not active.get(t, False):
                    continue
                m_ceil = centralized_aurocs.get(t, 0.0)
                m_eps  = float(r[col]) if not np.isnan(float(r[col])) else 0.0
                if m_ceil > 0:
                    eta_priv += (W[t] / total_w) * (m_eps / m_ceil)

        # Δ_m — aggregate across all active tasks using exp1.csv ST baselines
        if task_count == 1:
            delta_m = 0.0
        else:
            active_tasks = [
                t for t in _TASK_COLS
                if active.get(t, False)
                and st_aurocs.get(t, 0.0) > 0
                and np.isfinite(float(r[_TASK_COLS[t]]))
            ]
            if not active_tasks:
                delta_m = 0.0
            else:
                T = len(active_tasks)
                delta_m = sum(
                    W[t] * (float(r[_TASK_COLS[t]]) - st_aurocs[t]) / st_aurocs[t]
                    for t in active_tasks
                ) / T

        # η_comm — R_ref = single-task rounds (Option A)
        R = float(r["convergence_round"])
        try:
            R_ST = float(r_st_base.loc[(ed, seed)])
        except KeyError:
            R_ST = R_ref_global
        eta_comm = np.log(1.0 + R_ST / R) if R > 0 else float("nan")

        rows.append({**r.to_dict(),
                     "delta_m":  delta_m,
                     "eta_priv": eta_priv,
                     "eta_comm": eta_comm,
                     "R_ST":     R_ST})

    return pd.DataFrame(rows), R_ref_global


# ---------------------------------------------------------------------------
# Centralized anchor components
# ---------------------------------------------------------------------------

def compute_centralized_components(
    centralized_aurocs: dict[str, float],
    st_aurocs: dict[str, float],
    eta_comm_max: float,
) -> dict[str, float]:
    """
    Compute the three PCMU components for the centralized (non-federated, no-DP) oracle.

    η_priv = 1.0  — cen/cen = 1.0, log_factor = 1.0 (ε=∞ plain utility ratio)
    Δ_m    = weighted gain of centralized AUROC over VFL single-task baselines
    η_comm = max observed η_comm in factorial (centralized has no federation overhead)
    """
    tasks = [t for t in W if st_aurocs.get(t, 0.0) > 0 and not np.isnan(centralized_aurocs.get(t, float("nan")))]
    T = len(tasks)
    if T == 0:
        delta_m_cen = float("nan")
    else:
        delta_m_cen = sum(
            W[t] * (centralized_aurocs[t] - st_aurocs[t]) / st_aurocs[t]
            for t in tasks
        ) / T

    return {
        "delta_m":  delta_m_cen,
        "eta_priv": 1.0,
        "eta_comm": eta_comm_max,
    }


# ---------------------------------------------------------------------------
# Z-score + additive PCMU
# ---------------------------------------------------------------------------

def add_additive_pcmu(
    df: pd.DataFrame,
    cen_components: dict[str, float],
) -> pd.DataFrame:
    """
    Z-score each component over the factorial pool, compute additive PCMU,
    shift so PCMU = 1.0 = centralized (non-federated, no-DP) baseline.

    Z-score parameters are derived from the factorial only; centralized components
    are projected onto that scale to compute the anchor.
    """
    df = df.copy()

    z_params: dict[str, tuple[float, float]] = {}
    for col in ("delta_m", "eta_priv", "eta_comm"):
        vals = df[col].dropna()
        mu    = float(vals.mean())
        sigma = float(vals.std(ddof=1))
        z_params[col] = (mu, sigma)
        df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    df["pcmu_additive"] = (
        PCMU_WEIGHTS["delta_m"]  * df["delta_m_z"] +
        PCMU_WEIGHTS["eta_priv"] * df["eta_priv_z"] +
        PCMU_WEIGHTS["eta_comm"] * df["eta_comm_z"]
    )

    # Project centralized components onto factorial z-score scale
    cen_pcmu_additive = 0.0
    print("\nCentralized anchor components (z-scored):")
    for col, w_key in [("delta_m", "delta_m"), ("eta_priv", "eta_priv"), ("eta_comm", "eta_comm")]:
        mu, sigma = z_params[col]
        raw = cen_components[col]
        z   = (raw - mu) / sigma if sigma > 0 else 0.0
        cen_pcmu_additive += PCMU_WEIGHTS[w_key] * z
        print(f"  {col}: raw={raw:.4f}  z={z:.4f}  weighted={PCMU_WEIGHTS[w_key]*z:.4f}")
    print(f"  PCMU_centralized (pre-shift): {cen_pcmu_additive:.4f}")

    df["pcmu"] = df["pcmu_additive"] - cen_pcmu_additive + 1.0
    return df


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

def run_anova(df: pd.DataFrame, response: str) -> pd.DataFrame:
    """Three-way ANOVA with all interactions for embed_dim × eps_cat × task_count."""
    d = df[["embed_dim_cat", "eps_cat", "task_count", response]].dropna()
    formula = (
        f"{response} ~ C(embed_dim_cat) + C(eps_cat) + C(task_count)"
        f" + C(embed_dim_cat):C(eps_cat)"
        f" + C(embed_dim_cat):C(task_count)"
        f" + C(eps_cat):C(task_count)"
        f" + C(embed_dim_cat):C(eps_cat):C(task_count)"
    )
    model  = ols(formula, data=d).fit()
    table  = anova_lm(model, typ=2)
    ss_tot = table["sum_sq"].sum()
    table["eta_sq"] = table["sum_sq"] / ss_tot
    return table


# ---------------------------------------------------------------------------
# Partial correlations
# ---------------------------------------------------------------------------

def partial_corr(df: pd.DataFrame, x: str, y: str, cov: str) -> tuple[float, float]:
    d = df[[x, y, cov]].dropna()
    r_xy = stats.pearsonr(d[x], d[y])[0]
    r_xc = stats.pearsonr(d[x], d[cov])[0]
    r_yc = stats.pearsonr(d[y], d[cov])[0]
    denom = np.sqrt((1 - r_xc**2) * (1 - r_yc**2))
    if denom == 0:
        return float("nan"), float("nan")
    pr = (r_xy - r_xc * r_yc) / denom
    n  = len(d)
    t  = pr * np.sqrt((n - 2 - 1) / (1 - pr**2)) if abs(pr) < 1 else float("nan")
    p  = 2 * stats.t.sf(abs(t), df=n - 3) if np.isfinite(t) else float("nan")
    return float(pr), float(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial",    default="results/pcmu_phase2_factorial.csv")
    parser.add_argument("--centralized",  default="results/centralized.csv")
    parser.add_argument("--exp1",         default="results/exp1.csv")
    parser.add_argument("--out",          default="results/pcmu_phase2_gate.csv")
    args = parser.parse_args()

    # Load factorial
    df = pd.read_csv(args.factorial)
    df = df[pd.to_numeric(df["embed_dim"], errors="coerce").notna()].copy()
    df["embed_dim"]     = df["embed_dim"].astype(int)
    df["epsilon_level"] = df["epsilon_level"].astype(float)
    df["seed"]          = df["seed"].astype(int)
    print(f"Loaded {len(df)} rows from {args.factorial}")

    # Load centralized oracle and VFL-ST baselines
    cen_df   = pd.read_csv(args.centralized)
    exp1_df  = pd.read_csv(args.exp1)
    centralized_aurocs = load_centralized_aurocs(cen_df)
    st_aurocs          = load_st_aurocs(exp1_df)
    print(f"Centralized AUROCs: { {k: round(v,4) for k,v in centralized_aurocs.items()} }")
    print(f"VFL-ST AUROCs:      { {k: round(v,4) for k,v in st_aurocs.items()} }")

    # Compute raw components
    df, R_ref = compute_components(df, centralized_aurocs, st_aurocs)

    # Centralized anchor components
    eta_comm_max = float(df["eta_comm"].dropna().max())
    cen_components = compute_centralized_components(centralized_aurocs, st_aurocs, eta_comm_max)
    print(f"\nCentralized raw components: delta_m={cen_components['delta_m']:.4f}, "
          f"eta_priv={cen_components['eta_priv']:.4f}, eta_comm={cen_components['eta_comm']:.4f}")

    # Z-score + additive PCMU with centralized anchor
    df = add_additive_pcmu(df, cen_components)

    # Encode categorical factors
    df["embed_dim_cat"] = df["embed_dim"].astype(str)
    df["eps_cat"] = df["epsilon_level"].apply(
        lambda e: "inf" if not np.isfinite(e) else str(e)
    )
    df_dp = df[np.isfinite(df["epsilon_level"])].copy()

    print("\n" + "="*60)
    print("PHASE 2 GATE — ADDITIVE PCMU VALIDATION")
    print("="*60)

    gate_pass = True
    gate_rows = []

    # --- Component-level ANOVAs (documented ε-coupling finding) ---
    print("\n[Component interactions — documented ε-coupling finding]")
    for component, label, use_dp in [
        ("eta_priv", "η_priv", True),
        ("eta_comm", "η_comm", False),
        ("delta_m",  "Δ_m",   False),
    ]:
        subset = df_dp if use_dp else df
        print(f"\n── ANOVA: {label} ──")
        try:
            tbl = run_anova(subset, component)
            print(tbl[["sum_sq", "df", "F", "PR(>F)", "eta_sq"]].round(4))
            for idx, row in tbl[tbl.index.str.contains(":", regex=False)].iterrows():
                gate_rows.append({
                    "component": component, "term": idx,
                    "eta_sq": round(float(row["eta_sq"]), 4),
                    "p":      round(float(row["PR(>F)"]), 4),
                    "gate":   "documented-finding",
                })
        except Exception as e:
            print(f"  ANOVA failed: {e}")

    # --- Gate 1: embed_dim interactions in PCMU_additive ---
    print("\n[Gate 1 — embed_dim interactions in PCMU_additive (technical consistency)]")
    try:
        tbl = run_anova(df_dp, "pcmu")
        print(tbl[["sum_sq", "df", "F", "PR(>F)", "eta_sq"]].round(4))
        embed_terms = [idx for idx in tbl.index if "embed_dim_cat" in idx and ":" in idx]
        for idx in embed_terms:
            row  = tbl.loc[idx]
            eta2 = float(row["eta_sq"])
            pval = float(row["PR(>F)"])
            flag = "FAIL" if (eta2 >= ETA2_THRESHOLD and pval <= PVAL_THRESHOLD) else "ok"
            if flag == "FAIL":
                gate_pass = False
            gate_rows.append({
                "component": "pcmu_additive", "term": idx,
                "eta_sq": round(eta2, 4), "p": round(pval, 4), "gate": flag,
            })
            print(f"  {idx}: η²={eta2:.4f}  p={pval:.4f}  [{flag}]")

        # ε × task_count: documented finding
        eps_task_term = [idx for idx in tbl.index if "eps_cat" in idx and "task_count" in idx and "embed_dim" not in idx]
        for idx in eps_task_term:
            row = tbl.loc[idx]
            gate_rows.append({
                "component": "pcmu_additive", "term": idx,
                "eta_sq": round(float(row["eta_sq"]), 4),
                "p":      round(float(row["PR(>F)"]), 4),
                "gate":   "documented-finding",
            })
            print(f"  {idx}: η²={float(row['eta_sq']):.4f}  p={float(row['PR(>F)']):.4f}  [documented-finding]")
    except Exception as e:
        print(f"  ANOVA failed: {e}")

    # --- Gate 2: variance decomposition ---
    print("\n[Gate 2 — variance decomposition (weights work as intended)]")
    pcmu_var = float(df_dp["pcmu"].dropna().var())
    for col, wk in [("delta_m", "delta_m"), ("eta_priv", "eta_priv"), ("eta_comm", "eta_comm")]:
        contrib = PCMU_WEIGHTS[wk]**2 * float(df_dp[f"{col}_z"].dropna().var())
        pct = 100 * contrib / pcmu_var if pcmu_var > 0 else float("nan")
        print(f"  {col}: {pct:.1f}% of PCMU variance")
        gate_rows.append({
            "component": col, "term": "variance_contribution_pct",
            "eta_sq": round(pct / 100, 4), "p": float("nan"), "gate": "info",
        })
    dm_contrib  = PCMU_WEIGHTS["delta_m"]**2  * float(df_dp["delta_m_z"].dropna().var())
    ec_contrib  = PCMU_WEIGHTS["eta_comm"]**2 * float(df_dp["eta_comm_z"].dropna().var())
    var_gate = "ok" if dm_contrib > ec_contrib else "FAIL"
    if var_gate == "FAIL":
        gate_pass = False
    print(f"  Δ_m drives more variance than η_comm: [{var_gate}]")
    gate_rows.append({
        "component": "variance_order", "term": "delta_m > eta_comm",
        "eta_sq": float("nan"), "p": float("nan"), "gate": var_gate,
    })

    # --- Partial correlations (documented ε-coupling) ---
    print("\n[Partial correlations — documented ε-coupling]")
    for x, y, cov, label in [
        ("eta_priv", "eta_comm", "task_count", "r(η_priv, η_comm | task_count)"),
        ("delta_m",  "eta_priv", "task_count", "r(Δ_m, η_priv | task_count)"),
        ("delta_m",  "eta_comm", "task_count", "r(Δ_m, η_comm | task_count)"),
    ]:
        sub = df_dp if "eta_priv" in [x, y] else df
        pr, pv = partial_corr(sub, x, y, cov)
        flag = "documented-finding" if abs(pr) >= PARTIAL_R_THRESHOLD else "ok"
        print(f"  {label}: r={pr:.3f}  p={pv:.3f}  [{flag}]")
        gate_rows.append({
            "component": f"{x}_{y}", "term": f"partial_r|{cov}",
            "eta_sq": float("nan"), "p": round(pv, 4),
            "gate": flag, "partial_r": round(pr, 4),
        })

    print("\n" + "="*60)
    if gate_pass:
        print("✅ PHASE 2 GATE: PASS — additive PCMU validated.")
        print("   Proceed to Phase 3.")
    else:
        print("❌ PHASE 2 GATE: FAIL")
    print("="*60)

    pd.DataFrame(gate_rows).to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")

    print("\n── Mean PCMU by embed_dim (DP rows) ──")
    print(df_dp.groupby("embed_dim")["pcmu"].agg(["mean", "std"]).round(4))
    print("\n── Mean PCMU by ε (DP rows) ──")
    print(df_dp.groupby("epsilon_level")["pcmu"].agg(["mean", "std"]).round(4))
    print("\n── Mean PCMU by task_count (DP rows) ──")
    print(df_dp.groupby("task_count")["pcmu"].agg(["mean", "std"]).round(4))
    print(f"\n── VFL-MTL at ε=∞ mean PCMU (federation overhead) ──")
    nodp = df[~np.isfinite(df["epsilon_level"])]
    print(f"  {nodp['pcmu'].mean():.4f} (< 1.0 reflects cost of federation vs centralized)")


if __name__ == "__main__":
    main()
