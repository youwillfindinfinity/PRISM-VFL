#!/usr/bin/env bash
# run_figures.sh — Regenerate the revision figures in order.
#
# Run from PRISM-VFL root:
#   bash figures/run_figures.sh
#
# Reads results_revision/; all PNGs are written to Manuscript/figures_revision/ at DPI=800.
#
# Figure → RQ mapping
# ─────────────────────────────────────────────────────────────────────────────
# S2     (S2_Cohort.png)               Cohort attrition and acuity bias (reads results_revision/cohort_stats.csv)
# Fig 4 + S3 (Figure4_RQ.png, S3_LearningCurves.png) RQ — Baselines vs. PRISM (best-checkpoint val AUC)
# Fig 5a (Figure5a_TaskRelatedness.png) SRQ1 — Task relatedness / negative transfer
# Fig 5b (Figure5b_Scalability.png)    Exp3 — 2-site vs. 3-site scalability
# Fig 6  (Figure6_SQR2.png)            SRQ2 — DP stochasticity variance
# Fig 7  (Figure7_SRQ3.png)            SRQ3 — Privacy-utility curves
# S5     (S5_LabelInference.png)       SRQ3 — Theoretical bound vs. empirical attacks
# S6     (S6_Ablations.png, ablations_test.png) SRQ1 — Architectural ablations (val, test)
# S7A    (S7A_DP_ABL1.png)             SRQ2/3 — DP ablation 1 (uniform vs. stratified)
# S7B    (S7B_DP_ABL2.png)             SRQ2/3 — DP ablation 2 (task coupling)
# S7C    (S7C_DP_ABL3.png)             SRQ2/3 — DP ablation 3 (embed_dim × ε)
# brier  (brier_nodp.png, brier_dp.png) Calibration curves (Brier score)
# Fig 8  (Figure8_PCMU.png)            PCMU per ε (pcmu rows of prism_privacy_sweep_stats.csv)
# S8     (S8_PCMUSensitivity.png)      PCMU Morris/Sobol + weight sensitivity (experiments/evaluate_phase4.py)
#
# Fig 6, 7 and 8 read results_revision/prism_privacy_sweep_stats.csv from
# revision_scripts/analyse_prism_privacy_sweep.py; rerun it after the PCMU factorial changes.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."   # ensure we are in PRISM-VFL/

mkdir -p Manuscript/figures_revision

echo "=== S2: Cohort attrition ==="
python3 figures/figS2_cohort.py

echo ""
echo "=== Fig 4 + S3: Baselines vs. PRISM (RQ) ==="
python3 figures/figure4_baselines.py

echo ""
echo "=== Fig 5a: Task relatedness / negative transfer (SRQ1) ==="
python3 figures/figure5a_negative_transfer.py

echo ""
echo "=== Fig 5b: Scalability — 2 vs 3 institutions ==="
python3 figures/figure5b_scalability.py

echo ""
echo "=== Fig 6: DP resilience variance (SRQ2) ==="
python3 figures/figure6_resilience_variance.py

echo ""
echo "=== Fig 7: Privacy-utility curves (SRQ3) ==="
python3 figures/figure7_privacy_utility.py

echo ""
echo "=== S5: Label inference bound validation (SRQ3) ==="
python3 figures/figS5_label_inference.py

echo ""
echo "=== S6: Architectural ablations (SRQ1) ==="
python3 figures/figS6_ablations.py
python3 figures/figS6_ablations.py --source test

echo ""
echo "=== S7: DP ablations 1, 2, 3 (SRQ2/3) ==="
python3 figures/figS7_dp_ablations.py --abl 1
python3 figures/figS7_dp_ablations.py --abl 2
python3 figures/figS7_dp_ablations.py --abl 3

echo ""
echo "=== Calibration curves (Brier score) ==="
python3 figures/plot_brier.py

echo ""
echo "=== Fig 8: PCMU per privacy budget ==="
python3 figures/figure8_pcmu.py

echo ""
echo "=== S8: PCMU sensitivity (Morris/Sobol, weights) ==="
python3 experiments/evaluate_phase4.py

echo ""
echo "=== Done — all figures written to Manuscript/figures_revision/ ==="
ls -lh Manuscript/figures_revision/*.png 2>/dev/null || echo "(no PNG files found)"
