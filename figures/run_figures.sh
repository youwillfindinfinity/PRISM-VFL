#!/usr/bin/env bash
# run_figures.sh — Regenerate all manuscript figures in order.
#
# Run from PRISM-VFL root:
#   bash figures/run_figures.sh
#
# All PNGs are written to Manuscript/figures/ at DPI=800.
#
# Figure → RQ mapping
# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  (Figure4_RQ.png)              SRQ1 — Multi-task vs. single-task baselines
# S3     (S3_LearningCurves.png)       SRQ1 — Validation learning curves
# Fig 5a (Figure5a_TaskRelatedness.png) SRQ1 — Task relatedness / negative transfer
# Fig 5b (Figure5b_Scalability.png)    Exp3 — 2-site vs. 3-site scalability
# Fig 6  (Figure6_SQR2.png)            SRQ2 — DP stochasticity variance
# Fig 7  (Figure7_SRQ3.png)            SRQ3 — Privacy-utility curves
# Fig 8  (Figure8_PCMU.png)            PCMU — Unified composite ranking
# S5     (S5_LabelInference.png)       SRQ3 — Theoretical bound vs. empirical attacks
# S6     (S6_Ablations.png)            SRQ1 — Architectural ablations
# S7A    (S7A_DP_ABL1.png)             SRQ2/3 — DP ablation 1 (uniform vs. stratified)
# S7B    (S7B_DP_ABL2.png)             SRQ2/3 — DP ablation 2 (task coupling)
# S7C    (S7C_DP_ABL3.png)             SRQ2/3 — DP ablation 3 (embed_dim × ε)
# S8     (S8_PCMUSensitivity.png)      PCMU — Weight sensitivity surface
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."   # ensure we are in PRISM-VFL/

mkdir -p Manuscript/figures

echo "=== Fig 4 + S3: Baselines and learning curves (SRQ1) ==="
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
echo "=== Fig 8: PCMU composite ranking ==="
python3 figures/figure8_pcmu.py

echo ""
echo "=== S5: Label inference bound validation (SRQ3) ==="
python3 figures/figS5_label_inference.py

echo ""
echo "=== S6: Architectural ablations (SRQ1) ==="
python3 figures/figS6_ablations.py

echo ""
echo "=== S7: DP ablations 1, 2, 3 (SRQ2/3) ==="
python3 figures/figS7_dp_ablations.py --abl 1
python3 figures/figS7_dp_ablations.py --abl 2
python3 figures/figS7_dp_ablations.py --abl 3

echo ""
echo "=== S8: PCMU weight sensitivity surface ==="
python3 experiments/evaluate_phase4.py

echo ""
echo "=== Done — all figures written to Manuscript/figures/ ==="
ls -lh Manuscript/figures/*.png 2>/dev/null || echo "(no PNG files found)"
