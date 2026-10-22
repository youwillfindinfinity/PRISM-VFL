# T17 Manuscript Update Map

Maps each manuscript statement that changes to its revised value and source file.
Manuscript: `revision_docs/PRISM-VFL_v1_manuscript.md` (line numbers `L…`). Results: `results_revision/`.
Values are mean ± SD over 10 seeds {7, 42, 123, 0–6}, test set unless marked *val*.

Status: **update** = replace value · **rewrite** = finding changed, text must be reinterpreted (author decision) · **pending** = waits on the PCMU decision · **verify** = check before use.

## 1. Global changes

| Item | Manuscript | Revised | Source | Status |
|---|---|---|---|---|
| Seeds | {42, 123, 7}; +{0, 1, 2} for Sim 1 (L668–670, L749–750, L851–855) | 10 seeds {7, 42, 123, 0, 1, 2, 3, 4, 5, 6} for every simulation | `experiment_config.py` | update |
| Statistics | mean ± SD, no CIs (L855) | mean ± SD, bootstrap 95% CI, paired Δ per seed with sign test | `prism_privacy_sweep_stats.csv`, `prism_privacy_sweep_summary.md` | update |
| Task weighting | equal (Eq. 3), but uncertainty weighting in the DP sweep (Limitations L1293–1297) | equal weights in every run; uncertainty weighting only as a labelled side arm (ε = ∞, 5) | T11, `prism_privacy_sweep.csv` (`weighting`) | update; delete the limitation |
| Checkpoint policy | best mean val AUC, patience 15, max 100 (L509–512) | unchanged, now implemented in every run and evaluator (previously best-IHM checkpoint was loaded) | `audit_revision/checkpoint_policy_audit.json` | update (text OK; mention in response letter) |
| DP sweep stopping | not stated (runs used 100 rounds, no early stopping) | patience 15 for ε = ∞ and all DP levels | `dp_accounting_audit.json` | update |
| Baseline training budget | not stated (local-only and centralised capped at 50 epochs) | max 100 epochs, as for PRISM; no baseline reaches the cap | `slurm/run_baselines.sh`, manifest | update |
| Seed-42 Decomp collapse | central caveat (L873–887, L1154–1161, L1242–1251, Conclusion) | does not reproduce: Decomp 0.924 ± 0.005; seed-42 probes 0.917–0.926 (val) | `seed42_probes.csv`, `test_results.csv` | rewrite (remove all seed-42 text) |
| Figure numbering | Fig 4–7, S3–S9 | files in `Manuscript/figures_revision/` (see §6) | `figures/run_figures.sh` | update |

## 2. Methods

| Section | Manuscript | Revised | Source | Status |
|---|---|---|---|---|
| 2.3.3 Alignment (L407–410) | "18,090 unique patients, each containing one ICU stay" | 18,090 patients, 21,135 aligned ICU stays (1.17 stays/patient); every site trains on the same stays in the same order | `alignment_audit_before_fix.txt` | update |
| 2.3.3 Split (L415–418) | 70/15/15 | 14,679 / 3,221 / 3,235 stays (12,563 / 2,765 / 2,762 patients) | same | update |
| 2.3.3 / 2.4.3 Alignment mechanism | patient-level PSI only | add: stay-level intersection, canonical stay order, shared shuffle per epoch, batch-level stay-ID assertions | T05–T08, `smoke_test_report.json` | update |
| 2.3.1 / 2.3.3 Decomp positive rate 10.1%; class weight | — | Decomp pos_weight 8.0555 from aligned training labels; recompute positive rates on the aligned stays | T09 | verify |
| 2.4.1 Baselines (L440–446) | local-only and oracle described as lower/upper bound | local-only trains on all of the site's own stays (B 17,516, C 17,533 train) and is tested on the aligned 3,235 test stays; the oracle trains on the aligned 14,679 stays with one LSTM over all 14 features, i.e. an upper bound on data access, not on model capacity | `baselines/*.py`, Snellius log 28457902 | update |
| 2.4.3 Training (L489–512) | — | add synchronized sampling; checkpoint = best mean val AUROC over active tasks | T06, T10 | update |
| 2.4.5 / Table 2 DP (L539–604) | N_train A/B/C = 14,679 / 17,516 / 17,533 | 14,679 at every site; q = 1/229 = 0.00437 (q = 1 / batches per round) | `dp_accounting_audit.json` | update |
| 2.4.5 DP | stratified "ε_total = 5" | state per-task budgets 2 / 2 / 1 | same | update |
| 2.4.5 DP | — | report ε actually spent at the stopping round next to the target (early stopping now on) | `prism_privacy_sweep.csv` (`epsilon_*`) | update |
| Table 3 (L718–750) | B = 256, q = 256/18,094 ≈ 0.014 | B = 64, q = 1/229 ≈ 0.00437 | `dp_accounting_audit.json` | update |
| Table 3 | blank rows: max rounds, embed dim, experts, B, ε_total | 100; 64; 4; 64; σ per ε = 0.690 / 0.908 / 1.592 / 2.793 / 5.156 (ε = 10 / 5 / 2 / 1 / 0.5) | `experiment_config.py`, `bound_validation.csv` | update |
| 2.5 Compute (L671–676) | 5.54 s/round (3-site), 4.18 s (2-site), 9.4 GPU-h | 4.71 s/round (3-site), 3.50 s (2-site); total GPU-h from SLURM job records | `exp3.csv` (`elapsed_s`), manifest | update; GPU-h verify |
| 2.6.4 PCMU (L813–849), S1 Text (L1378–1425) | η_comm via 90%-of-max convergence round; 108-cell factorial, 3 seeds | 360-run factorial (3 embed × 4 ε × 3 configs × 10 seeds); η_comm definition per co-author decision | `pcmu_phase2_factorial.csv` | pending |

## 3. Results

| Section | Manuscript | Revised | Source | Status |
|---|---|---|---|---|
| 3.1 E1 PRISM (L872) | IHM 0.781, Decomp 0.632 ± 0.116, Pheno 0.613 | IHM 0.891 ± 0.006, Decomp 0.924 ± 0.005, Pheno 0.662 ± 0.007 (AUPRC 0.730 / 0.778) | `test_results.csv` | update |
| 3.1 E1 single-task | — | ST-IHM 0.890 ± 0.004, ST-Decomp 0.923 ± 0.003, ST-Pheno 0.708 ± 0.001 | same | update |
| 3.1 Multi-task gain Δ (L878–882) | Decomp +0.093, IHM +0.003, Pheno +0.002; no negative transfer | IHM +0.001 [−0.003, 0.005], Decomp +0.000 [−0.002, 0.003], Pheno −0.046 [−0.051, −0.042], negative in 10/10 seeds (sign test p = 0.002) → negative transfer on Phenotyping | `prism_privacy_sweep_summary.md` | rewrite |
| 3.1 Oracle gap δ (L888) | 0.099 / 0.286 / 0.041 | −0.010 / −0.004 / −0.014: PRISM is above the centralised oracle on all tasks (oracle 0.881 ± 0.004 / 0.919 ± 0.004 / 0.648 ± 0.008), all within the 0.02 practical threshold | `test_results.csv` | rewrite (PRISM matches the oracle; oracle = data-access bound) |
| 3.1 Local-only | — | Local A 0.781 ± 0.003, B 0.681 ± 0.005, C 0.615 ± 0.002 | same | update |
| 3.1 Calibration, Brier no-DP (L889–899) | IHM PRISM 0.086 / ST 0.087 / Local 0.086 / oracle 0.056; Decomp 0.245 / 0.280 / 0.270 / 0.101 | IHM 0.053 / 0.054 / 0.087 / 0.056; Decomp 0.085 / 0.094 / 0.266 / 0.094 | `predictions_nodp.csv` | update |
| Fig 4 caption (L901–907) | val AUC, 6 seeds | val AUC at each model's best-checkpoint epoch, 10 seeds: PRISM 0.878 / 0.928 / 0.661, ST 0.881 / 0.928 / 0.706, oracle 0.870 / 0.921 / 0.644, local 0.794 / 0.711 / 0.623 | `figures/figure4_baselines.py` | update |
| 3.2 E2 (L908–926, Fig 5) | Decomp 0.713 (2-task) → 0.632 (3-task); stable IHM/Pheno | IHM 0.890 (IHM-only), 0.888 (+Decomp), 0.894 (+Pheno), 0.891 (all); Decomp 0.925 (+IHM) vs 0.924 (all); Pheno 0.692 (+IHM) vs 0.662 (all) → Pheno drops when Decomp is added | `test_exp2.csv` | rewrite |
| 3.3 E3 (L940–951) | 60.3 → 70.0 rounds; IHM 0.781 at 2 and 3 sites | rounds to best checkpoint 48.1 ± 16.5 (2-site) vs 33.5 ± 6.8 (3-site); IHM 0.797 ± 0.008 vs 0.891 ± 0.006; Decomp 0.829 vs 0.924 | `exp3.csv`, `test_exp3.csv` | rewrite; verify the 2-site IHM drop |
| 3.4 E4 stability (L958–964) | IHM SD peaks 0.122 at ε = 1 | SD 0.002–0.013 at every ε, no trend | `prism_privacy_sweep_stats.csv` | rewrite |
| 3.5 E5 utility by ε (L998–1008) | IHM 0.768 at ε = 2; ε*_IHM ∈ (1, 2); Decomp/Pheno below floors | IHM 0.891 / 0.863 / 0.852 / 0.834 / 0.822 / 0.805; Decomp 0.924 / 0.909 / 0.901 / 0.887 / 0.874 / 0.858; Pheno 0.662 / 0.619 / 0.610 / 0.587 / 0.567 / 0.565 (ε = ∞ / 10 / 5 / 2 / 1 / 0.5) | same | rewrite |
| 3.5 ε* (utility floors IHM 0.75, Pheno 0.65, Decomp 0.70 reference) | IHM (1, 2); others none | IHM above floor at every tested ε (ε* < 0.5); Decomp above 0.70 at every ε; Pheno above 0.65 only without DP (ε* ∈ (10, ∞)) | same | rewrite |
| 3.5 Stratified vs uniform at ε = 5 | no recovery | stratified 0.825 / 0.880 / 0.569 vs uniform 0.852 / 0.901 / 0.610: lower on all tasks | same | update |
| 3.5 Brier under DP (L1009–1013) | IHM 0.086 (∞), 0.088 (2), 0.098 (1), 0.102 (5); Decomp 0.245 → 0.283 | IHM 0.053 / 0.056 / 0.057 / 0.058 / 0.061 / 0.062; Decomp 0.085 / 0.106 / 0.098 / 0.090 / 0.091 / 0.110 (ε = ∞ / 10 / 5 / 2 / 1 / 0.5) | `predictions_dp.csv` | update |
| 3.6 E6 label inference (L1018–1024) | IHM 0.498 at ε = 0.5; bound exceeded at ε = 1, 2 by 0.013 / 0.030 | IHM 0.764 / 0.732 / 0.701 / 0.656 / 0.649 / 0.575 ± 0.090 (ε = ∞ … 0.5); bound (ρ = 0.1) 0.646 at ε = 1 exceeded by 0.003, all other levels within the bound | `label_inference.csv`, `bound_validation.csv` | rewrite |
| 3.6 MIA | ≈ 0.50 at every ε | 0.499–0.512 at every ε (0.512 ± 0.016 at ε = ∞) | `embedding_mia.csv` | update |
| 3.6 Feature reconstruction | "results in repository" | mean R² 0.601 (ε = ∞) → 0.316 (ε = 0.5) | `feature_reconstruction.csv` | update (consider reporting) |
| 3.7.1 Architecture ablations (L1045–1053) | no-MMoE / uniform gating −0.019 Pheno; ablations remove seed-42 failure | Pheno: no-MMoE 0.649 (−0.013), uniform gating 0.653 (−0.009), experts 2/8 0.658 / 0.665, embed 32/128 0.661 / 0.664; IHM and Decomp within ±0.004 of PRISM | `test_ablations.csv` | rewrite |
| 3.7.2 DP-1 (L1056–1057) | stratified −0.045 Pheno | −0.041 Pheno, −0.027 IHM, −0.021 Decomp (*val*) | `dp_ablations.csv` | update |
| 3.7.2 DP-2 coupling ρ | −0.009 to −0.030 | 0.037 (IHM+Decomp) vs 0.028 (IHM+Pheno); label inference AUC 0.712 vs 0.765 | `dp_ablations.csv`, `test_ablations_dp.csv` | rewrite ("high/low ρ" labels no longer apt) |
| 3.7.2 DP-3 (L1063–1065) | ε = 1: dim-32 IHM 0.766 vs dim-128 0.537 | ε = 1: IHM 0.818 (32), 0.822 (64), 0.812 (128) → no capacity effect | `test_ablations_dp.csv` | rewrite |
| 3.8 Fairness (L1066–1076) | 46% attrition; Decomp 10.1% vs 6.6% | recompute rates on aligned stays | — | verify |
| 3.9 PCMU, Fig 7 (L1077–1097) | −0.652 (ε = ∞), −2.725 (ε = 0.5) | — | — | pending |

## 4. Abstract, Author summary, Discussion, Conclusion

| Section | Statements to change | Status |
|---|---|---|
| Abstract (L66–95) | AUCs 0.781 / 0.632 ± 0.116 / 0.613; ε* ∈ (1, 2); "Decomp and Pheno do not reach floors"; label inference 0.498 at ε = 0.5 | rewrite |
| Author summary (L98–119) | "decompensation and phenotyping require further development" | rewrite |
| Discussion, main RQ (L1118–1135) | δ_IHM = 0.099; IHM 0.781 vs APACHE/SOFA; "IHM only task reaching floor" | rewrite |
| SRQ1 (L1140–1184) | +0.093 Decomp; MARS-VFL Pheno comparison 61.3%; seed-42; ρ; ablation −0.019; scalability 60.3 → 70.0 | rewrite |
| SRQ2 (L1185–1193) | IHM SD 0.122 at ε = 1; non-monotonic | rewrite |
| SRQ3 (L1194–1221) | ε* (1, 2); label inference 0.498 ± 0.045 | rewrite |
| Ablations + PCMU paragraph (L1222–1232) | dim-32 vs dim-128; seed-42; PCMU −0.652 / −2.725 | rewrite; PCMU pending |
| Limitations (L1233–1297) | seed-42 paragraph; uncertainty-weighting confound; 0.498 | rewrite (remove both) |
| Conclusion (L1311–1338) | Decomp gain; no negative transfer; ε* (1, 2); MARS-VFL match; PCMU | rewrite; PCMU pending |
| End of Results summary (L1098–1116) | "no negative transfer", routing drives benefit, ε ≈ 2 | rewrite |

## 5. Supporting information

| Item | Change | Status |
|---|---|---|
| S1 Text | factorial size, Phase 1–4 numbers, η_comm definition | pending |
| S3 Fig | 10 seeds; curves for all models | update |
| S4 / S5 Fig | Brier values (§3) | update |
| S6 Fig (bound) | new values (§3.6) | update |
| S7 Fig (architecture ablations) | 10 seeds; new values | update |
| S8 Fig (DP ablations A–C) | new values; file names `results_revision/dp_ablations.csv` | update |
| S9 Fig (PCMU weight sensitivity) | — | pending |

## 6. Figure files (`Manuscript/figures_revision/`)

| Manuscript | File |
|---|---|
| Fig 4 / S3 | `Figure4_RQ.png` / `S3_LearningCurves.png` |
| Fig 5 (A–B / C) | `Figure5a_TaskRelatedness.png` / `Figure5b_Scalability.png` |
| Fig 6 (A / B) | `Figure6_SQR2.png` / `Figure7_SRQ3.png` |
| Fig 7 (PCMU) | pending |
| S4 / S5 | `brier_nodp.png` / `brier_dp.png` |
| S6 | `S5_LabelInference.png` |
| S7 | `S6_Ablations.png` (val), `ablations_test.png` (test) |
| S8 A–C | `S7A_DP_ABL1.png`, `S7B_DP_ABL2.png`, `S7C_DP_ABL3.png` |
| S9 | pending |

## 7. Response letter items

- R1-Q4: the published ε = ∞ point was the early-stopped Exp 1 model, compared against fixed 100-round DP runs; the revised sweep uses the same seeds and patience 15 at every ε (`dp_accounting_audit.json`).
- Alignment defect (T02–T05), checkpoint-loading mismatch (T10), baseline 50-epoch cap (now 100, as for PRISM) and η_comm baseline mismatch (code used no-DP instead of R_ST/R_MTL): issues we found and fixed ourselves.
