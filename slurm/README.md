# SLURM Scripts — Snellius (SURF)

These scripts run PRISM-VFL experiments on the Snellius HPC cluster using the
`gpu_a100` partition. Adjust `--account`, `--partition`, and module versions to
match your allocation and cluster configuration.

All scripts assume they are submitted from the `PRISM-VFL/` root directory:

```bash
cd /path/to/PRISM-VFL
sbatch slurm/<script>.sh
```

---

## Prerequisites

1. Data must be set up (see `data/DATA_SETUP.md`).
2. Create a `logs/` directory before submitting:
   ```bash
   mkdir -p logs
   ```
3. Install Python dependencies once (interactive node or prologue):
   ```bash
   pip install --user opacus>=1.4.0 scikit-multilearn>=0.2.0 SALib>=1.4
   ```

---

## Script overview

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `run_exp1.sh` | Exp 1: VFL-MTL vs single-task baselines (3 seeds) | 45 min |
| `run_exp2.sh` | Exp 2: task relatedness and negative transfer (3 seeds × 4 configs) | 90 min |
| `run_exp3.sh` | Exp 3: scalability — n_sites ∈ {2, 3} (3 seeds each) | 60 min |
| `run_ablations.sh` | Architecture ablations (7 configs × 3 seeds) | 90 min |
| `run_baselines.sh` | Centralized oracle and local-only baselines | 30 min |
| `run_eps_sweep.sh` | ε sweep — one array job per ε level (5 jobs × 3 seeds) | 60 min each |
| `run_dp_ablations.sh` | DP ablations (Abl 1/2/3) | 60 min |
| `run_factorial_pcmu.sh` | PCMU Phase 2: 108-cell full factorial | 3 × 60 min array |
| `run_attacks.sh` | Label inference + MIA attacks across ε levels | 30 min |
| `run_validate_bound.sh` | Multi-task label inference bound validation | 20 min |

---

## Order of execution

Run in this order to reproduce all results:

1. `run_baselines.sh`
2. `run_exp1.sh`
3. `run_exp2.sh` → then `python experiments/merge_exp2_rerun.py` if ihm_decomp rerun needed
4. `run_exp3.sh` → then `python experiments/merge_exp3_rerun.py` if n_sites=2 rerun needed
5. `run_ablations.sh`
6. `run_eps_sweep.sh` (submit all five ε levels)
7. `run_dp_ablations.sh`
8. `run_factorial_pcmu.sh`
9. `run_attacks.sh` (requires checkpoints from step 6)
10. `run_validate_bound.sh`

After all jobs complete, generate figures:
```bash
python figures/figure4_baselines.py
python figures/figure5a_negative_transfer.py
python figures/figure5b_scalability.py
python figures/figure7_privacy_utility.py
python figures/figure6_resilience_variance.py
python figures/figS6_label_inference.py
python figures/figS7_ablations.py
python figures/figS8_dp_ablations.py
python figures/figure8_pcmu.py
```
