# PRISM-VFL

**Privacy-Resilient Integrated System for Multi-task Vertical Federated Learning**

[![DOI](https://zenodo.org/badge/1250180243.svg)](https://doi.org/10.5281/zenodo.20708652)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Opacus](https://img.shields.io/badge/Differential%20Privacy-Opacus%201.4%2B-5C4B8A.svg)](https://opacus.ai/)
[![Data: MIMIC-III](https://img.shields.io/badge/Data-MIMIC--III-008080.svg)](https://physionet.org/content/mimiciii/1.4/)

A framework for vertical federated multi-task learning (VFL-MTL) on clinical time-series data, with differential privacy and embedding-space privacy attack evaluation. Experiments use the MIMIC-III benchmark with a three-site vertical feature partition.

---

## Contents

```
PRISM-VFL/
├── train.py                     # Main training entry point
├── data/
│   └── DATA_SETUP.md            # Instructions for MIMIC-III access and preprocessing
├── data_prep/
│   ├── vertical_split.py        # MIMIC-III vertical partition into three sites
│   ├── psi_alignment.py         # Patient set intersection and cohort alignment
│   ├── dataset.py               # PyTorch DataLoader builder
│   └── verify_workspace.py      # Workspace integrity checks
├── model/
│   ├── encoder.py               # Per-site LSTM encoder (cut layer)
│   └── mmoe.py                  # Server-side MMoE with per-task heads
├── fl/
│   ├── client.py                # VFL client: embedding send and gradient receive
│   ├── server.py                # VFL server: aggregation, loss, gradient similarity
│   ├── fedavg.py                # FedAvg encoder aggregation
│   └── fedprox.py               # FedProx proximal regularisation
├── privacy/
│   ├── adaptive_dpsgd.py        # Per-task DP-SGD gradient clipping and noise
│   └── renyi_accountant.py      # Renyi DP accounting and coupling matrix
├── attacks/
│   ├── label_inference.py       # Logistic probe on cut-layer embeddings
│   ├── embedding_mia.py         # Membership inference on embeddings
│   └── feature_reconstruction.py# MLP decoder: reconstruct Site A features from z_A
├── baselines/
│   ├── centralized.py           # Centralised oracle
│   └── local_only.py            # Per-site local-only baselines
├── experiments/
│   ├── run_exp1.py              # Exp 1: VFL-MTL vs single-task baselines
│   ├── run_exp2.py              # Exp 2: task relatedness and negative transfer
│   ├── run_exp3.py              # Exp 3: scalability (n_sites in {2, 3})
│   ├── run_ablations.py         # Architecture ablations (MMoE, gating, embed_dim)
│   ├── run_baselines.py         # All baselines in one script
│   ├── run_factorial_pcmu.py    # PCMU Phase 2: 108-cell full factorial
│   ├── privacy_utility_curves.py# Epsilon sweep with Renyi accounting
│   ├── ablations_dp.py          # Uniform vs. stratified noise ablations
│   ├── validate_bound.py        # Multi-task label inference bound
│   ├── compute_pcmu.py          # PCMU metric computation
│   ├── metrics.py               # Per-task metric helpers
│   ├── merge_exp2_rerun.py      # Merge ihm_decomp bug-fix rerun into exp2.csv
│   ├── merge_exp3_rerun.py      # Merge n_sites=2 bug-fix rerun into exp3.csv
│   ├── evaluate_ablations.py    # Test-set evaluation for architecture ablations
│   ├── evaluate_exp2.py         # Test-set evaluation for Exp 2
│   ├── evaluate_exp3.py         # Test-set evaluation for Exp 3
│   ├── evaluate_test.py         # Test-set evaluation for main VFL-MTL results
│   ├── evaluate_test_dp.py      # Test-set evaluation for DP runs
│   ├── evaluate_test_ablations_dp.py # Test-set evaluation for DP ablations
│   └── evaluate_phase4.py       # PCMU Phase 4: sensitivity surface
├── figures/
│   ├── figure4_baselines.py     # Fig 4: baselines comparison
│   ├── figure5a_negative_transfer.py  # Fig 5a: task relatedness heatmap
│   ├── figure5b_scalability.py  # Fig 5b: scalability curves
│   ├── figure6_resilience_variance.py # Fig 6: std(AUC) vs. epsilon (SRQ1)
│   ├── figure7_privacy_utility.py     # Fig 7: AUC vs. epsilon per task (SRQ2)
│   ├── figure8_pcmu.py          # Fig 8: PCMU Cleveland dot plot
│   ├── figS5_label_inference.py # Supp S5: label inference accuracy
│   ├── figS6_ablations.py       # Supp S6: architecture ablations
│   ├── figS7_dp_ablations.py    # Supp S7: DP ablations
│   └── pcmu_vs_epsilon.py       # PCMU vs. epsilon line plot
├── plots/                       # Generated figures (PNG output)
├── results/                     # Experiment result CSVs
├── checkpoints/                 # Saved model checkpoints (generated during training)
├── logs/                        # SLURM job logs (generated during training)
├── slurm/
│   ├── README.md                # Submission order and cluster notes
│   ├── run_exp1.sh
│   ├── run_exp2.sh
│   ├── run_exp3.sh
│   ├── run_ablations.sh
│   ├── run_baselines.sh
│   ├── run_eps_sweep.sh         # Array job: one task per epsilon level
│   ├── run_dp_ablations.sh
│   ├── run_factorial_pcmu.sh    # Array job: one task per embed_dim
│   ├── run_attacks.sh
│   └── run_validate_bound.sh
├── tests/
│   ├── test_fedavg.py
│   ├── test_privacy.py
│   ├── test_centralized.py
│   ├── test_local_only.py
│   └── test_integration.py
├── requirements.txt
├── environment.yml
└── APPLYING_PCMU.md             # How to use the PCMU metric with other frameworks
```

---

## Requirements

Python 3.10 or later is required.

```bash
# Option A — pip
pip install -r requirements.txt

# Option B — conda
conda env create -f environment.yml
conda activate prism-vfl
```

Key dependencies: `torch>=2.2.0`, `opacus>=1.4.0`, `scikit-learn>=1.4`, `scikit-multilearn>=0.2.0`, `SALib>=1.4`.

For GPU training, install a CUDA-compatible build of PyTorch before running `pip install -r requirements.txt`.

---

## Data Setup

Experiments require MIMIC-III (PhysioNet credentialled access). Full setup instructions
including PhysioNet registration, CITI training, download, YerevaNN preprocessing,
vertical split, and PSI alignment are in **`data/DATA_SETUP.md`**.

Quick summary — access is granted after completing CITI training and signing the data
use agreement at https://physionet.org/content/mimiciii/1.4/.

### Step 1: Extract and preprocess MIMIC-III

Use the YerevaNN benchmark pipeline to extract per-stay time-series and create the four task datasets:

```bash
git clone https://github.com/YerevaNN/mimic3-benchmarks.git
cd mimic3-benchmarks
pip install -r requirements.txt

python -m mimic3benchmark.scripts.extract_subjects \
    /path/to/mimiciii/1.4/ data/root/

python -m mimic3benchmark.scripts.validate_events data/root/

python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

python -m mimic3benchmark.scripts.split_train_and_test data/root/

# Create val splits for each task
for task in in-hospital-mortality decompensation length-of-stay phenotyping; do
    python mimic3models/split_train_val.py data/$task/
done

python -m mimic3benchmark.scripts.create_in_hospital_mortality \
    data/root/ data/in-hospital-mortality/

python -m mimic3benchmark.scripts.create_decompensation \
    data/root/ data/decompensation/

python -m mimic3benchmark.scripts.create_length_of_stay \
    data/root/ data/length-of-stay/

python -m mimic3benchmark.scripts.create_phenotyping \
    data/root/ data/phenotyping/
```

### Step 2: Create the vertical split

```bash
python data_prep/vertical_split.py \
    --root /path/to/mimic3-benchmarks/data/ \
    --output data/vertical_splits/
```

This produces three site CSV files:

| File | Features | Task label |
|------|----------|------------|
| site_A_vitals.csv | HR, SBP, DBP, Temp, SpO2, RespRate, GCS total (7) | In-hospital mortality (binary) |
| site_B_labs.csv | Glucose, pH, FiO2, CapRefill (4) | Decompensation (binary, 24h) |
| site_C_composite.csv | Height, Weight, MeanBP (3) | Phenotyping (25 binary ICD codes) |

Three GCS sub-scores (eye, motor, verbal) are excluded to prevent cross-site feature reconstruction via GCS total = eye + motor + verbal.

### Step 3: Align patient sets

```bash
python data_prep/psi_alignment.py \
    --site_a data/vertical_splits/site_A_vitals.csv \
    --site_b data/vertical_splits/site_B_labs.csv \
    --site_c data/vertical_splits/site_C_composite.csv \
    --output data/vertical_splits/aligned_patient_ids.csv
```

### Step 4: Verify

```bash
python data_prep/verify_workspace.py
```

---

## Training

### Basic run

```bash
python train.py --root . --rounds 50 --seed 42
```


```bash
# Uniform noise
python train.py --root . --rounds 100 --seed 42 \
    --privacy-mode uniform --privacy-sigma 1.0

# Task-stratified noise (tighter noise on high-stakes tasks)
python train.py --root . --rounds 100 --seed 42 \
    --privacy-mode stratified \
    --sigma-ihm 0.5 --sigma-decomp 1.0 --sigma-pheno 1.5
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds` | 50 | Training rounds (one pass over training data each) |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Adam learning rate for all components |
| `--hidden-dim` | 128 | LSTM hidden size |
| `--embed-dim` | 64 | Cut-layer embedding dimension |
| `--num-experts` | 4 | MMoE shared expert count |
| `--fedprox-mu` | 0.0 | FedProx proximal coefficient (0 = disabled) |
| `--device` | auto | `cpu` or `cuda` |
| `--seed` | 42 | Random seed |
| `--use-synthetic` | off | Smoke-test with random data (no MIMIC required) |

All results reported in the paper use seeds 42, 123, and 7.

---

## Reproducing Experiments

Run each experiment script from the repository root (`PRISM-VFL/`) after completing data
setup. All results reported in the paper use seeds 42, 123, and 7.

### VFL-MTL experiments

```bash
# Baselines (run first — centralized.csv used by figure4)
python experiments/run_baselines.py --splits_dir data/vertical_splits

# Exp 1: VFL-MTL vs single-task baselines (3 seeds)
python experiments/run_exp1.py --splits_dir data/vertical_splits --output results/exp1.csv

# Exp 2: task relatedness and negative transfer
python experiments/run_exp2.py --splits_dir data/vertical_splits --output results/exp2.csv
# If the ihm_decomp config needs to be rerun (server.py bug fix):
python experiments/run_exp2.py --splits_dir data/vertical_splits \
    --task_config ihm_decomp --output results/exp2_ihm_decomp.csv
python experiments/merge_exp2_rerun.py

# Exp 3: scalability (2 and 3 sites)
python experiments/run_exp3.py --splits_dir data/vertical_splits --output results/exp3.csv
# If n_sites=2 needs rerun with fixed early stopping:
python experiments/run_exp3.py --splits_dir data/vertical_splits \
    --n_sites 2 --output results/exp3_n_sites_2.csv
python experiments/merge_exp3_rerun.py

# Architecture ablations (7 configurations)
python experiments/run_ablations.py --splits_dir data/vertical_splits \
    --output results/ablations.csv

# Test-set evaluation
python experiments/evaluate_test.py --splits_dir data/vertical_splits
python experiments/evaluate_exp2.py --splits_dir data/vertical_splits
python experiments/evaluate_exp3.py --splits_dir data/vertical_splits
python experiments/evaluate_ablations.py --splits_dir data/vertical_splits
```

### Differential privacy experiments

```bash
# Epsilon sweep (one call per level; results appended to combined CSV)
for EPS in 0.5 1.0 2.0 5.0 10.0; do
    python experiments/privacy_utility_curves.py \
        --splits_dir data/vertical_splits \
        --epsilon $EPS \
        --output results/privacy_utility_combined.csv
done

# DP ablations (Abl 1/2/3)
python experiments/ablations_dp.py \
    --splits_dir data/vertical_splits --output results/dp_ablations.csv
python experiments/evaluate_test_ablations_dp.py \
    --splits_dir data/vertical_splits --output results/test_ablations_dp.csv

# Embedding-space attacks (requires checkpoints from eps sweep)
python attacks/label_inference.py \
    --splits_dir data/vertical_splits --ckpt_dir checkpoints \
    --output results/label_inference.csv
python attacks/embedding_mia.py \
    --splits_dir data/vertical_splits --ckpt_dir checkpoints \
    --output results/embedding_mia.csv
python attacks/feature_reconstruction.py \
    --splits_dir data/vertical_splits --ckpt_dir checkpoints \
    --output results/feature_reconstruction.csv

# Multi-task label inference bound validation
python experiments/validate_bound.py \
    --results_dir results --output results/bound_validation.csv
```

### PCMU metric

```bash
# PCMU Phase 2: 108-cell full factorial (36 cells × 3 seeds per embed_dim slice)
python experiments/run_factorial_pcmu.py \
    --splits_dir data/vertical_splits \
    --output results/pcmu_phase2_factorial.csv \
    --rounds_output results/pcmu_phase2_factorial_rounds.csv

# PCMU Phase 4: sensitivity surface
python experiments/evaluate_phase4.py
```

### Running on Snellius (Snellius HPC, SURF)

SLURM scripts for all experiments are in `slurm/`. See `slurm/README.md` for the
recommended submission order.

```bash
mkdir -p logs
sbatch slurm/run_baselines.sh
sbatch slurm/run_exp1.sh
sbatch slurm/run_exp2.sh
sbatch slurm/run_exp3.sh
sbatch slurm/run_ablations.sh
sbatch slurm/run_eps_sweep.sh       # array job: 5 epsilon levels in parallel
sbatch slurm/run_dp_ablations.sh
sbatch slurm/run_factorial_pcmu.sh  # array job: 3 embed_dim slices in parallel
sbatch slurm/run_attacks.sh
sbatch slurm/run_validate_bound.sh
```

---

## Figures

All figure scripts read from `results/` and write PNGs to `plots/`. Run after all
experiments complete.

```bash
# Main paper figures
python figures/figure4_baselines.py
python figures/figure5a_negative_transfer.py
python figures/figure5b_scalability.py
python figures/figure6_resilience_variance.py
python figures/figure7_privacy_utility.py
python figures/figure8_pcmu.py

# Supplementary figures
python figures/figS5_label_inference.py
python figures/figS6_ablations.py
python figures/figS7_dp_ablations.py
python figures/pcmu_vs_epsilon.py
```

| Script | Figure | Source CSV |
|--------|--------|-----------|
| `figure4_baselines.py` | Baselines comparison | `results/centralized.csv`, `results/local_only_*.csv`, `results/exp1.csv` |
| `figure5a_negative_transfer.py` | Task relatedness heatmap | `results/exp2.csv`, `results/test_exp2.csv` |
| `figure5b_scalability.py` | Scalability curves | `results/exp3.csv`, `results/test_exp3.csv` |
| `figure6_resilience_variance.py` | std(AUC) vs. ε (SRQ1) | `results/privacy_utility_combined.csv` |
| `figure7_privacy_utility.py` | AUC vs. ε per task (SRQ2) | `results/privacy_utility_combined.csv` |
| `figure8_pcmu.py` | PCMU Cleveland dot plot | `results/pcmu_paper_results.csv` |
| `figS5_label_inference.py` | Label inference accuracy | `results/label_inference.csv` |
| `figS6_ablations.py` | Architecture ablations | `results/ablations.csv`, `results/test_ablations.csv` |
| `figS7_dp_ablations.py` | DP ablations | `results/dp_ablations.csv`, `results/test_ablations_dp.csv` |

---

## Tests

```bash
pytest tests/ -v
```

Tests cover FedAvg aggregation, DP-SGD clipping behaviour, centralized baseline, local-only baseline, and an integration smoke test with synthetic data.

---

## Vertical Split Protocol

The 17 variables marked STATUS="ready" in the YerevaNN benchmark are partitioned across three simulated hospital sites. Three GCS sub-scores are excluded (they allow reconstruction of GCS total, which is held at Site A). The remaining 14 variables are assigned as follows:

| Site | Variables | Task |
|------|-----------|------|
| A | Heart Rate, Systolic BP, Diastolic BP, Temperature, SpO2, Respiratory Rate, GCS Total | In-hospital mortality |
| B | Glucose, pH, FiO2, Capillary Refill Rate | Decompensation |
| C | Height, Weight, Mean Blood Pressure | Phenotyping (25 ICD codes) |

Each site holds its own task label. The server receives only embedding vectors from the cut layer; no raw features or labels are transmitted.

---

## Architecture

**Per-site client:**
- `SiteEncoder`: LSTM (hidden 128, 2 layers) followed by a linear projection to 64-dimensional embeddings and LayerNorm. Input shape: `(B, T, n_features_at_site)`. Output: `(B, 64)`.

**Server:**
- `MMoEServer`: four shared ExpertMLPs (64-128-64 with ReLU), three per-task softmax gating networks, three task heads (binary sigmoid for IHM and Decompensation; 25-way sigmoid for Phenotyping).
- Input: concatenated site embeddings `(B, 192)`.

**Training protocol:**
1. Each client runs its LSTM encoder and sends a detached 64-dimensional embedding to the server (cut layer).
2. The server concatenates embeddings, runs MMoE, computes weighted BCE losses.
3. The server backpropagates and slices the embedding gradient back to each client.
4. Each client applies the gradient to its encoder.
5. Optional FedAvg aggregation every five rounds (skipped when sites have different input dimensions, as in the default heterogeneous setup).

---

## Privacy Module

Differential privacy is implemented via Opacus. The `AdaptiveDPSGD` class supports two modes:

- **Uniform**: one noise multiplier sigma applied to all task gradients.
- **Stratified**: per-task sigma values (sigma_ihm, sigma_decomp, sigma_pheno) reflecting clinical risk hierarchy.

Privacy accounting uses Renyi DP (Mironov 2017) via `opacus.accountants.RDPAccountant`, with one accountant per task. The `RenyiAccountant` wrapper adds cross-task gradient correlation tracking via `cross_task_coupling_matrix()`.

---

## PCMU Metric

The PCMU (Privacy-Communication-adjusted Multi-task Utility) metric provides a single scalar for comparing configurations that trade off utility, privacy, and communication cost. See `APPLYING_PCMU.md` for how to apply it to other frameworks.

---

## Citation

If you use this codebase, please cite:

> Soare, A. A., Bumbuc, R. V.,  Korkmaz, H. I., & Sheraton, V. M. (2026). PRISM-VFL: A Differentially Private Vertical Federated Framework for Heterogeneous Multi-Task Clinical Prediction. *DOI: to be updated on publication.*

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

---

## License

This repository is licensed under [CC BY 4.0](LICENSE).
