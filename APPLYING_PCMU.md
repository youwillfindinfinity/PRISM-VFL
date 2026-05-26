# Applying the PCMU Metric to Other Frameworks

This document explains the **PCMU (Privacy-Communication-adjusted Multi-task Utility)** metric, what it measures, and how to compute it for a VFL-MTL system other than the one in this repository.

The metric is defined and implemented in `experiments/compute_pcmu.py`. This guide is intended for readers who want to apply it to a different federated learning framework, different datasets, or a different set of tasks.

---

## What PCMU Measures

PCMU is a single-score summary for comparing configurations in vertical federated multi-task learning when three objectives are in tension:

1. **Multi-task utility gain** (delta_m): does the multi-task setup produce better predictions than training each task independently at its own site?
2. **Privacy efficiency** (eta_priv): how much prediction utility is retained under differential privacy, relative to a no-DP reference?
3. **Communication efficiency** (eta_comm): how quickly does the system converge, relative to a reference round count?

Higher PCMU is better. A value of 1.0 corresponds to the no-DP baseline (epsilon = infinity).

### Formula

```
PCMU = w1 * delta_m_z + w2 * eta_priv_z + w3 * eta_comm_z - PCMU_baseline + 1.0
```

where `_z` denotes z-scoring over the full evaluation pool (all configurations under comparison), and `PCMU_baseline` is the score of the no-DP run before the shift. The shift anchors the no-DP baseline at 1.0.

Default component weights (from Rieke et al. 2020 clinical deployment priority hierarchy):

| Component | Weight | Rationale |
|-----------|--------|-----------|
| delta_m | 0.70 | Multi-task utility is the primary objective |
| eta_priv | 0.20 | Privacy is a hard constraint; utility degradation matters |
| eta_comm | 0.10 | Communication cost is secondary |

These weights can be changed via `PCMUConfig.pcmu_weights`.

### Why additive and not multiplicative

A multiplicative form `(1 + delta_m) * eta_priv * eta_comm` compounds the shared variance introduced by epsilon, which drives all three components simultaneously. Phase 2 ANOVA (eta_squared >= 0.12 for embed_dim x epsilon interactions) confirmed that additive aggregation with z-score normalisation is the correct choice here. The geometric form is archived as `compute_pcmu_geometric()` for comparison; it was the original formulation and was superseded after that analysis.

---

## Component Definitions

### delta_m: multi-task gain

```
delta_m = (1/T) * sum_t [ w_t * (M_t^MTL - M_t^ST) / M_t^ST ]
```

where:
- `T` is the number of tasks with non-NaN values in both the MTL and single-task results
- `w_t` is the per-task weight (default: IHM=0.5, Decomp=0.3, Pheno=0.2, sum=1.0)
- `M_t^MTL` is the peak validation AUC-ROC of the VFL-MTL model on task t
- `M_t^ST` is the peak validation AUC-ROC of the single-task VFL baseline on task t

This formulation follows Maninis et al. (CVPR 2019).

Positive delta_m means the multi-task setup improves on single-task. Negative means negative transfer.

### eta_priv: privacy efficiency

```
eta_priv = sum_t [ w_t * (M_t / M_t^nodp) ]
```

where:
- `M_t` is the peak validation AUC-ROC at the current epsilon level
- `M_t^nodp` is the peak validation AUC-ROC at epsilon = infinity (no DP)

This is a plain weighted utility ratio relative to the no-DP reference. At epsilon = infinity it equals 1.0; it falls as DP degrades utility.

Note: eta_priv is undefined (NaN) when epsilon = infinity, so those rows are excluded from z-scoring for this component.

### eta_comm: communication efficiency

```
eta_comm = log(1 + R_ref / R)
```

where:
- `R` is the convergence round of the current configuration
- `R_ref` is the convergence round of the no-DP baseline

Convergence round is defined as the first round where the validation AUC-ROC reaches 90% of its maximum over the full run.

Lower R (faster convergence) gives higher eta_comm. At `R = R_ref` (same convergence speed as the baseline) the value is `log(2) = 0.693`.

---

## Required Inputs

To compute PCMU for a new framework or dataset, you need the following CSV files:

### 1. exp1.csv (or equivalent)

A results file from the multi-task and single-task runs. Required columns:

| Column | Description |
|--------|-------------|
| model | String identifier: "VFL-MTL", "ST-IHM", "ST-Decomp", "ST-Pheno" |
| seed | Integer seed (e.g. 42, 123, 7) |
| round | Integer round index (starting from 1) |
| val_ihm_auroc | IHM validation AUC-ROC at this round |
| val_decomp_auroc | Decompensation validation AUC-ROC |
| val_pheno_macro_auroc | Phenotyping macro validation AUC-ROC |

The single-task baselines must use the same VFL setup as the MTL model (same site encoders, same cut layer) but with only one task active. This isolates the MTL contribution from the VFL contribution.

### 2. privacy_utility_combined.csv

A results file from the epsilon sweep. Required columns:

| Column | Description |
|--------|-------------|
| epsilon_level | float, use `float('inf')` for no-DP |
| mode | "uniform" or "stratified" |
| seed | Integer |
| round | Integer |
| val_ihm_auroc | IHM validation AUC-ROC |
| val_decomp_auroc | Decompensation validation AUC-ROC |
| val_pheno_macro_auroc | Phenotyping macro AUC-ROC |

### 3. centralized.csv (optional)

Centralized oracle results, used only for reporting context. If you do not have a centralised baseline, you can skip this file; it is not used in the PCMU formula itself.

---

## Quick Start

```python
import pandas as pd
from experiments.compute_pcmu import evaluate_sweep, PCMUConfig

exp1_df    = pd.read_csv("results/exp1.csv")
privacy_df = pd.read_csv("results/privacy_utility_combined.csv")

# Default config: task weights IHM=0.5, Decomp=0.3, Pheno=0.2
# PCMU weights: delta_m=0.70, eta_priv=0.20, eta_comm=0.10
cfg = PCMUConfig()

result_df = evaluate_sweep(privacy_df, exp1_df, mode="uniform", config=cfg)

print(result_df.groupby("epsilon_level")[["delta_m", "eta_priv", "eta_comm", "pcmu"]].mean())
```

Output columns in `result_df`:

| Column | Description |
|--------|-------------|
| epsilon_level | Epsilon value (inf for no DP) |
| seed | Seed for this row |
| delta_m | Raw multi-task gain |
| eta_priv | Raw privacy efficiency |
| eta_comm | Raw communication efficiency |
| delta_m_z | Z-scored delta_m over pool |
| eta_priv_z | Z-scored eta_priv over pool |
| eta_comm_z | Z-scored eta_comm over pool |
| pcmu_additive | Weighted sum before baseline shift |
| pcmu | Final PCMU (shifted so no-DP = 1.0) |
| pcmu_geometric | Archived geometric form for comparison |

---

## Adapting to a Different Framework

### Different task set

If your system has fewer or more tasks than three, change `PCMUConfig.task_weights` so that it covers only the tasks you have and sums to 1.0:

```python
cfg = PCMUConfig(
    task_weights={"ihm": 0.6, "decomp": 0.4},   # two-task system
    pcmu_weights={"delta_m": 0.70, "eta_priv": 0.20, "eta_comm": 0.10},
)
```

The `multitask_gain()` and `privacy_efficiency()` functions skip any task that has NaN in either input dict, so missing tasks degrade gracefully to fewer terms in the average.

### Different metric (not AUC-ROC)

If your tasks use a different primary metric (e.g. Cohen's kappa for regression, F1 for detection), update the `_TASK_AUC_COLS` mapping at the top of `compute_pcmu.py` to point to the correct columns in your results CSV:

```python
_TASK_AUC_COLS = {
    "task1": "val_task1_auc",
    "task2": "val_task2_kappa",
}
```

The PCMU formulas use ratios and differences of the metric, so any bounded metric on the same scale works. Metrics that can be negative (e.g. raw kappa) will produce unexpected signs in delta_m; normalise to [0, 1] first if that is a concern.

### Different DP mechanism

PCMU does not depend on the DP mechanism itself; it only uses the resulting AUC-ROC values and convergence round at each epsilon level. As long as your results CSV contains `epsilon_level` and validation metric columns, the computation is the same. The Renyi accountant in `privacy/renyi_accountant.py` is used by the training loop to compute per-task epsilon; it is not required for PCMU post-hoc evaluation.

### No DP baseline

If your framework does not use differential privacy, compute only the multi-task gain component:

```python
from experiments.compute_pcmu import multitask_gain, PCMUConfig

cfg = PCMUConfig()
dm = multitask_gain(
    mtl_aurocs={"ihm": 0.782, "decomp": 0.712, "pheno": 0.620},
    st_aurocs={"ihm": 0.795, "decomp": 0.701, "pheno": 0.612},
    weights=cfg.task_weights,
)
print(f"delta_m = {dm:.4f}")
```

### Single-task frameworks

For a single-task VFL framework, `delta_m` is undefined (the whole point is to compare MTL to single-task). You can still use `eta_priv` and `eta_comm` independently as privacy-utility and communication-efficiency scores; just set the corresponding weight to zero and normalise the remaining weights.

---

## Component weights

The default weights (delta_m=0.70, eta_priv=0.20, eta_comm=0.10) reflect the clinical deployment priority hierarchy from Rieke et al. (2020). For non-clinical applications or systems where communication cost is the primary constraint, you may want to increase the weight on eta_comm:

```python
cfg = PCMUConfig(
    task_weights={"task1": 0.5, "task2": 0.5},
    pcmu_weights={"delta_m": 0.40, "eta_priv": 0.20, "eta_comm": 0.40},
)
```

The sum of `pcmu_weights` must equal 1.0; `PCMUConfig.__post_init__` will raise a `ValueError` otherwise.

---

## Z-scoring and the evaluation pool

The z-scoring step normalises each component over the full set of (epsilon, seed) combinations being compared. This means:

1. You must have at least two distinct epsilon levels (including the no-DP baseline at infinity) to get a non-trivial z-score.
2. Adding configurations to the pool changes the z-score parameters and therefore all PCMU values. Fix the pool once you have the final set of configurations you want to compare, and do not add runs afterwards without recomputing from scratch.
3. If you are comparing two separate systems, run `evaluate_sweep()` on each separately. Cross-system comparisons using the same z-score pool are possible but require careful interpretation: the z-score absorbs variance from both systems, and the resulting PCMU values are not on a common scale.

---

## Task-relatedness

The file also provides Hellinger-distance utilities for measuring how related task label distributions are:

```python
from experiments.compute_pcmu import hellinger_binary, task_relatedness_from_prevalences

# Pairwise Hellinger distance between Bernoulli label distributions
d = hellinger_binary(p=0.10, q=0.25)   # 0.0 = identical, 1.0 = maximally different

# Pairwise distances across multiple tasks
dists = task_relatedness_from_prevalences({
    "ihm":    0.10,
    "decomp": 0.25,
    "pheno":  np.array([0.05, 0.12, ...]),   # multi-label: array of per-label prevalences
})
```

Lower Hellinger distance means more similar label distributions (a proxy for task relatedness). This is used in ablation 2 to characterise the related-task pair (IHM + Decomp) vs. the unrelated pair (IHM + Pheno).

---

## Reference

Rieke, N., et al. (2020). The future of digital health with federated learning. *npj Digital Medicine*, 3, 119.

Maninis, K.-K., et al. (2019). Attentive single-tasking of multiple tasks. *CVPR 2019*.

Mironov, I. (2017). Renyi differential privacy. *CSF 2017*.
