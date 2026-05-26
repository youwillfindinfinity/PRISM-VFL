#!/bin/bash
#SBATCH --job-name=prism_pcmu_factorial
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=01:00:00
#SBATCH --output=logs/pcmu_factorial_%A_%a.out
#SBATCH --error=logs/pcmu_factorial_%A_%a.err

# PCMU Phase 2: 108-cell full factorial
# Grid: embed_dim {32,64,128} × epsilon {inf,5,1,0.5} × task_config {all,ihm_decomp,ihm_only}
# 3 seeds per cell → 108 runs total, split across 3 array tasks (36 runs each)
# Outputs: results/pcmu_phase2_factorial.csv, results/pcmu_phase2_factorial_rounds.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

# Each array task covers one embed_dim slice (36 cells × 3 seeds / slice)
EMBED_DIMS=(32 64 128)
EMBED_DIM=${EMBED_DIMS[$SLURM_ARRAY_TASK_ID]}

echo "[pcmu_factorial] Running embed_dim=$EMBED_DIM"

python experiments/run_factorial_pcmu.py \
    --splits_dir data/vertical_splits \
    --embed_dim $EMBED_DIM \
    --output results/pcmu_phase2_factorial.csv \
    --rounds_output results/pcmu_phase2_factorial_rounds.csv \
    --device cuda

echo "[pcmu_factorial] Done for embed_dim=$EMBED_DIM"
