#!/bin/bash
#SBATCH --job-name=prism_baselines
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=00:30:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# Centralized oracle and local-only baselines
# Outputs: results/centralized.csv, results/local_only_{A,B,C}.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/run_baselines.py \
    --splits_dir data/vertical_splits \
    --device cuda

echo "[baselines] Done."
