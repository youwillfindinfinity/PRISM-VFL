#!/bin/bash
#SBATCH --job-name=prism_validate_bound
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=00:20:00
#SBATCH --output=logs/validate_bound_%j.out
#SBATCH --error=logs/validate_bound_%j.err

# Multi-task label inference bound validation
# Requires: results/privacy_utility_combined.csv, results/label_inference.csv
# Output: results/bound_validation.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/validate_bound.py \
    --results_dir results \
    --output results/bound_validation.csv \
    --device cuda

echo "[validate_bound] Done. Results in results/bound_validation.csv"
