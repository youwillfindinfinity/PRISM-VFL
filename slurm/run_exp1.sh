#!/bin/bash
#SBATCH --job-name=prism_exp1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=00:45:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

# Exp 1: VFL-MTL vs single-task baselines (3 seeds: 42, 123, 7)
# Output: results/exp1.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/run_exp1.py \
    --splits_dir data/vertical_splits \
    --output results/exp1.csv \
    --device cuda \
    --save_checkpoint

echo "[exp1] Done. Results in results/exp1.csv"
