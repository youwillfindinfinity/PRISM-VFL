#!/bin/bash
#SBATCH --job-name=prism_exp3
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err

# Exp 3: Scalability — n_sites in {2, 3}, 3 seeds each
# Output: results/exp3.csv
#
# Note: n_sites=2 convergence_round column records first trigger (not actual
# termination). Report per-round wall-clock time in text rather than
# convergence_round from CSV for n_sites=3.

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/run_exp3.py \
    --splits_dir data/vertical_splits \
    --output results/exp3.csv \
    --device cuda \
    --save_checkpoint

echo "[exp3] Done. Results in results/exp3.csv"
