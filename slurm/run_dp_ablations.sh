#!/bin/bash
#SBATCH --job-name=prism_dp_ablations
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=01:00:00
#SBATCH --output=logs/dp_ablations_%j.out
#SBATCH --error=logs/dp_ablations_%j.err

# DP ablations (Abl 1/2/3):
#   Abl 1: uniform vs task-stratified sigma at epsilon_total=5
#   Abl 2: related task pair (IHM+Decomp) vs unrelated (IHM+Pheno) — label inference
#   Abl 3: embed_dim in {32, 64, 128} × epsilon in {1, 5, inf}
# Output: results/dp_ablations.csv (Abl 1 val metrics)
#
# Test-set inference for Abl 2+3:
#   python experiments/evaluate_test_ablations_dp.py \
#       --splits_dir data/vertical_splits --output results/test_ablations_dp.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/ablations_dp.py \
    --splits_dir data/vertical_splits \
    --output results/dp_ablations.csv \
    --device cuda \
    --save_checkpoint

echo "[dp_ablations] Done. Results in results/dp_ablations.csv"
