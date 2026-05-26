#!/bin/bash
#SBATCH --job-name=prism_ablations
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=01:30:00
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.err

# Architecture ablations: 7 configs × 3 seeds
# Configs: VFL-MTL, abl_no_mmoe, abl_experts_2, abl_experts_8,
#          abl_uniform_gating, abl_embed_32, abl_embed_128
# Output: results/ablations.csv
# Test-set evaluation: python experiments/evaluate_ablations.py

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/run_ablations.py \
    --splits_dir data/vertical_splits \
    --output results/ablations.csv \
    --device cuda \
    --save_checkpoint

echo "[ablations] Done. Results in results/ablations.csv"
echo "[ablations] Run test-set evaluation with:"
echo "  python experiments/evaluate_ablations.py --splits_dir data/vertical_splits"
