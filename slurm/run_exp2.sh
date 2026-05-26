#!/bin/bash
#SBATCH --job-name=prism_exp2
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=01:30:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

# Exp 2: Task relatedness and negative transfer
# Configs: all_tasks / ihm_only / ihm_decomp / ihm_pheno, 3 seeds each
# Output: results/exp2.csv
#
# If the ihm_decomp config needs a rerun (server.py bug fix), run separately:
#   sbatch slurm/run_exp2_ihm_decomp_rerun.sh
#   python experiments/merge_exp2_rerun.py

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python experiments/run_exp2.py \
    --splits_dir data/vertical_splits \
    --output results/exp2.csv \
    --device cuda \
    --save_checkpoint

echo "[exp2] Done. Results in results/exp2.csv"
