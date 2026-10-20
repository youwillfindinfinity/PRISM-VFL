#!/bin/bash
#SBATCH --job-name=prism_attacks
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --output=logs/attacks_%A_%a.out
#SBATCH --error=logs/attacks_%A_%a.err

# Embedding-space attack suite (requires checkpoints from eps sweep).
# One array task per attack, so each gets its own time limit, log and exit status:
#   0  label_inference         — logistic probe AUC on cut-layer embeddings
#   1  embedding_mia           — membership inference AUC across epsilon levels
#   2  feature_reconstruction  — feature reconstruction MSE / R2 (Site A)
# Outputs: results_revision/<attack>.csv
# Rerun selected attacks only, e.g.: sbatch --array=1,2 slurm/run_attacks.sh

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

# Strict mode only after module load, whose scripts reference unset variables.
set -euo pipefail

ATTACKS=(label_inference embedding_mia feature_reconstruction)
ATTACK=${ATTACKS[$SLURM_ARRAY_TASK_ID]}

python -u attacks/${ATTACK}.py \
    --splits_dir data/vertical_splits \
    --ckpt_dir checkpoints_revision \
    --output results_revision/${ATTACK}.csv

echo "[attacks] ${ATTACK} done."
