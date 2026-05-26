#!/bin/bash
#SBATCH --job-name=prism_attacks
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=00:30:00
#SBATCH --output=logs/attacks_%j.out
#SBATCH --error=logs/attacks_%j.err

# Embedding-space attack suite (requires checkpoints from eps sweep)
# Outputs:
#   results/label_inference.csv   — logistic probe AUC on cut-layer embeddings
#   results/embedding_mia.csv     — membership inference AUC across epsilon levels
#   results/feature_reconstruction.csv — feature reconstruction MSE / R2 (Site A)

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

python attacks/label_inference.py \
    --splits_dir data/vertical_splits \
    --ckpt_dir checkpoints \
    --output results/label_inference.csv \
    --seed 42

python attacks/embedding_mia.py \
    --splits_dir data/vertical_splits \
    --ckpt_dir checkpoints \
    --output results/embedding_mia.csv \
    --seed 42

python attacks/feature_reconstruction.py \
    --splits_dir data/vertical_splits \
    --ckpt_dir checkpoints \
    --output results/feature_reconstruction.csv \
    --seed 42

echo "[attacks] Done."
