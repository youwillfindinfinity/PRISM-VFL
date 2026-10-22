#!/bin/bash
#SBATCH --job-name=prism_baselines
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=04:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# Centralized oracle and local-only baselines (all seeds in experiment_config.SEEDS)
# Outputs: results_revision/local_only_{A,B,C}.csv, results_revision/centralized.csv (per-epoch logs),
#          results_revision/baselines_comparison.csv (final epoch per seed),
#          checkpoints_revision/best_local_{A,B,C}_seed*.pt, checkpoints_revision/best_centralized_seed*.pt

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

# Strict mode only after module load, whose scripts reference unset variables.
set -euo pipefail

SEEDS=$(python -c "from experiment_config import SEEDS; print(*SEEDS)")
MAX_ROUNDS=$(python -c "from experiment_config import MAX_ROUNDS; print(MAX_ROUNDS)")

for SITE in A B C; do
    python -u baselines/local_only.py --site ${SITE} --seeds ${SEEDS} --n_epochs ${MAX_ROUNDS} \
        --ckpt_dir checkpoints_revision --output results_revision/local_only_${SITE}.csv
done

python -u baselines/centralized.py --seeds ${SEEDS} --n_epochs ${MAX_ROUNDS} \
    --ckpt_dir checkpoints_revision --output results_revision/centralized.csv

python -u experiments/run_baselines.py --skip_rerun \
    --local_a results_revision/local_only_A.csv --local_b results_revision/local_only_B.csv \
    --local_c results_revision/local_only_C.csv --central results_revision/centralized.csv \
    --output results_revision/baselines_comparison.csv

echo "[baselines] Done."
