#!/bin/bash
#SBATCH --job-name=prism_eps_sweep
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=01:00:00
#SBATCH --output=logs/eps_sweep_%A_%a.out
#SBATCH --error=logs/eps_sweep_%A_%a.err

# Epsilon sweep: uniform noise, 3 seeds (42, 123, 7), 100 rounds per level
# ε levels: 0.5, 1, 2, 5, 10  (ε=∞ / no-DP is provided by exp1.csv)
# Each array task runs one ε level.
# Outputs merged into results/privacy_utility_combined.csv by the script itself.
#
# After all 5 array tasks complete, merge with no-DP baseline:
#   python -c "
#   import pandas as pd
#   eps = pd.read_csv('results/privacy_utility_combined.csv')
#   nodp = pd.read_csv('results/exp1.csv')
#   nodp = nodp[nodp.model=='VFL-MTL'][['seed','val_ihm_auroc','val_decomp_auroc','val_pheno_macro_auroc']]
#   nodp['epsilon'] = float('inf'); nodp['mode'] = 'no_dp'
#   # take last round per seed
#   nodp = nodp.groupby('seed').last().reset_index()
#   combined = pd.concat([eps, nodp], ignore_index=True)
#   combined.to_csv('results/privacy_utility_combined.csv', index=False)
#   "

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

EPS_LEVELS=(0.5 1.0 2.0 5.0 10.0)
EPS=${EPS_LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "[eps_sweep] Running epsilon=$EPS"

python experiments/privacy_utility_curves.py \
    --splits_dir data/vertical_splits \
    --epsilon $EPS \
    --output results/privacy_utility_combined.csv \
    --device cuda \
    --save_checkpoint

echo "[eps_sweep] Done for epsilon=$EPS"
