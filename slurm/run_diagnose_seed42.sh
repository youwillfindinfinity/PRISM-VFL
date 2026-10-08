#!/bin/bash
#SBATCH --job-name=prism_seed42_probes
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=03:00:00
#SBATCH --output=logs/seed42_probes_%j.out
#SBATCH --error=logs/seed42_probes_%j.err

# Seed-42 instability probes: 1-round init donor + 5 PRISM runs at seed 42.
# Output: results_revision/seed42_probes.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

pip install --quiet --user "opacus>=1.4.0" "scikit-multilearn>=0.2.0" 2>/dev/null

OUT=results_revision/seed42_probes.csv
CKPT=checkpoints_revision
EXTRA=""
# Real-data pre-flight: sbatch --time=02:00:00 slurm/run_diagnose_seed42.sh 2roundtest
if [ "$1" = "2roundtest" ]; then
    OUT=smoke_tests/revision1/2roundtest_probes.csv
    CKPT=smoke_tests/revision1/2roundtest_checkpoints
    EXTRA="--n_rounds 2"
fi

python revision_scripts/diagnose_seed42.py \
    --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
    --output $OUT \
    --ckpt_dir $CKPT \
    --device cuda $EXTRA

echo "[seed42_probes] Done. Results in $OUT"
