#!/bin/bash
#SBATCH --job-name=prism_privacy_sweep
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_a100
#SBATCH --account=ausei18360
#SBATCH --time=24:00:00
#SBATCH --output=logs/prism_privacy_sweep_%j.out
#SBATCH --error=logs/prism_privacy_sweep_%j.err

# PRISM equal/uncertainty weighting x epsilon grid + single-task baselines,
# 10 seeds, one process (loaders built once per seed, reused across arms).
# ~15-17 GPU-hours estimated -- longer than every other job in this repo
# (all <=90 min; see slurm/README.md). Safe to resubmit after a timeout:
# seeds already in the output CSV are skipped. --seeds 7 42 ... restricts
# a job to a subset (use a distinct --output per concurrent job).
# Output: results_revision/prism_privacy_sweep.csv

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$SLURM_SUBMIT_DIR"

# opacus >=1.5.4 needs torch >=2.4 (nn.RMSNorm); --no-deps stops pip pulling a newer torch into ~/.local
pip install --quiet --user --no-deps "opacus>=1.4.0,<1.5.4" "scikit-multilearn>=0.2.0" 2>/dev/null

set -e

OUT=results_revision/prism_privacy_sweep.csv
CKPT=checkpoints_revision
EXTRA=""
# Real-data pre-flight: sbatch --time=02:00:00 slurm/run_prism_privacy_sweep.sh 2roundtest
if [ "$1" = "2roundtest" ]; then
    OUT=smoke_tests/revision1/2roundtest_sweep.csv
    CKPT=smoke_tests/revision1/2roundtest_checkpoints
    EXTRA="--n_rounds 2 --seeds 7"
    rm -f $OUT   # the sweep resumes from an existing CSV; the test always starts clean
fi

python -u revision_scripts/run_prism_privacy_sweep.py \
    --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
    --output $OUT \
    --ckpt_dir $CKPT \
    --device cuda $EXTRA

python -u revision_scripts/evaluate_test_prism_privacy_sweep.py \
    --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
    --input $OUT \
    --ckpt_dir $CKPT \
    --output ${OUT%.csv}_test.csv \
    --device cuda

echo "[prism_privacy_sweep] Done. Results in $OUT and ${OUT%.csv}_test.csv"
