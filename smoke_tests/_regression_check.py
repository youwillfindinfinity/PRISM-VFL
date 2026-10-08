"""Throwaway regression check for the train.py/dataset.py P0 edits. Not part of
the plan's deliverables — imports the real CONFIGS/SEEDS from run_exp1.py and
privacy_utility_curves.py, redirects ckpt_dir to smoke_tests/revision1/ only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig
from experiments.run_exp1 import CONFIGS as EXP1_CONFIGS
from experiments.privacy_utility_curves import (
    _build_uniform_privacy_config, EPSILON_LEVELS,
)

CKPT = "smoke_tests/revision1/checkpoints"

print("== run_exp1.py CONFIGS ==")
for model_name, model_cfg in EXP1_CONFIGS.items():
    cfg = TrainConfig(n_rounds=3, seed=42, use_synthetic=True, model_name=model_name,
                       ckpt_dir=CKPT, **model_cfg)
    rows = run_training(cfg)
    assert rows and "best_round" in rows[-1] and "stopping_round" in rows[-1]
    print(f"  {model_name}: {len(rows)} rows, best_round={rows[-1]['best_round']}")

print("== privacy_utility_curves.py DP config ==")
cfg = TrainConfig(n_rounds=3, seed=42, use_synthetic=True, model_name="DP-smoke",
                   ckpt_dir=CKPT, task_weights={"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
                   uncertainty_weighting=True,
                   privacy_config=_build_uniform_privacy_config(1.0))
rows = run_training(cfg)
assert rows and "epsilon_ihm" in rows[-1] and "grad_norm_ihm" in rows[-1]
print(f"  DP-smoke: {len(rows)} rows, epsilon_ihm={rows[-1]['epsilon_ihm']:.3f}")

print("REGRESSION CHECK PASSED")
