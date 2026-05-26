import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.centralized import train_centralized


def test_returns_one_row_per_epoch():
    rows = train_centralized(root=".", n_epochs=2, lr=1e-3,
                             batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2


def test_row_fields():
    rows = train_centralized(root=".", n_epochs=1, lr=1e-3,
                             batch_size=32, seed=42, use_synthetic=True)
    r = rows[0]
    assert r["model"] == "centralized_oracle"
    assert r["epoch"] == 1
    assert r["train_loss"] > 0


def test_has_all_task_metrics():
    rows = train_centralized(root=".", n_epochs=1, lr=1e-3,
                             batch_size=32, seed=42, use_synthetic=True)
    r = rows[0]
    assert any(k.startswith("val_ihm_") for k in r)
    assert any(k.startswith("val_decomp_") for k in r)
    assert any(k.startswith("val_pheno_") for k in r)
