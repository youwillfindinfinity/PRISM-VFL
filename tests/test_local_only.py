import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.local_only import train_local


@pytest.mark.parametrize("site", ["A", "B", "C"])
def test_returns_one_row_per_epoch(site):
    rows = train_local(site=site, root=".", n_epochs=2,
                       lr=1e-3, batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2


@pytest.mark.parametrize("site", ["A", "B", "C"])
def test_row_fields(site):
    rows = train_local(site=site, root=".", n_epochs=1,
                       lr=1e-3, batch_size=32, seed=42, use_synthetic=True)
    r = rows[0]
    assert r["model"] == f"local_{site}"
    assert r["site"] == site
    assert r["train_loss"] > 0
    assert r["epoch"] == 1


def test_site_a_has_ihm_metric():
    rows = train_local(site="A", root=".", n_epochs=1,
                       lr=1e-3, batch_size=32, seed=42, use_synthetic=True)
    assert any(k.startswith("val_") for k in rows[0])


def test_site_b_has_decomp_metric():
    rows = train_local(site="B", root=".", n_epochs=1,
                       lr=1e-3, batch_size=32, seed=42, use_synthetic=True)
    assert any("auc_roc" in k or "auc_pr" in k for k in rows[0])


def test_site_c_has_pheno_metric():
    rows = train_local(site="C", root=".", n_epochs=1,
                       lr=1e-3, batch_size=32, seed=42, use_synthetic=True)
    assert any("auc" in k.lower() for k in rows[0])
