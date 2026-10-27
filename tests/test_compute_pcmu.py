import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.compute_pcmu import (
    PCMUConfig, best_round, bytes_per_round, evaluate_sweep, factorial_components, rounds_used,
)

INF = float("inf")
COLS = ["val_ihm_auroc", "val_decomp_auroc", "val_pheno_macro_auroc"]


def curve(best: int, patience: int = 15, peak: float = 0.8) -> pd.DataFrame:
    """Per-round curve rising to `peak` at round `best`, then flat-lower for `patience` rounds."""
    rounds = range(1, best + patience + 1)
    auc = [peak - 0.01 * abs(r - best) if r <= best else peak - 0.05 for r in rounds]
    return pd.DataFrame({"round": list(rounds), **{c: auc for c in COLS}})


def sweep(eps_best: dict[float, int], seeds=(0, 1)) -> pd.DataFrame:
    return pd.concat(
        [curve(b).assign(epsilon_level=e, seed=s, mode="uniform") for e, b in eps_best.items() for s in seeds],
        ignore_index=True,
    )


def factorial(eps_best: dict[float, int], seeds=(0, 1), embed_dim=64) -> pd.DataFrame:
    return pd.concat(
        [curve(b).assign(epsilon_level=e, seed=s, embed_dim=embed_dim, task_config="ihm_only")
         for e, b in eps_best.items() for s in seeds],
        ignore_index=True,
    )


def st_exp1() -> pd.DataFrame:
    return pd.concat(
        [curve(10, peak=0.75).assign(model=m, seed=s)
         for m in ("ST-IHM", "ST-Decomp", "ST-Pheno") for s in (0, 1)],
        ignore_index=True,
    )


def test_total_is_best_plus_patience():
    g = curve(best=12)
    assert best_round(g, ["ihm", "decomp", "pheno"]) == 12
    assert rounds_used(g, ["ihm"], "best") == 12
    assert rounds_used(g, ["ihm"], "total") == 12 + 15


def test_best_round_is_first_of_ties():
    g = pd.DataFrame({"round": [1, 2, 3], **{c: [0.7, 0.8, 0.8] for c in COLS}})
    assert best_round(g.iloc[::-1], ["ihm"]) == 2


def test_round_def_validated():
    with pytest.raises(ValueError):
        PCMUConfig(round_def="r90")


def test_reference_is_same_epsilon():
    out = evaluate_sweep(sweep({INF: 30, 1.0: 10}), st_exp1(), factorial({INF: 40, 1.0: 20}))
    ref = out.groupby("epsilon_level")["r_ref"].first()
    assert ref[INF] == 40 + 15 and ref[1.0] == 20 + 15
    row = out[out.epsilon_level == 1.0].iloc[0]
    assert math.isclose(row["eta_comm"], math.log(1 + 35 / 25))
    assert row["comm_bytes"] == 25 * bytes_per_round(64, 14_679)


def test_best_round_sensitivity():
    out = evaluate_sweep(sweep({INF: 30, 1.0: 10}), st_exp1(), factorial({INF: 40, 1.0: 20}),
                         config=PCMUConfig(round_def="best"))
    row = out[out.epsilon_level == 1.0].iloc[0]
    assert row["r"] == 10 and row["r_ref"] == 20


def test_missing_epsilon_raises():
    with pytest.raises(ValueError, match=r"ε = \[2\.0\]"):
        evaluate_sweep(sweep({INF: 30, 2.0: 10}), st_exp1(), factorial({INF: 40}))


def test_missing_seed_raises():
    with pytest.raises(ValueError, match=r"ε = \[1\.0\]"):
        evaluate_sweep(sweep({INF: 30, 1.0: 10}), st_exp1(),
                       pd.concat([factorial({INF: 40}), factorial({1.0: 20}, seeds=(0,))]))


def test_other_embed_dim_not_used():
    fact = pd.concat([factorial({INF: 40}), factorial({1.0: 20}, embed_dim=128)])
    with pytest.raises(ValueError, match=r"ε = \[1\.0\]"):
        evaluate_sweep(sweep({INF: 30, 1.0: 10}), st_exp1(), fact)


def test_factorial_uses_same_embed_reference():
    fact = pd.concat([
        factorial({INF: 40, 1.0: 20}, embed_dim=d).assign(task_config=c)
        for d in (32, 128) for c in ("ihm_only", "all_tasks")
    ])
    fact.loc[(fact.embed_dim == 128) & (fact.task_config == "ihm_only") & (fact["round"] > 45), "round"] = None
    fact = fact.dropna(subset=["round"])   # embed 128 ST-IHM stops at round 45 at eps=inf
    st = {"ihm": 0.75, "decomp": 0.75, "pheno": 0.75}
    out = factorial_components(fact, st).set_index(["embed_dim", "epsilon_level", "seed"])
    assert out.loc[(32, INF, 0), "r_ref"] == 55 and out.loc[(128, INF, 0), "r_ref"] == 45
    assert out.loc[(128, 1.0, 0), "comm_bytes"] == 35 * bytes_per_round(128, 14_679)


def test_payload_per_embed_dim():
    # 11.3 / 22.5 / 45.1 MB per round (audit_revision/pcmu_definition_comparison/summary.md)
    assert [round(bytes_per_round(d, 14_679) / 1e6, 1) for d in (32, 64, 128)] == [11.3, 22.5, 45.1]
