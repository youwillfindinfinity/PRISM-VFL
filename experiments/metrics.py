"""
experiments/metrics.py — Per-task metric computation for VFL-MTL.

All functions take numpy arrays. Call after collecting predictions over
the full val/test set.

Metrics per task:
  IHM (binary)        : AUC-ROC, AUC-PR
  Decomp (binary)     : AUC-ROC, AUC-PR
  Phenotyping (multi) : Macro-AUC, Micro-AUC
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


def ihm_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N,) int or float binary labels
    y_prob : (N,) predicted probabilities
    """
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr":  float(average_precision_score(y_true, y_prob)),
    }


def decomp_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N,) int or float binary decompensation labels
    y_prob : (N,) predicted probabilities
    """
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr":  float(average_precision_score(y_true, y_prob)),
    }


def pheno_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N, 25) binary multi-label targets
    y_prob : (N, 25) predicted probabilities

    Notes
    -----
    Columns with only one class present (e.g. rare phenotypes absent from val
    split) are excluded from macro-AUC to avoid undefined-metric warnings and
    NaN propagation.  micro-AUC is computed over the full array.
    """
    # Mask out constant columns for macro-AUC
    valid = (y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < y_true.shape[0])
    macro_auc = (
        float(roc_auc_score(y_true[:, valid], y_prob[:, valid], average="macro"))
        if valid.any() else float("nan")
    )
    micro_auc = float(roc_auc_score(y_true, y_prob, average="micro"))
    return {"macro_auc": macro_auc, "micro_auc": micro_auc}


def compute_all_metrics(
    ihm_true, ihm_prob,
    decomp_true, decomp_prob,
    pheno_true, pheno_prob,
) -> dict[str, float]:
    """Convenience: compute all task metrics and return flat dict."""
    out = {}
    out.update({f"ihm_{k}":    v for k, v in ihm_metrics(ihm_true, ihm_prob).items()})
    out.update({f"decomp_{k}": v for k, v in decomp_metrics(decomp_true, decomp_prob).items()})
    out.update({f"pheno_{k}":  v for k, v in pheno_metrics(pheno_true, pheno_prob).items()})
    return out
