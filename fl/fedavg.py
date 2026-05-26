"""
fl/fedavg.py — FedAvg aggregation for VFL-MTL encoder parameters.

fedavg_aggregate() computes a weighted average of encoder state dicts
(weighted by local dataset size) and returns a new global state dict.

Usage in train.py:
    from fl.fedavg import fedavg_aggregate

    global_params = fedavg_aggregate(
        [c.get_encoder_params() for c in clients],
        weights=[len(train_split_A), len(train_split_B), len(train_split_C)],
    )
    for client in clients:
        client.set_encoder_params(global_params)
"""

from __future__ import annotations
import torch


def fedavg_aggregate(
    params_list: list[dict],
    weights: list[int],
) -> dict:
    """
    Compute weighted average of encoder state dicts.

    Parameters
    ----------
    params_list : list of state dicts from VFLClient.get_encoder_params()
    weights     : non-negative integer weights (e.g. local dataset sizes)

    Returns
    -------
    Averaged state dict (same keys and dtypes as inputs)
    """
    assert len(params_list) == len(weights), "params_list and weights must have same length"
    total = sum(weights)
    assert total > 0, "Total weight must be positive"

    avg = {}
    for key in params_list[0]:
        weighted_sum = sum(
            params[key].float() * w
            for params, w in zip(params_list, weights)
        )
        avg[key] = (weighted_sum / total).to(params_list[0][key].dtype)

    return avg
