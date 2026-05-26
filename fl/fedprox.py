"""
fl/fedprox.py — FedProx proximal penalty for VFL-MTL client training.

The proximal term (mu/2) * ||w_local - w_global||^2 is added to the
client's loss before backpropagation, penalising deviation from the
global model. This improves convergence in heterogeneous federated settings.

Reference: Li et al. (2020), "Federated Optimization in Heterogeneous Networks"

Usage in train.py:
    from fl.fedprox import fedprox_penalty

    penalty = fedprox_penalty(client.encoder, global_params, mu=0.01)
    total_loss = total_loss + penalty
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


def fedprox_penalty(
    local_model: nn.Module,
    global_params: dict,
    mu: float = 0.01,
) -> Tensor:
    """
    Compute the FedProx proximal penalty.

    penalty = (mu / 2) * sum_over_layers( ||w_local - w_global||^2 )

    Parameters
    ----------
    local_model   : the client's local nn.Module (SiteEncoder)
    global_params : global state dict from fedavg_aggregate()
    mu            : proximal penalty coefficient (default 0.01)

    Returns
    -------
    Scalar tensor (differentiable w.r.t. local_model.parameters())
    """
    device = next(local_model.parameters()).device
    penalty = torch.zeros(1, device=device)
    for name, param in local_model.named_parameters():
        if name in global_params:
            global_val = global_params[name].to(device)
            penalty = penalty + ((param - global_val) ** 2).sum()
    return (mu / 2) * penalty
