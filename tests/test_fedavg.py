# tests/test_fedavg.py
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fl.client import VFLClient
from fl.fedavg import fedavg_aggregate
from fl.fedprox import fedprox_penalty


def make_clients(n: int = 3, input_dim: int = 7):
    return [VFLClient(input_dim=input_dim) for _ in range(n)]


def test_fedavg_output_is_weighted_average():
    """fedavg_aggregate must return a state dict that is the weighted average."""
    torch.manual_seed(0)
    clients = make_clients()

    key = list(clients[0].encoder.state_dict().keys())[0]
    for i, c in enumerate(clients):
        state = c.encoder.state_dict()
        state[key] = torch.zeros_like(state[key]) + float(i)
        c.set_encoder_params(state)

    weights = [1, 2, 3]
    params_list = [c.get_encoder_params() for c in clients]
    avg = fedavg_aggregate(params_list, weights)

    expected_val = (0*1 + 1*2 + 2*3) / (1+2+3)  # = 8/6 ≈ 1.333
    actual_val = avg[key].mean().item()
    assert abs(actual_val - expected_val) < 1e-5, f"Expected {expected_val}, got {actual_val}"


def test_fedavg_sets_params_on_clients():
    """After aggregation, setting params on clients should update their weights."""
    torch.manual_seed(1)
    clients = make_clients()
    weights = [10, 10, 10]
    params_list = [c.get_encoder_params() for c in clients]
    avg = fedavg_aggregate(params_list, weights)
    for c in clients:
        c.set_encoder_params(avg)


def test_fedprox_penalty_zero_when_same():
    """Penalty must be 0 when local params equal global params."""
    torch.manual_seed(2)
    client = VFLClient(input_dim=7)
    global_params = client.get_encoder_params()
    penalty = fedprox_penalty(client.encoder, global_params, mu=1.0)
    assert penalty.item() < 1e-8, f"Expected ~0, got {penalty.item()}"


def test_fedprox_penalty_positive_when_different():
    """Penalty must be positive when local params differ from global."""
    torch.manual_seed(3)
    client = VFLClient(input_dim=7)
    global_params = client.get_encoder_params()
    for param in client.encoder.parameters():
        param.data += 1.0
    penalty = fedprox_penalty(client.encoder, global_params, mu=1.0)
    assert penalty.item() > 0.0


def test_fedprox_penalty_scales_with_mu():
    """Doubling mu must double the penalty."""
    torch.manual_seed(4)
    client = VFLClient(input_dim=7)
    global_params = client.get_encoder_params()
    for param in client.encoder.parameters():
        param.data += 0.5

    p1 = fedprox_penalty(client.encoder, global_params, mu=1.0).item()
    p2 = fedprox_penalty(client.encoder, global_params, mu=2.0).item()
    assert abs(p2 - 2 * p1) < 1e-5, f"Expected p2=2*p1, got p1={p1}, p2={p2}"
