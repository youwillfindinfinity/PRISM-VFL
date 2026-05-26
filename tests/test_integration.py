"""
tests/test_integration.py — End-to-end 3-round VFL-MTL integration test.

Uses synthetic tensors matching real data shapes. No file I/O required.
Asserts:
  - loss decreases from round 1 to round 3
  - embedding shapes are correct at every stage
  - gradients flow back to each client encoder (weights change)
  - server prediction outputs have correct shapes
"""

import sys
from pathlib import Path

import torch
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.client import VFLClient
from fl.server import VFLServer

# ---------------------------------------------------------------------------
# Synthetic batch factory
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
MAX_SEQ_LEN = 48
EMBED_DIM = 64

SITE_CONFIGS = {
    "A": {"input_dim": 7,  "task": "binary"},
    "B": {"input_dim": 4,  "task": "binary"},
    "C": {"input_dim": 3,  "task": "multilabel"},
}

def make_batch(input_dim: int, task: str, B: int = BATCH_SIZE):
    """Return (x, mask, y) tensors for one site."""
    x    = torch.randn(B, MAX_SEQ_LEN, input_dim)
    # Variable-length sequences: lengths between 10 and 48
    lengths = torch.randint(10, MAX_SEQ_LEN + 1, (B,))
    mask = torch.zeros(B, MAX_SEQ_LEN)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    if task == "binary":
        y = torch.randint(0, 2, (B,)).float()
    else:  # multilabel
        y = torch.randint(0, 2, (B, 25)).float()

    return x, mask, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_3_round_loss_decreases():
    """Loss must strictly decrease over 3 rounds on the same batch."""
    torch.manual_seed(42)

    clients = {
        site: VFLClient(input_dim=cfg["input_dim"])
        for site, cfg in SITE_CONFIGS.items()
    }
    server = VFLServer()

    # Fix a single batch (same data every round to isolate optimisation signal)
    batches = {
        site: make_batch(cfg["input_dim"], cfg["task"])
        for site, cfg in SITE_CONFIGS.items()
    }

    round_losses = []

    for round_idx in range(3):
        # Step 1 — each client encodes its local features
        cut_embeddings = {}
        for site, client in clients.items():
            x, mask, _ = batches[site]
            cut_embeddings[site] = client.forward(x, mask)

        # Step 2 — server concatenates embeddings and computes loss
        server.aggregate_embeddings(cut_embeddings)
        labels = {
            "ihm":    batches["A"][2],
            "decomp": batches["B"][2],
            "pheno":  batches["C"][2],
        }
        total_loss, task_losses = server.forward_and_loss(labels)
        round_losses.append(total_loss.item())

        # Step 3 — server backprops and returns gradients to clients
        server.backward_and_step(total_loss)
        grads = server.get_embedding_gradients()

        # Step 4 — each client applies its gradient
        for site, client in clients.items():
            client.receive_gradient(grads[site])

        print(f"  Round {round_idx + 1}: total_loss={total_loss.item():.4f} "
              f"| ihm={task_losses['ihm'].item():.4f} "
              f"| decomp={task_losses['decomp'].item():.4f} "
              f"| pheno={task_losses['pheno'].item():.4f}")

    assert round_losses[2] < round_losses[0], (
        f"Loss did not decrease: round 1={round_losses[0]:.4f}, "
        f"round 3={round_losses[2]:.4f}"
    )


def test_embedding_shapes():
    """Cut-layer embeddings and gradients must have correct shapes."""
    torch.manual_seed(0)

    clients = {
        site: VFLClient(input_dim=cfg["input_dim"])
        for site, cfg in SITE_CONFIGS.items()
    }
    server = VFLServer()

    cut_embeddings = {}
    for site, cfg in SITE_CONFIGS.items():
        x, mask, _ = make_batch(cfg["input_dim"], cfg["task"])
        emb = clients[site].forward(x, mask)
        assert emb.shape == (BATCH_SIZE, EMBED_DIM), \
            f"Site {site}: expected ({BATCH_SIZE}, {EMBED_DIM}), got {emb.shape}"
        assert emb.grad_fn is None,       "cut embedding must be detached"
        assert emb.requires_grad is True, "cut embedding must have requires_grad=True"
        cut_embeddings[site] = emb

    concat = server.aggregate_embeddings(cut_embeddings)
    assert concat.shape == (BATCH_SIZE, 3 * EMBED_DIM), \
        f"concat embedding shape wrong: {concat.shape}"

    labels = {
        "ihm":    make_batch(7, "binary")[2],
        "decomp": make_batch(4, "binary")[2],
        "pheno":  make_batch(3, "multilabel")[2],
    }
    total_loss, _ = server.forward_and_loss(labels)
    server.backward_and_step(total_loss)
    grads = server.get_embedding_gradients()

    for site in ("A", "B", "C"):
        assert grads[site].shape == (BATCH_SIZE, EMBED_DIM), \
            f"Gradient shape wrong for site {site}: {grads[site].shape}"


def test_prediction_output_shapes():
    """Server predict() must return tensors with correct task output shapes."""
    torch.manual_seed(1)
    server = VFLServer()
    clients = {
        site: VFLClient(input_dim=cfg["input_dim"])
        for site, cfg in SITE_CONFIGS.items()
    }

    embeddings = {}
    for site, cfg in SITE_CONFIGS.items():
        x, mask, _ = make_batch(cfg["input_dim"], cfg["task"])
        embeddings[site] = clients[site].eval_forward(x, mask)

    preds = server.predict(embeddings)

    assert preds["ihm"].shape    == (BATCH_SIZE, 1),  f"IHM shape: {preds['ihm'].shape}"
    assert preds["decomp"].shape == (BATCH_SIZE, 1),  f"Decomp shape: {preds['decomp'].shape}"
    assert preds["pheno"].shape  == (BATCH_SIZE, 25), f"Pheno shape: {preds['pheno'].shape}"


def test_encoder_weights_update_after_gradient():
    """Client encoder weights must change after receive_gradient()."""
    torch.manual_seed(2)
    client = VFLClient(input_dim=7)
    server = VFLServer()
    clients_B = VFLClient(input_dim=4)
    clients_C = VFLClient(input_dim=3)

    # Capture initial weights for Site A
    before = {k: v.clone() for k, v in client.encoder.state_dict().items()}

    x_A, mask_A, y_A = make_batch(7, "binary")
    x_B, mask_B, y_B = make_batch(4, "binary")
    x_C, mask_C, y_C = make_batch(3, "multilabel")

    emb_A = client.forward(x_A, mask_A)
    emb_B = clients_B.forward(x_B, mask_B)
    emb_C = clients_C.forward(x_C, mask_C)

    server.aggregate_embeddings({"A": emb_A, "B": emb_B, "C": emb_C})
    loss, _ = server.forward_and_loss({"ihm": y_A, "decomp": y_B, "pheno": y_C})
    server.backward_and_step(loss)
    grads = server.get_embedding_gradients()
    client.receive_gradient(grads["A"])

    after = client.encoder.state_dict()
    changed = any(
        not torch.equal(before[k], after[k]) for k in before
    )
    assert changed, "Encoder weights did not change after receive_gradient()"


def test_grad_sim_values_in_range():
    """compute_task_gradient_similarity must return cosine values in [-1, 1] for active tasks."""
    torch.manual_seed(5)
    clients = {s: VFLClient(input_dim=cfg["input_dim"]) for s, cfg in SITE_CONFIGS.items()}
    server = VFLServer()

    embeddings = {}
    for site, cfg in SITE_CONFIGS.items():
        x, mask, _ = make_batch(cfg["input_dim"], cfg["task"])
        embeddings[site] = clients[site].forward(x, mask)

    labels = {
        "ihm":    make_batch(7, "binary")[2],
        "decomp": make_batch(4, "binary")[2],
        "pheno":  make_batch(3, "multilabel")[2],
    }
    server.aggregate_embeddings(embeddings)
    total_loss, _ = server.forward_and_loss(labels)
    server.backward_and_step(total_loss)

    sims = server.compute_task_gradient_similarity(labels)

    expected_keys = {"grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno"}
    assert set(sims.keys()) == expected_keys, f"Unexpected keys: {sims.keys()}"

    for key, val in sims.items():
        assert not (val != val), f"{key} is NaN (all tasks active — should not be NaN)"
        assert -1.0 <= val <= 1.0, f"{key}={val:.4f} outside [-1, 1]"


def test_grad_sim_inactive_task_gives_nan():
    """Gradient similarity involving a zero-weight task must be NaN."""
    torch.manual_seed(6)
    # pheno weight = 0 → ihm_vs_pheno and decomp_vs_pheno must be NaN
    clients = {s: VFLClient(input_dim=cfg["input_dim"]) for s, cfg in SITE_CONFIGS.items()}
    server = VFLServer(task_weights={"ihm": 1.0, "decomp": 1.0, "pheno": 0.0})

    embeddings = {}
    for site, cfg in SITE_CONFIGS.items():
        x, mask, _ = make_batch(cfg["input_dim"], cfg["task"])
        embeddings[site] = clients[site].forward(x, mask)

    labels = {
        "ihm":    make_batch(7, "binary")[2],
        "decomp": make_batch(4, "binary")[2],
        "pheno":  make_batch(3, "multilabel")[2],
    }
    server.aggregate_embeddings(embeddings)
    total_loss, _ = server.forward_and_loss(labels)
    server.backward_and_step(total_loss)

    sims = server.compute_task_gradient_similarity(labels)

    import math
    assert math.isnan(sims["grad_sim_ihm_pheno"]),    "ihm_vs_pheno should be NaN when pheno weight=0"
    assert math.isnan(sims["grad_sim_decomp_pheno"]), "decomp_vs_pheno should be NaN when pheno weight=0"

    # ihm_vs_decomp: both active, must be a real number in [-1, 1]
    assert not math.isnan(sims["grad_sim_ihm_decomp"]), "ihm_vs_decomp should not be NaN"
    assert -1.0 <= sims["grad_sim_ihm_decomp"] <= 1.0


def test_grad_sim_does_not_corrupt_embedding_gradients():
    """
    Calling compute_task_gradient_similarity between backward_and_step and
    get_embedding_gradients must not zero or overwrite the embedding gradients
    that get_embedding_gradients relies on.
    """
    torch.manual_seed(7)
    clients = {s: VFLClient(input_dim=cfg["input_dim"]) for s, cfg in SITE_CONFIGS.items()}
    server = VFLServer()

    embeddings = {}
    for site, cfg in SITE_CONFIGS.items():
        x, mask, _ = make_batch(cfg["input_dim"], cfg["task"])
        embeddings[site] = clients[site].forward(x, mask)

    labels = {
        "ihm":    make_batch(7, "binary")[2],
        "decomp": make_batch(4, "binary")[2],
        "pheno":  make_batch(3, "multilabel")[2],
    }
    server.aggregate_embeddings(embeddings)
    total_loss, _ = server.forward_and_loss(labels)
    server.backward_and_step(total_loss)

    # Capture gradients BEFORE calling compute_task_gradient_similarity
    grads_before = server.get_embedding_gradients()
    grad_norm_before = {s: grads_before[s].norm().item() for s in ("A", "B", "C")}

    # Call the method — must not alter _concat_embedding.grad
    server.compute_task_gradient_similarity(labels)

    grads_after = server.get_embedding_gradients()
    grad_norm_after = {s: grads_after[s].norm().item() for s in ("A", "B", "C")}

    for site in ("A", "B", "C"):
        assert abs(grad_norm_before[site] - grad_norm_after[site]) < 1e-6, (
            f"Site {site} embedding gradient norm changed after compute_task_gradient_similarity: "
            f"before={grad_norm_before[site]:.6f}, after={grad_norm_after[site]:.6f}"
        )


if __name__ == "__main__":
    print("Running integration tests...")
    print("\n[1/4] test_embedding_shapes")
    test_embedding_shapes()
    print("  PASSED")

    print("\n[2/4] test_prediction_output_shapes")
    test_prediction_output_shapes()
    print("  PASSED")

    print("\n[3/4] test_encoder_weights_update_after_gradient")
    test_encoder_weights_update_after_gradient()
    print("  PASSED")

    print("\n[4/4] test_3_round_loss_decreases")
    test_3_round_loss_decreases()
    print("  PASSED")

    print("\n[5/7] test_grad_sim_values_in_range")
    test_grad_sim_values_in_range()
    print("  PASSED")

    print("\n[6/7] test_grad_sim_inactive_task_gives_nan")
    test_grad_sim_inactive_task_gives_nan()
    print("  PASSED")

    print("\n[7/7] test_grad_sim_does_not_corrupt_embedding_gradients")
    test_grad_sim_does_not_corrupt_embedding_gradients()
    print("  PASSED")

    print("\nAll integration tests passed.")
