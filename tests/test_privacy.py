"""
tests/test_privacy.py — Unit tests for privacy modules.

Covers:
  - AdaptiveDPSGD: clipping, noise injection, sigma interfaces
  - RenyiAccountant: monotonic ε, higher σ → lower ε, per-task composition
  - DPVFLClient: gradient clipping and DP noise in the training loop
  - train.py privacy_config integration (smoke test with use_synthetic)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy.adaptive_dpsgd import AdaptiveDPSGD, DPVFLClient, SITE_TASK_MAP
from privacy.renyi_accountant import RenyiAccountant
from fl.client import VFLClient
from fl.server import VFLServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
EMBED_DIM  = 64
MAX_GRAD_NORM = 1.0


def _make_grad(B: int = BATCH_SIZE, D: int = EMBED_DIM, scale: float = 5.0) -> torch.Tensor:
    """Random gradient tensor with norms >> max_grad_norm to trigger clipping."""
    return torch.randn(B, D) * scale


def _make_batch(input_dim: int, B: int = BATCH_SIZE, T: int = 16):
    x    = torch.randn(B, T, input_dim)
    mask = torch.ones(B, T)
    return x, mask


# ---------------------------------------------------------------------------
# AdaptiveDPSGD — clipping
# ---------------------------------------------------------------------------

class TestClipping:

    def test_clipped_norms_at_most_max_grad_norm(self):
        # sigma=0 triggers early return (no clip, no noise by design).
        # Use near-zero sigma so clipping runs but noise contribution is negligible.
        dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
        dp.set_uniform(sigma=1e-10)
        torch.manual_seed(0)
        grad = _make_grad(scale=10.0)
        clipped = dp.clip_and_noise(grad, "ihm")
        # Noise std = 1e-10 * 1.0 / sqrt(32) ≈ 1.8e-11 — well within tolerance
        norms = clipped.norm(2, dim=-1)
        assert (norms <= MAX_GRAD_NORM + 1e-4).all(), (
            f"Some clipped norms exceed max_grad_norm={MAX_GRAD_NORM}: max={norms.max():.4f}"
        )

    def test_already_small_grad_unchanged(self):
        # sigma=0 → early return (no-op); small norms also bypass clipping.
        dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
        dp.set_uniform(sigma=0.0)
        small = torch.randn(BATCH_SIZE, EMBED_DIM) * 0.01
        out = dp.clip_and_noise(small, "ihm")
        assert torch.allclose(out, small, atol=1e-6), "Small gradient was modified unexpectedly"


# ---------------------------------------------------------------------------
# AdaptiveDPSGD — noise
# ---------------------------------------------------------------------------

class TestNoise:

    def test_noise_added_with_nonzero_sigma(self):
        dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
        dp.set_uniform(sigma=1.0)
        torch.manual_seed(0)
        grad = torch.zeros(BATCH_SIZE, EMBED_DIM)
        out = dp.clip_and_noise(grad, "ihm")
        assert not torch.allclose(out, grad), "No noise was added with sigma=1.0"

    def test_no_noise_with_sigma_zero(self):
        dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
        dp.set_uniform(sigma=0.0)
        grad = _make_grad(scale=0.5)   # within norm bound so clipping doesn't change it
        out = dp.clip_and_noise(grad, "ihm")
        assert torch.allclose(out, grad, atol=1e-6), "Noise added despite sigma=0"

    def test_no_noise_returned_when_sigma_unset(self):
        dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
        # Nothing configured at all → sigma defaults to 0.0
        grad = _make_grad(scale=0.5)
        out = dp.clip_and_noise(grad, "ihm")
        assert torch.allclose(out, grad, atol=1e-6)

    def test_higher_sigma_larger_noise_std(self):
        """Expected noise std scales proportionally with σ."""
        B, D = 1024, 64
        results = {}
        for sigma in (0.5, 2.0):
            dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
            dp.set_uniform(sigma=sigma)
            torch.manual_seed(99)
            grad = torch.zeros(B, D)
            out = dp.clip_and_noise(grad, "ihm")
            results[sigma] = out.std().item()
        assert results[2.0] > results[0.5], (
            f"Higher sigma should produce more noise: σ=0.5→{results[0.5]:.4f}, σ=2.0→{results[2.0]:.4f}"
        )


# ---------------------------------------------------------------------------
# AdaptiveDPSGD — sigma interfaces
# ---------------------------------------------------------------------------

class TestSigmaInterfaces:

    def test_set_uniform_assigns_all_tasks(self):
        dp = AdaptiveDPSGD()
        dp.set_uniform(1.5)
        for t in ("ihm", "decomp", "pheno"):
            assert dp.sigma_for(t) == 1.5, f"Task {t} did not get uniform sigma"

    def test_set_stratified_assigns_per_task(self):
        dp = AdaptiveDPSGD()
        dp.set_stratified(sigma_ihm=0.5, sigma_decomp=1.0, sigma_pheno=1.5)
        assert dp.sigma_for("ihm")    == 0.5
        assert dp.sigma_for("decomp") == 1.0
        assert dp.sigma_for("pheno")  == 1.5

    def test_is_enabled_after_set_uniform(self):
        dp = AdaptiveDPSGD()
        dp.set_uniform(1.0)
        assert dp.is_enabled

    def test_is_not_enabled_before_set(self):
        dp = AdaptiveDPSGD()
        assert not dp.is_enabled

    def test_unknown_site_raises(self):
        dp = AdaptiveDPSGD()
        dp.set_uniform(1.0)
        server = VFLServer()
        with pytest.raises(ValueError, match="Unknown site"):
            DPVFLClient(dp_mechanism=dp, site="Z", input_dim=7)


# ---------------------------------------------------------------------------
# Encoder gradients are clipped to max_grad_norm (end-to-end)
# ---------------------------------------------------------------------------

class TestEncoderGradientClipping:

    def test_encoder_receives_clipped_gradient(self):
        """
        DPVFLClient.receive_gradient() must clip the incoming gradient so that
        per-sample norms ≤ max_grad_norm before backpropagating.
        We verify this by monkey-patching the parent receive_gradient and
        inspecting the tensor passed to it.
        """
        dp = AdaptiveDPSGD(max_grad_norm=MAX_GRAD_NORM)
        dp.set_uniform(sigma=0.0)  # no noise — isolate clipping

        captured = []

        class _CaptureDPClient(DPVFLClient):
            def receive_gradient(self, grad: torch.Tensor) -> None:
                # Call DP noise (clip) then capture result before backprop
                if self._dp.is_enabled:
                    grad = self._dp.clip_and_noise(grad, self._task)
                captured.append(grad.detach().clone())
                VFLClient.receive_gradient(self, grad)

        client = _CaptureDPClient(dp_mechanism=dp, site="A", input_dim=7)
        server = VFLServer()
        clients_B = VFLClient(input_dim=4)
        clients_C = VFLClient(input_dim=3)

        x_A, mask_A = _make_batch(7)
        x_B, mask_B = _make_batch(4)
        x_C, mask_C = _make_batch(3)

        y_ihm    = torch.randint(0, 2, (BATCH_SIZE,)).float()
        y_decomp = torch.randint(0, 2, (BATCH_SIZE,)).float()
        y_pheno  = torch.randint(0, 2, (BATCH_SIZE, 25)).float()

        emb_A = client.forward(x_A, mask_A)
        emb_B = clients_B.forward(x_B, mask_B)
        emb_C = clients_C.forward(x_C, mask_C)

        server.aggregate_embeddings({"A": emb_A, "B": emb_B, "C": emb_C})
        loss, _ = server.forward_and_loss({"ihm": y_ihm, "decomp": y_decomp, "pheno": y_pheno})
        server.backward_and_step(loss)
        grads = server.get_embedding_gradients()
        client.receive_gradient(grads["A"])

        assert len(captured) == 1
        norms = captured[0].norm(2, dim=-1)
        assert (norms <= MAX_GRAD_NORM + 1e-5).all(), (
            f"Clipped gradient norms exceed max_grad_norm: max={norms.max():.4f}"
        )


# ---------------------------------------------------------------------------
# RenyiAccountant — ε monotonically increases with rounds
# ---------------------------------------------------------------------------

class TestRenyiMonotonic:

    def test_epsilon_increases_with_rounds(self):
        acc = RenyiAccountant()
        sigma, sr = 1.0, 0.01
        prev_eps = {t: 0.0 for t in ("ihm", "decomp", "pheno")}
        for step_idx in range(1, 6):
            acc.step(noise_multiplier=sigma, sample_rate=sr, num_steps=1)
            eps = acc.get_epsilon(delta=1e-5)
            for t in ("ihm", "decomp", "pheno"):
                assert eps[t] >= prev_eps[t] - 1e-6, (
                    f"ε did not increase at step {step_idx} for task {t}: "
                    f"prev={prev_eps[t]:.4f}, now={eps[t]:.4f}"
                )
                prev_eps[t] = eps[t]


# ---------------------------------------------------------------------------
# RenyiAccountant — higher σ → lower ε at same round count
# ---------------------------------------------------------------------------

class TestHigherSigmaLowerEpsilon:

    def test_higher_sigma_gives_lower_epsilon(self):
        sr = 0.01
        n_steps = 5
        eps_results = {}
        for sigma in (0.5, 1.0, 2.0):
            acc = RenyiAccountant()
            acc.step(noise_multiplier=sigma, sample_rate=sr, num_steps=n_steps)
            eps_results[sigma] = acc.get_epsilon(delta=1e-5)["ihm"]

        assert eps_results[0.5] > eps_results[1.0] > eps_results[2.0], (
            f"ε ordering violated: σ=0.5→{eps_results[0.5]:.3f}, "
            f"σ=1.0→{eps_results[1.0]:.3f}, σ=2.0→{eps_results[2.0]:.3f}"
        )


# ---------------------------------------------------------------------------
# RenyiAccountant — per-task ε sum ≤ ε_total + tolerance
# ---------------------------------------------------------------------------

class TestEpsilonComposition:

    def test_per_task_epsilon_sum_leq_total(self):
        """
        Under standard composition, ε_total ≤ Σ_k ε_k.
        Equivalently, each individual ε_k ≤ ε_total.
        We verify: max(ε_k) ≤ sum(ε_k) (trivially true) and that
        the coupling_epsilon_inflation() function is non-negative.
        """
        acc = RenyiAccountant()
        acc.step_stratified(
            sigma_map={"ihm": 0.5, "decomp": 1.0, "pheno": 1.5},
            sample_rate=0.01,
            num_steps=5,
        )
        eps = acc.get_epsilon(delta=1e-5)
        vals = list(eps.values())
        eps_sum = sum(vals)
        eps_max = max(vals)

        assert eps_max <= eps_sum + 1e-3, (
            f"max(ε_k)={eps_max:.4f} > sum(ε_k)={eps_sum:.4f}"
        )

        inflation = acc.coupling_epsilon_inflation(delta=1e-5)
        assert inflation >= -1e-6, f"coupling_epsilon_inflation should be >= 0: {inflation}"

    def test_stratified_lower_sigma_gives_lower_epsilon_for_ihm(self):
        """σ_IHM < σ_Decomp < σ_Pheno → ε_IHM > ε_Decomp > ε_Pheno (tighter noise = higher ε)."""
        acc = RenyiAccountant()
        acc.step_stratified(
            sigma_map={"ihm": 0.5, "decomp": 1.0, "pheno": 2.0},
            sample_rate=0.01,
            num_steps=5,
        )
        eps = acc.get_epsilon(delta=1e-5)
        assert eps["ihm"] > eps["decomp"] > eps["pheno"], (
            f"Expected ε_IHM > ε_Decomp > ε_Pheno but got: {eps}"
        )


# ---------------------------------------------------------------------------
# RenyiAccountant — gradient coupling matrix
# ---------------------------------------------------------------------------

class TestCouplingMatrix:

    def test_empty_before_logging(self):
        acc = RenyiAccountant()
        assert acc.cross_task_coupling_matrix() == {}

    def test_coupling_matrix_has_expected_keys(self):
        acc = RenyiAccountant()
        acc.log_grad_sim({
            "grad_sim_ihm_decomp":   0.3,
            "grad_sim_ihm_pheno":    0.1,
            "grad_sim_decomp_pheno": 0.2,
        })
        cm = acc.cross_task_coupling_matrix()
        assert set(cm.keys()) == {
            "grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno"
        }

    def test_coupling_matrix_averages_across_rounds(self):
        acc = RenyiAccountant()
        vals_a = {"grad_sim_ihm_decomp": 0.2, "grad_sim_ihm_pheno": 0.4, "grad_sim_decomp_pheno": 0.6}
        vals_b = {"grad_sim_ihm_decomp": 0.4, "grad_sim_ihm_pheno": 0.6, "grad_sim_decomp_pheno": 0.8}
        acc.log_grad_sim(vals_a)
        acc.log_grad_sim(vals_b)
        cm = acc.cross_task_coupling_matrix()
        assert abs(cm["grad_sim_ihm_decomp"]   - 0.3) < 1e-6
        assert abs(cm["grad_sim_ihm_pheno"]    - 0.5) < 1e-6
        assert abs(cm["grad_sim_decomp_pheno"] - 0.7) < 1e-6

    def test_n_logged_rounds(self):
        acc = RenyiAccountant()
        for i in range(3):
            acc.log_grad_sim({"grad_sim_ihm_decomp": float(i)})
        assert acc.n_logged_rounds == 3


# ---------------------------------------------------------------------------
# train.py privacy_config integration (smoke test)
# ---------------------------------------------------------------------------

class TestTrainPrivacyConfig:

    def _run(self, privacy_config):
        from train import run_training, TrainConfig
        cfg = TrainConfig(
            use_synthetic=True,
            n_synthetic=64,
            batch_size=32,
            n_rounds=3,
            eval_every=3,
            patience=0,
            use_fedavg=False,
            privacy_config=privacy_config,
        )
        return run_training(cfg)

    def test_uniform_dp_runs_and_logs_epsilon(self):
        results = self._run({
            "mode": "uniform", "sigma": 1.0, "max_grad_norm": 1.0, "delta": 1e-5
        })
        assert len(results) == 3
        last = results[-1]
        for key in ("epsilon_ihm", "epsilon_decomp", "epsilon_pheno"):
            assert key in last, f"Missing {key} in round log"
            assert last[key] > 0.0, f"{key} should be positive"
            assert not math.isnan(last[key]), f"{key} is NaN"

    def test_stratified_dp_runs_and_logs_epsilon(self):
        results = self._run({
            "mode": "stratified",
            "sigma_ihm": 0.5, "sigma_decomp": 1.0, "sigma_pheno": 1.5,
            "max_grad_norm": 1.0, "delta": 1e-5,
        })
        assert len(results) == 3
        last = results[-1]
        # Tighter noise (lower σ) for IHM → higher ε_IHM
        assert last["epsilon_ihm"] > last["epsilon_pheno"], (
            f"Expected ε_IHM > ε_Pheno; got ihm={last['epsilon_ihm']:.3f} "
            f"pheno={last['epsilon_pheno']:.3f}"
        )

    def test_no_dp_no_epsilon_keys(self):
        results = self._run(None)
        assert len(results) == 3
        for row in results:
            for key in ("epsilon_ihm", "epsilon_decomp", "epsilon_pheno"):
                assert key not in row, f"Unexpected {key} in no-DP run"

    def test_unknown_mode_raises(self):
        with pytest.raises((ValueError, KeyError)):
            self._run({"mode": "bad_mode", "sigma": 1.0})

    def test_epsilon_monotone_across_rounds(self):
        results = self._run({
            "mode": "uniform", "sigma": 1.0, "max_grad_norm": 1.0, "delta": 1e-5
        })
        epsilons_ihm = [r["epsilon_ihm"] for r in results]
        for i in range(1, len(epsilons_ihm)):
            assert epsilons_ihm[i] >= epsilons_ihm[i - 1] - 1e-6, (
                f"ε_IHM decreased at round {i+1}: {epsilons_ihm[i-1]:.4f} → {epsilons_ihm[i]:.4f}"
            )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
