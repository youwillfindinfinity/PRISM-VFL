"""
privacy/adaptive_dpsgd.py — DP mechanism for VFL cut-layer gradients.

In VFL the server returns per-sample embedding gradients to each client:
  grad[i] = ∂L/∂z_i  shape (B, embed_dim)

Privacy is protected by applying DP noise at the cut layer before the
client uses those gradients for local backpropagation. This reduces label
leakage via cut-layer embeddings by adding DP noise to the encoder's gradient
updates (McMahan et al. 2018), making embeddings less label-informative and
reducing linear-probe and MIA attack success (Weng et al. 2021, Luo et al. 2021).

DPVFLClient subclasses VFLClient and overrides receive_gradient() to
apply clip_and_noise() transparently — no changes needed to the training
loop or fl/server.py.

Stratified mode assigns tighter noise (lower σ) to higher-stakes tasks:
  σ_IHM < σ_Decomp < σ_Pheno  (clinical risk hierarchy)
"""

from __future__ import annotations

import torch
from torch import Tensor

from fl.client import VFLClient

# Site → primary task assignment (fixed by MIMIC-III vertical split protocol)
SITE_TASK_MAP: dict[str, str] = {"A": "ihm", "B": "decomp", "C": "pheno"}

_TASKS = ("ihm", "decomp", "pheno")


class AdaptiveDPSGD:
    """
    DP mechanism for embedding gradients at the VFL cut layer.

    Applies per-sample L2 norm clipping and Gaussian noise to the
    (B, embed_dim) gradient tensor received from the server.

    Noise scale follows Abadi et al. (2016):
      - Clip each sample's gradient: ||g_i|| ≤ C  (C = max_grad_norm)
      - Add noise: g_i += N(0, (σ·C/√B)²·I)
    When summed during backward(), the total additive noise is N(0, (σ·C)²·I),
    matching the DP-SGD guarantee with sensitivity C and noise multiplier σ.

    Parameters
    ----------
    max_grad_norm : float
        Per-sample gradient clipping bound (sensitivity C). Default 1.0.
    """

    def __init__(self, max_grad_norm: float = 1.0) -> None:
        self.max_grad_norm = max_grad_norm
        self._sigma: dict[str, float] = {}

    # ------------------------------------------------------------------

    def set_uniform(self, sigma: float) -> None:
        """All tasks use the same noise multiplier σ."""
        for task in _TASKS:
            self._sigma[task] = sigma

    def set_stratified(
        self,
        sigma_ihm: float,
        sigma_decomp: float,
        sigma_pheno: float,
    ) -> None:
        """
        Per-task σ allocation based on clinical risk hierarchy.
        Tighter noise (lower σ) → stronger privacy for high-stakes tasks.
        Expected: σ_IHM < σ_Decomp < σ_Pheno.
        """
        self._sigma = {
            "ihm":    sigma_ihm,
            "decomp": sigma_decomp,
            "pheno":  sigma_pheno,
        }

    # ------------------------------------------------------------------

    def clip_and_noise(self, grad: Tensor, task: str) -> Tensor:
        """
        Apply DP to embedding gradient tensor.

        Parameters
        ----------
        grad : (B, embed_dim) — per-sample embedding gradients from server
        task : 'ihm' | 'decomp' | 'pheno' — selects per-task σ

        Returns
        -------
        (B, embed_dim) — clipped and noised gradient (same device as input)

        Note: the server returns the MEAN gradient ∂L/∂z = (1/B)·Σ ∂L_i/∂z_i,
        so each row has sensitivity C/B, not C.  Both the clipping bound and the
        noise std are scaled by 1/B.  The noise-multiplier σ (= noise/sensitivity)
        is unchanged, so Rényi-DP accounting via RenyiAccountant is unaffected.
        """
        sigma = self._sigma.get(task, 0.0)
        if sigma == 0.0:
            return grad

        B = grad.shape[0]

        # Effective per-row sensitivity: C/B (rows are batch-averaged gradients)
        effective_C = self.max_grad_norm / B

        # Per-sample L2 norm clipping to effective_C
        norms = grad.norm(2, dim=-1, keepdim=True)                    # (B, 1)
        scale = (effective_C / (norms + 1e-8)).clamp(max=1.0)
        clipped = grad * scale                                         # (B, embed_dim)

        # Gaussian noise: std = σ·(C/B)/√B → summed-row noise std = σ·(C/B)
        noise_std = sigma * effective_C / (B ** 0.5)
        noise = torch.randn_like(clipped) * noise_std
        return clipped + noise

    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True when at least one task has σ > 0 configured."""
        return any(v > 0.0 for v in self._sigma.values())

    def sigma_for(self, task: str) -> float:
        """Return σ for the given task (0.0 if not set = no noise)."""
        return self._sigma.get(task, 0.0)

    def __repr__(self) -> str:
        return (
            f"AdaptiveDPSGD(max_grad_norm={self.max_grad_norm}, sigma={self._sigma})"
        )


# ---------------------------------------------------------------------------


class DPVFLClient(VFLClient):
    """
    VFLClient with differential privacy at the cut layer.

    Overrides receive_gradient() to apply AdaptiveDPSGD.clip_and_noise()
    before backpropagating the server gradient through the local LSTM encoder.
    All other behaviour (forward, eval_forward, FedAvg interface) is unchanged.

    Parameters
    ----------
    dp_mechanism : AdaptiveDPSGD
        Shared DP mechanism. Set σ via set_uniform() / set_stratified() before
        starting training.
    site : 'A' | 'B' | 'C'
        Determines which task σ is applied (A→ihm, B→decomp, C→pheno).
    All remaining kwargs are forwarded to VFLClient.__init__().
    """

    def __init__(
        self,
        dp_mechanism: AdaptiveDPSGD,
        site: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if site not in SITE_TASK_MAP:
            raise ValueError(f"Unknown site {site!r}. Expected one of {list(SITE_TASK_MAP)}")
        self._dp = dp_mechanism
        self._task = SITE_TASK_MAP[site]

    def receive_gradient(self, grad: Tensor) -> None:
        """Apply DP noise to grad then delegate to VFLClient.receive_gradient()."""
        if self._dp.is_enabled:
            grad = self._dp.clip_and_noise(grad, self._task)
        super().receive_gradient(grad)
