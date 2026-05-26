"""
fl/server.py — VFL server: aggregation, loss, and gradient distribution.

Training protocol per batch:
  1. aggregate_embeddings()  — concatenate the three site embeddings → (B, 192)
  2. forward_and_loss()      — run MMoEServer, compute per-task losses
  3. backward_and_step()     — backprop through MMoE, update server weights
  4. get_embedding_gradients() — slice the gradient of the concatenated embedding
                                  back into per-site pieces and return to clients

DP hook: subclass and override get_embedding_gradients() to add noise
before returning gradients to clients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.mmoe import MMoEServer


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

_BCE = nn.BCELoss()


def _weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """
    BCE with per-class positive-sample upweighting.  Works on Sigmoid outputs.

    Equivalent to BCEWithLogitsLoss(pos_weight=tensor([pw])) applied to logits,
    but operates directly on probabilities produced by the Sigmoid task head.

    weight per sample = pos_weight * target + (1 - target):
      positive sample → multiplied by pos_weight
      negative sample → multiplied by 1
    """
    w = pos_weight * target + (1.0 - target)
    return nn.functional.binary_cross_entropy(pred, target, weight=w)


class VFLServer:
    """
    Parameters
    ----------
    embed_dim         : per-site embedding size (default 64); concat input = 3 × embed_dim
    num_experts       : MMoE expert count (default 4)
    expert_hidden     : hidden size inside each expert MLP (default 128)
    lr                : Adam learning rate for MMoE + task heads
    device            : cpu or cuda
    task_weights      : relative loss weights for {'ihm', 'decomp', 'pheno'} (default equal).
                        Ignored when uncertainty_weighting=True.
    decomp_pos_weight : pos_weight for decompensation BCE loss (≈ N_neg/N_pos).
    uncertainty_weighting : if True, use Kendall et al. (2018) homoscedastic uncertainty
                        weighting. Learns log(σ_i²) per task; replaces task_weights.
                        Decompensation has ~2–5% positive rate; raw BCE collapses
                        to all-zero prediction.  Compute from training CSV and pass in.
                        Default 1.0 (no upweighting — use only for synthetic/smoke tests).
    """

    SITES = ("A", "B", "C")

    TASKS = ("ihm", "decomp", "pheno")

    def __init__(
        self,
        embed_dim: int = 64,
        num_experts: int = 4,
        expert_hidden: int = 128,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
        task_weights: dict[str, float] | None = None,
        n_sites: int = 3,
        decomp_pos_weight: float = 1.0,
        use_mmoe: bool = True,
        uniform_gating: bool = False,
        uncertainty_weighting: bool = False,
    ):
        self.device             = torch.device(device)
        self.embed_dim          = embed_dim
        self.SITES              = self.__class__.SITES[:n_sites]
        self.decomp_pos_weight  = decomp_pos_weight
        self.uncertainty_weighting = uncertainty_weighting

        self.model = MMoEServer(
            input_dim=n_sites * embed_dim,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
            expert_out=embed_dim,
            use_mmoe=use_mmoe,
            uniform_gating=uniform_gating,
        ).to(self.device)

        # Kendall et al. (2018): learn log(σ_i²) per task, init=0 → σ_i=1 at start.
        # Optimised jointly with MMoE via the same Adam instance.
        if uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1, device=self.device))
                for t in self.TASKS
            })
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.log_vars.parameters()), lr=lr
            )
        else:
            self.log_vars  = None
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.task_weights = task_weights or {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0}

        # Stored between aggregate_embeddings() and get_embedding_gradients()
        self._concat_embedding: Tensor | None = None

    # ------------------------------------------------------------------

    def aggregate_embeddings(self, embeddings: dict[str, Tensor]) -> Tensor:
        """
        Concatenate per-site embeddings into a single vector.

        Parameters
        ----------
        embeddings : {'A': (B, 64), 'B': (B, 64), 'C': (B, 64)}

        Returns
        -------
        (B, 192)  — stored internally; also returned for convenience
        """
        parts = [embeddings[s].to(self.device) for s in self.SITES]
        self._concat_embedding = torch.cat(parts, dim=-1)  # (B, 192)
        self._concat_embedding.retain_grad()  # non-leaf: opt in to keep .grad after backward
        return self._concat_embedding

    def forward_and_loss(
        self,
        labels: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Run MMoEServer and compute the weighted sum of per-task losses.

        Parameters
        ----------
        labels : {
            'ihm'   : (B,)    float32  — binary mortality label
            'decomp': (B,)    float32  — binary decompensation label
            'pheno' : (B, 25) float32  — multi-label phenotype flags
        }

        Returns
        -------
        total_loss : scalar Tensor
        task_losses: {'ihm': scalar, 'decomp': scalar, 'pheno': scalar}
        """
        assert self._concat_embedding is not None, "call aggregate_embeddings() first"

        self.model.train()
        preds = self.model(self._concat_embedding)  # dict[str, Tensor]

        task_losses = {
            "ihm":    _BCE(preds["ihm"].squeeze(-1),
                           labels["ihm"].to(self.device)),
            "decomp": _weighted_bce(preds["decomp"].squeeze(-1),
                                    labels["decomp"].to(self.device).float(),
                                    self.decomp_pos_weight),
            "pheno":  _BCE(preds["pheno"],
                           labels["pheno"].to(self.device)),
        }

        if self.uncertainty_weighting:
            # Kendall et al. (2018): L = Σ_i [ exp(-s_i)/2 · L_i + s_i/2 ]
            # where s_i = log(σ_i²). Precision exp(-s_i) down-weights high-variance tasks;
            # s_i/2 regularises to prevent σ_i → ∞.
            # Only include tasks with weight > 0 so zero-weight tasks are truly inactive.
            total_loss = sum(
                0.5 * torch.exp(-self.log_vars[t]) * loss + 0.5 * self.log_vars[t]
                for t, loss in task_losses.items()
                if self.task_weights.get(t, 1.0) > 0
            )
        else:
            total_loss = sum(
                self.task_weights[t] * loss for t, loss in task_losses.items()
            )
        return total_loss, task_losses

    def backward_and_step(self, total_loss: Tensor) -> None:
        """Backprop through MMoE and update server weights."""
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def get_embedding_gradients(self) -> dict[str, Tensor]:
        """
        Slice the gradient of the concatenated embedding back into per-site pieces.

        Called after backward_and_step(). Each client receives its own 64-dim slice.

        DP hook: subclass and add noise to grad slices here.

        Returns
        -------
        {'A': (B, 64), 'B': (B, 64), 'C': (B, 64)}
        """
        assert self._concat_embedding is not None and \
               self._concat_embedding.grad is not None, \
            "call backward_and_step() before get_embedding_gradients()"

        grad = self._concat_embedding.grad  # (B, 192)
        slices = grad.split(self.embed_dim, dim=-1)  # three (B, 64) tensors
        return dict(zip(self.SITES, slices))

    # ------------------------------------------------------------------

    def compute_task_gradient_similarity(self, labels: dict[str, Tensor]) -> dict[str, float]:
        """
        Pairwise cosine similarity between per-task gradients at shared expert parameters.

        Yu et al. (2020) NeurIPS "Gradient Surgery for Multi-Task Learning".
        Negative values indicate conflicting gradients (potential negative transfer).
        Tasks with weight 0.0 are skipped; their similarity entries are NaN.

        Call after aggregate_embeddings() and backward_and_step(). Uses a detached
        copy of the concatenated embedding so the training graph is unaffected.
        """
        assert self._concat_embedding is not None, "call aggregate_embeddings() first"
        emb = self._concat_embedding.detach()

        if self.model.use_mmoe:
            shared_params = (
                list(self.model.mmoe.experts.parameters())
                + list(self.model.mmoe.gates.parameters())
            )
        else:
            shared_params = list(self.model.shared_bottom.parameters())

        task_grads: dict[str, Tensor | None] = {}
        self.model.train()

        for task in self.TASKS:
            if self.task_weights.get(task, 1.0) == 0.0:
                task_grads[task] = None
                continue

            self.model.zero_grad()
            preds = self.model(emb)

            if task == "ihm":
                loss = _BCE(preds["ihm"].squeeze(-1), labels["ihm"].to(self.device))
            elif task == "decomp":
                loss = _weighted_bce(
                    preds["decomp"].squeeze(-1),
                    labels["decomp"].to(self.device).float(),
                    self.decomp_pos_weight,
                )
            else:
                loss = _BCE(preds["pheno"], labels["pheno"].to(self.device))

            loss.backward()

            g_parts = [p.grad.detach().flatten() for p in shared_params if p.grad is not None]
            task_grads[task] = torch.cat(g_parts) if g_parts else None

        self.model.zero_grad()

        def _cos(a: Tensor | None, b: Tensor | None) -> float:
            if a is None or b is None:
                return float("nan")
            return float(
                torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            )

        return {
            "grad_sim_ihm_decomp":   _cos(task_grads.get("ihm"),   task_grads.get("decomp")),
            "grad_sim_ihm_pheno":    _cos(task_grads.get("ihm"),   task_grads.get("pheno")),
            "grad_sim_decomp_pheno": _cos(task_grads.get("decomp"), task_grads.get("pheno")),
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, embeddings: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Run inference (no gradient tracking).

        Returns raw model outputs: probabilities for IHM/pheno, logits for LOS.
        """
        self.model.eval()
        concat = torch.cat(
            [embeddings[s].to(self.device) for s in self.SITES], dim=-1
        )
        return self.model(concat)
