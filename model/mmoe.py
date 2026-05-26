"""
model/mmoe.py — Server-side Mixture-of-Experts for VFL-MTL.

Receives the concatenated site embeddings [h_A; h_B; h_C] (B, 192) and
routes them through shared expert MLPs with per-task gating networks,
following Ma et al. (2018) KDD MMoE.

Architecture
------------
Input: concat embedding (B, input_dim=192)

MMoELayer:
  4 shared ExpertMLPs:  input_dim → expert_hidden → expert_out_dim
  3 per-task gating networks: Linear(input_dim, num_experts) + Softmax
  Output: 3 task-specific vectors, each (B, expert_out_dim=64)

Task heads:
  IHM   (Site A) → binary      : Linear(64, 1)  + Sigmoid
  Decomp (Site B) → binary     : Linear(64, 1)  + Sigmoid
  Pheno (Site C) → multi-label : Linear(64, 25) + Sigmoid

MMoEServer:
  Wraps MMoELayer + all three task heads.
  forward(concat_embedding) → dict[str, Tensor]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Expert MLP
# ---------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    """Single shared expert: input_dim → hidden_dim → output_dim with ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MMoE Layer
# ---------------------------------------------------------------------------

class MMoELayer(nn.Module):
    """
    Mixture-of-Experts layer with per-task gating (Ma et al., 2018).

    Parameters
    ----------
    input_dim      : dimension of the concatenated embedding (3 × 64 = 192)
    num_experts    : number of shared expert MLPs (default 4)
    expert_hidden  : hidden size inside each expert MLP
    expert_out     : output size of each expert (= embed_dim fed to task heads)
    num_tasks      : number of prediction tasks (3: IHM, Decomp, Pheno)
    uniform_gating : if True, replace learned softmax gating with fixed equal
                     weights (1/num_experts per expert). Ablation 4.
    """

    def __init__(
        self,
        input_dim: int = 192,
        num_experts: int = 4,
        expert_hidden: int = 128,
        expert_out: int = 64,
        num_tasks: int = 3,
        uniform_gating: bool = False,
    ):
        super().__init__()
        self.num_experts    = num_experts
        self.num_tasks      = num_tasks
        self.uniform_gating = uniform_gating

        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, expert_hidden, expert_out)
            for _ in range(num_experts)
        ])
        # One softmax gating network per task (unused when uniform_gating=True)
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False)
            for _ in range(num_tasks)
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        list of num_tasks tensors, each (B, expert_out)
        """
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, num_experts, expert_out)

        task_outputs = []
        for gate in self.gates:
            if self.uniform_gating:
                # Ablation 4: fixed equal weights — no learned routing
                weights = torch.full(
                    (x.size(0), self.num_experts),
                    1.0 / self.num_experts,
                    device=x.device,
                )
            else:
                weights = torch.softmax(gate(x), dim=-1)  # (B, num_experts)
            out = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, expert_out)
            task_outputs.append(out)

        return task_outputs  # 3 × (B, expert_out)


# ---------------------------------------------------------------------------
# Task Heads
# ---------------------------------------------------------------------------

class TaskHead(nn.Module):
    """
    Per-task output head.

    task_type : 'binary'    → Linear(in, 1)  + Sigmoid     → (B, 1)
                'los_bins'  → Linear(in, n_classes)         → (B, 10) logits
                'multilabel'→ Linear(in, n_classes) + Sigmoid → (B, 25)
    """

    def __init__(self, in_dim: int, task_type: str, n_classes: int = 1):
        super().__init__()
        assert task_type in ("binary", "los_bins", "multilabel")
        self.task_type = task_type

        if task_type == "binary":
            self.head = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        elif task_type == "los_bins":
            self.head = nn.Linear(in_dim, n_classes)   # raw logits
        else:  # multilabel
            self.head = nn.Sequential(nn.Linear(in_dim, n_classes), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# MMoEServer
# ---------------------------------------------------------------------------

class MMoEServer(nn.Module):
    """
    Server-side model for VFL-MTL.

    Expects the concatenated embedding [h_A; h_B; h_C] and returns
    per-task predictions.

    Parameters
    ----------
    input_dim      : size of concatenated embedding (default 192 = 3 × 64)
    num_experts    : shared expert count (default 4)
    expert_hidden  : expert MLP hidden size (default 128)
    expert_out     : expert output / task head input size (default 64)
    use_mmoe       : if False, replace MMoE with a shared-bottom MLP (Abl 1)
    uniform_gating : if True, use fixed equal expert weights instead of learned
                     softmax gating (Abl 4); ignored when use_mmoe=False
    """

    # Task order is fixed; names are used as dict keys throughout the codebase.
    TASK_ORDER = ("ihm", "decomp", "pheno")

    def __init__(
        self,
        input_dim: int = 192,
        num_experts: int = 4,
        expert_hidden: int = 128,
        expert_out: int = 64,
        use_mmoe: bool = True,
        uniform_gating: bool = False,
    ):
        super().__init__()
        self.use_mmoe = use_mmoe

        if use_mmoe:
            self.mmoe = MMoELayer(
                input_dim=input_dim,
                num_experts=num_experts,
                expert_hidden=expert_hidden,
                expert_out=expert_out,
                num_tasks=len(self.TASK_ORDER),
                uniform_gating=uniform_gating,
            )
        else:
            # Ablation 1: single shared-bottom MLP, no gating
            self.shared_bottom = nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_out),
                nn.ReLU(),
            )

        self.heads = nn.ModuleDict({
            "ihm":    TaskHead(expert_out, "binary"),
            "decomp": TaskHead(expert_out, "binary"),
            "pheno":  TaskHead(expert_out, "multilabel", n_classes=25),
        })

    def forward(self, concat_embedding: Tensor) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        concat_embedding : (B, input_dim)  — [h_A; h_B; h_C]

        Returns
        -------
        dict with keys 'ihm', 'decomp', 'pheno' and tensor values:
          'ihm'   : (B, 1)   float32  probabilities
          'decomp': (B, 1)   float32  probabilities
          'pheno' : (B, 25)  float32  probabilities
        """
        if self.use_mmoe:
            task_vecs = self.mmoe(concat_embedding)  # 3 × (B, expert_out)
        else:
            shared = self.shared_bottom(concat_embedding)  # (B, expert_out)
            task_vecs = [shared] * len(self.TASK_ORDER)
        return {
            task: self.heads[task](vec)
            for task, vec in zip(self.TASK_ORDER, task_vecs)
        }
