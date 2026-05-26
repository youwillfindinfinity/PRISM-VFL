"""
fl/client.py — VFL client: local encoder for one hospital site.

Training protocol per batch:
  1. forward()         — run LSTM, return detached embedding to server
  2. (server computes loss and gradients)
  3. receive_gradient() — apply server gradient through local LSTM, update weights

Detaching: the embedding returned to the server has its autograd link to the LSTM
severed. The server computes gradients w.r.t. this vector, then hands them back.
receive_gradient() uses those as the starting point for local backprop.
"""

from __future__ import annotations

import torch
from torch import Tensor

from model.encoder import SiteEncoder


class VFLClient:
    """
    Parameters
    ----------
    input_dim  : number of features at this site (7 / 4 / 3)
    hidden_dim : LSTM hidden size
    num_layers : stacked LSTM layers
    embed_dim  : embedding size sent to server at the cut layer
    dropout    : dropout between LSTM layers
    lr         : local Adam learning rate
    device     : cpu or cuda
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embed_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.encoder = SiteEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embed_dim=embed_dim,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self._local_embedding: Tensor | None = None  # kept for backward pass

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Run encoder; return a detached copy of the embedding for the server.
        The original (graph-linked) tensor is stored for receive_gradient().

        Returns: (B, embed_dim)
        """
        self.encoder.train()
        self._local_embedding = self.encoder(x.to(self.device), mask.to(self.device))
        return self._local_embedding.detach().requires_grad_(True)

    def receive_gradient(self, grad: Tensor) -> None:
        """
        Backprop the server's gradient through the local encoder and update weights.

        DP hook: subclass and override here to clip/add noise to grad.
        """
        assert self._local_embedding is not None, "call forward() first"
        self.optimizer.zero_grad()
        self._local_embedding.backward(grad.to(self.device))
        self.optimizer.step()
        self._local_embedding = None

    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode in eval mode (no gradient tracking). Returns (B, embed_dim)."""
        self.encoder.eval()
        return self.encoder(x.to(self.device), mask.to(self.device))

    # ------------------------------------------------------------------
    # FedAvg parameter access

    def get_encoder_params(self) -> dict:
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def set_encoder_params(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict)
