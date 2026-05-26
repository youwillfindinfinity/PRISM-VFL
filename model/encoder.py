"""
model/encoder.py — Per-party LSTM encoder for VFL-MTL.

Each hospital site runs a SiteEncoder locally. It consumes the site's
feature time-series and outputs a fixed-size embedding vector that is
transmitted to the VFL server (the cut layer).

Input:  x    (B, T, input_dim)   float32 — padded time-series
        mask (B, T)              float32 — 1=real timestep, 0=padding
Output: (B, embed_dim)           float32 — site embedding

Architecture:
  LSTM(input_dim → hidden_dim, num_layers, bidirectional=False, dropout)
  → take last *real* hidden state (not last padded step)
  → Linear(hidden_dim → embed_dim)
  → LayerNorm(embed_dim)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SiteEncoder(nn.Module):
    """
    Parameters
    ----------
    input_dim  : number of features at this site (7 / 4 / 3)
    hidden_dim : LSTM hidden size (default 128)
    num_layers : stacked LSTM layers (default 2)
    embed_dim  : output embedding size transmitted at cut layer (default 64)
    dropout    : dropout between LSTM layers (ignored when num_layers=1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, embed_dim)
        self.norm       = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x    : (B, T, input_dim)
        mask : (B, T)  float32, 1=real timestep, 0=padding

        Returns
        -------
        (B, embed_dim)
        """
        # Sequence lengths from mask: number of real timesteps per sample.
        # Clamp to at least 1 to avoid pack_padded_sequence errors on all-zero masks.
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, B, hidden_dim) — take last layer
        last_hidden = h_n[-1]               # (B, hidden_dim)

        out = self.projection(last_hidden)  # (B, embed_dim)
        out = self.norm(out)
        return out
