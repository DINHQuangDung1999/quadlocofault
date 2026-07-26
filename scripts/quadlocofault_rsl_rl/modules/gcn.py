from __future__ import annotations

import torch
from torch import nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_nodes: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,bjd->bid", adj_norm, x_nodes)
        return self.linear(x)
