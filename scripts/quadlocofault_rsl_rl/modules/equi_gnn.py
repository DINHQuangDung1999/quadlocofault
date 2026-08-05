from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class C2EquivariantGCNLayer(nn.Module):
    """A graph-convolution layer equivariant to one reflection.

    The non-identity element of C2 acts by permuting nodes and optionally
    flipping feature signs. Equivariance is enforced with the Reynolds
    projection

        0.5 * (F(x) + T_out(F(T_in(x)))),

    where ``F`` is an ordinary shared-weight graph convolution and ``T`` is
    the reflection action. The node permutation must be an involution.

    This layer supplies the equivariant message-passing operation described by
    MS-PPO. Exact end-to-end morphological equivariance additionally requires
    the observation encoder and action decoder to use the appropriate physical
    sign representations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_permutation: Sequence[int] | torch.Tensor,
        input_sign: Sequence[float] | torch.Tensor | None = None,
        output_sign: Sequence[float] | torch.Tensor | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        permutation = torch.as_tensor(node_permutation, dtype=torch.long)
        if permutation.ndim != 1:
            raise ValueError("node_permutation must be a one-dimensional sequence.")
        if not torch.equal(permutation[permutation], torch.arange(permutation.numel())):
            raise ValueError("node_permutation must be an involution for a C2 action.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.register_buffer("node_permutation", permutation)
        self.register_buffer(
            "input_sign", self._make_sign(input_sign, permutation, in_dim, "input_sign")
        )
        self.register_buffer(
            "output_sign", self._make_sign(output_sign, permutation, out_dim, "output_sign")
        )

    @staticmethod
    def _make_sign(
        sign: Sequence[float] | torch.Tensor | None,
        permutation: torch.Tensor,
        feature_dim: int,
        name: str,
    ) -> torch.Tensor:
        if sign is None:
            return torch.ones(feature_dim)
        sign_tensor = torch.as_tensor(sign, dtype=torch.float)
        valid_shapes = ((feature_dim,), (permutation.numel(), feature_dim))
        if sign_tensor.shape not in valid_shapes:
            raise ValueError(
                f"{name} must have shape ({feature_dim},) or "
                f"({permutation.numel()}, {feature_dim}), got {tuple(sign_tensor.shape)}."
            )
        if not torch.all((sign_tensor == 1.0) | (sign_tensor == -1.0)):
            raise ValueError(f"{name} entries must be either +1 or -1.")
        if sign_tensor.ndim == 2:
            reflected_sign = sign_tensor.index_select(0, permutation)
            if not torch.all(sign_tensor * reflected_sign == 1.0):
                raise ValueError(f"{name} and node_permutation must define an involutive C2 action.")
        return sign_tensor

    def _reflect(self, features: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        return features.index_select(-2, self.node_permutation) * sign

    def _convolve(self, x_nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,bjd->bid", adjacency, x_nodes)
        return self.linear(aggregated)

    def forward(self, x_nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply C2-equivariant message passing.

        Args:
            x_nodes: Node features with shape ``(batch, num_nodes, in_dim)``.
            adjacency: Dense (optionally normalized) adjacency matrix with shape
                ``(num_nodes, num_nodes)``.
        """
        num_nodes = self.node_permutation.numel()
        if x_nodes.ndim != 3 or x_nodes.shape[1:] != (num_nodes, self.in_dim):
            raise ValueError(
                f"Expected x_nodes shape (batch, {num_nodes}, {self.in_dim}), got {tuple(x_nodes.shape)}."
            )
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Expected adjacency shape ({num_nodes}, {num_nodes}), got {tuple(adjacency.shape)}."
            )

        direct = self._convolve(x_nodes, adjacency)
        reflected_input = self._reflect(x_nodes, self.input_sign)
        reflected_output = self._convolve(reflected_input, adjacency)
        reflected_output = self._reflect(reflected_output, self.output_sign)
        return 0.5 * (direct + reflected_output)
