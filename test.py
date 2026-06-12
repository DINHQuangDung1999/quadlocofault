from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor, nn

# class MambaBlock(nn.Module):
#     """
#     Initialize a single Mamba block.

#     Args:
#         dim (int): The input dimension.
#         dim_inner (Optional[int]): The inner dimension. If not provided, it is set to dim * expand.
#         depth (int): The depth of the Mamba block.
#         d_state (int): The state dimension. Default is 16.
#         expand (int): The expansion factor. Default is 2.
#         dt_rank (Union[int, str]): The rank of the temporal difference (Δ) tensor. Default is "auto".
#         d_conv (int): The dimension of the convolutional kernel. Default is 4.
#         conv_bias (bool): Whether to include bias in the convolutional layer. Default is True.
#         bias (bool): Whether to include bias in the linear layers. Default is False.

#     Examples:
#         >>> import torch
#         >>> from zeta.nn.modules.simple_mamba import MambaBlock
#         >>> block = MambaBlock(dim=64, depth=1)
#         >>> x = torch.randn(1, 10, 64)
#         >>> y = block(x)
#         >>> y.shape
#         torch.Size([1, 10, 64])
#     """

#     def __init__(
#         self,
#         dim: int = 16,
#         depth: int = 5,
#         d_state: int = 16,
#         expand: int = 2,
#         d_conv: int = 4,
#         conv_bias: bool = True,
#         bias: bool = False,
#     ):
#         """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
#         super().__init__()
#         self.dim = dim
#         self.depth = depth
#         self.d_state = d_state
#         self.expand = expand
#         self.d_conv = d_conv
#         self.conv_bias = conv_bias
#         self.bias = bias

#         # If dt_rank is not provided, set it to ceil(dim / d_state)
#         dt_rank = math.ceil(self.dim / 16)
#         self.dt_rank = dt_rank

#         # If dim_inner is not provided, set it to dim * expand
#         dim_inner = dim * expand
#         self.dim_inner = dim_inner

#         # If dim_inner is not provided, set it to dim * expand
#         self.in_proj = nn.Linear(dim, dim_inner * 2, bias=bias)

#         self.conv1d = nn.Conv1d(
#             in_channels=dim_inner,
#             out_channels=dim_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=dim_inner,
#             padding=d_conv - 1,
#         )

#         # x_proj takes in `x` and outputs the input-specific Δ, B, C
#         self.x_proj = nn.Linear(
#             dim_inner, dt_rank + self.d_state * 2, bias=False
#         )

#         # dt_proj projects Δ from dt_rank to d_in
#         self.dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)

#         A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=dim_inner)
#         self.A_log = nn.Parameter(torch.log(A))
#         self.D = nn.Parameter(torch.ones(dim_inner))
#         self.out_proj = nn.Linear(dim_inner, dim, bias=bias)

#     def forward(self, x: Tensor):
#         """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

#         Args:
#             x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

#         Returns:
#             output: shape (b, l, d)


#         Official Implementation:
#             class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
#             mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

#         """
#         (b, l, d) = x.shape

#         x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
#         x_and_res = rearrange(x_and_res, "b l x -> b x l")
#         (x, res) = x_and_res.split(
#             split_size=[self.dim_inner, self.dim_inner], dim=1
#         )

#         x = self.conv1d(x)[:, :, :l]
#         x = F.silu(x)

#         y = self.ssm(x)

#         y = y * F.silu(res)

#         output = self.out_proj(rearrange(y, "b dim l -> b l dim"))

#         return output

#     def ssm(self, x: Tensor):
#         """Runs the SSM. See:
#             - Algorithm 2 in Section 3.2 in the Mamba paper [1]
#             - run_SSM(A, B, C, u) in The Annotated S4 [2]

#         Args:
#             x: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)

#         Returns:
#             output: shape (b, d_in, l)

#         Official Implementation:
#             mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

#         """
#         (d_in, n) = self.A_log.shape

#         # Compute ∆ A B C D, the state space parameters.
#         #     A, D are input independent
#         #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4)

#         A = -torch.exp(self.A_log.float())  # shape (d_in, n)
#         D = self.D.float()

#         x_dbl = rearrange(x, "b d l -> b l d")
#         x_dbl = self.x_proj(x_dbl)  # (b, l, dt_rank + 2*n)

#         (delta, B, C) = x_dbl.split(
#             split_size=[self.dt_rank, n, n], dim=-1
#         )  # delta: (b, l, dt_rank). B, C: (b, l, n)
#         delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

#         y = self.selective_scan(
#             x, delta, A, B, C, D
#         )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

#         return y

#     def selective_scan(self, u, delta, A, B, C, D):
#         """Does selective scan algorithm. See:
#             - Section 2 State Space Models in the Mamba paper [1]
#             - Algorithm 2 in Section 3.2 in the Mamba paper [1]
#             - run_SSM(A, B, C, u) in The Annotated S4 [2]

#         This is the classic discrete state space formula:
#             x(t + 1) = Ax(t) + Bu(t)
#             y(t)     = Cx(t) + Du(t)
#         except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

#         Args:
#             u: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)
#             delta: shape (b, l, d_in)
#             A: shape (d_in, n)
#             B: shape (b, l, n)
#             C: shape (b, l, n)
#             D: shape (d_in,)

#         Returns:
#             output: shape (b, d_in, l)

#         Official Implementation:
#             selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
#             Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

#         """
#         (b, d_in, l) = u.shape
#         n = A.shape[1]

#         # Discretize continuous parameters (Δ, A, B)  (see Section 2 Equation 4 in the Mamba paper [1])
#         # Note that B is parameterized directly
#         deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b d_in l n"))
#         deltaB_u = einsum(
#             delta, B, u, "b l d_in, b l n, b d_in l -> b d_in l n"
#         )

#         # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
#         x = torch.zeros((b, d_in, n), device=next(self.parameters()).device)
#         ys = []
#         for i in range(l):
#             x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
#             y = einsum(x, C[:, i, :], "b d_in n , b n -> b d_in")
#             ys.append(y)
#         y = torch.stack(ys, dim=2)  # (b d_in l)

#         if D is not None:
#             y = y + u * rearrange(D, "d_in -> d_in 1")
#         breakpoint()
#         return y

# dim = 49
# batch = 4
# length = 50
# x = torch.rand(batch, length, dim)
# m = MambaBlock(dim=dim, bias = True)
# m.forward(x)
# breakpoint()
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_nodes, adj_norm):
        """
        x_nodes:  [B, V, in_dim]
        adj_norm: [V, V]
        """
        h = torch.einsum("ij,bjd->bid", adj_norm, x_nodes)
        h = self.linear(h)
        return h


class GCNTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_dim,
        node_base_dim,
        projection_dim,
        gcn_hidden_dim,
        temporal_hidden_dim,
        edges,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.node_base_dim = node_base_dim
        self.num_nodes = num_nodes
        self.num_joints = num_nodes - 1

        adj = self.build_adj(self.num_nodes, edges)
        adj_norm = self.normalize_adj(adj)
        self.register_buffer("adj_norm", adj_norm)

        self.node_base_projection = nn.Linear(node_base_dim, projection_dim)
        self.node_joint_projection = nn.Linear(node_dim, projection_dim)

        # Important: input dim is projection_dim, not projection_dim * 2
        self.gcn1 = GCNLayer(projection_dim, gcn_hidden_dim)
        self.gcn2 = GCNLayer(gcn_hidden_dim, gcn_hidden_dim)

        self.temporal_encoder = nn.GRU(
            input_size=gcn_hidden_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )

        self.latent_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
        )

    def build_adj(self, num_nodes, edges):
        adj = torch.zeros(num_nodes, num_nodes)

        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        adj = adj + torch.eye(num_nodes)

        return adj

    def normalize_adj(self, adj):
        degree = adj.sum(dim=1)

        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return adj_norm

    def forward(self, x):
        """
        x: [N, H, D]

        Assumes:
        x[:, :, :6]   = base features
        x[:, :, 6:]   = joint features
        """

        N, H, D = x.shape

        expected_D = self.node_base_dim + self.num_joints * self.node_dim
        assert D == expected_D, f"Expected D={expected_D}, but got D={D}"

        # Base node
        # [N, H, 6]
        x_base = x[:, :, :self.node_base_dim]

        # Joint nodes
        # [N, H, 12 * joint_dim] -> [N, H, 12, joint_dim]
        x_joints = x[:, :, self.node_base_dim:].view(
            N, H, self.num_joints, self.node_dim
        )

        # Project raw features to common projection dimension
        # [N, H, 12, projection_dim]
        x_joints = self.node_joint_projection(x_joints)

        # [N, H, 6] -> [N, H, projection_dim]
        x_base = self.node_base_projection(x_base)

        # [N, H, projection_dim] -> [N, H, 1, projection_dim]
        x_base = x_base.unsqueeze(2)

        # Combine joint nodes and base node
        # [N, H, 13, projection_dim]
        x_nodes = torch.cat([x_joints, x_base], dim=2)

        # Merge batch and time before GCN
        # [N, H, 13, projection_dim] -> [N * H, 13, projection_dim]
        x_nodes = x_nodes.reshape(N * H, self.num_nodes, -1)

        # GCN over robot graph at each timestep
        h = self.gcn1(x_nodes, self.adj_norm)
        h = F.relu(h)

        h = self.gcn2(h, self.adj_norm)
        h = F.relu(h)

        # Pool over graph nodes
        # [N * H, 13, gcn_hidden_dim] -> [N * H, gcn_hidden_dim]
        h = h.mean(dim=1)

        # Restore temporal shape
        # [N * H, gcn_hidden_dim] -> [N, H, gcn_hidden_dim]
        h = h.view(N, H, -1)

        # Temporal encoder
        out, _ = self.temporal_encoder(h)

        # Last timestep
        last = out[:, -1, :]

        z = self.latent_head(last)

        return z

N = 32
H = 50

num_joints = 12
joint_dim = 3
base_dim = 6

history = torch.randn(N, H, base_dim + num_joints * joint_dim)

# In zero-based indexing:
edges = [
    (0, 1), (1, 2),        # FL leg
    (3, 4), (4, 5),        # FR leg
    (6, 7), (7, 8),        # RL leg
    (9, 10), (10, 11),     # RR leg
    (12, 0),               # base -- FL hip
    (12, 3),               # base -- FR hip
    (12, 6),               # base -- RL hip
    (12, 9),               # base -- RR hip
]

encoder = GCNTemporalEncoder(
    num_nodes=num_joints+1,
    node_dim=joint_dim,
    node_base_dim = 6,
    projection_dim = 16,
    gcn_hidden_dim=32,
    temporal_hidden_dim=32,
    edges=edges,
)
breakpoint()
y = encoder(history)

print(y.shape)