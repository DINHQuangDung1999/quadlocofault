from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import GCNLayer

from .gcn_actor import GCNActor


class EquivGCNTemporalEncoder(nn.Module):
    """C2-equivariant temporal GCN with two mirrored seven-node halves.

    Node order:
        0: left base copy
        1..3: FL hip, thigh, calf
        4..6: RL hip, thigh, calf
        7: right base copy
        8..10: FR hip, thigh, calf
        11..13: RR hip, thigh, calf

    The right-side physical observations are mapped into the left-side
    canonical coordinates before shared temporal encoding. This follows the
    sign-aware encoder / equivariant GNN separation used by MS-PPO.
    """

    _C2_NODE_PERMUTATION = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6]
    # Reorder Isaac joint features from
    # [FL/FR/RL/RR hip, FL/FR/RL/RR thigh, FL/FR/RL/RR calf]
    # to [FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf,
    # RR hip/thigh/calf].
    _JOINT_PERMUTATION = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    # Select joint nodes from the graph in the original Isaac joint order.
    _JOINT_NODE_IDS = [1, 8, 4, 11, 2, 9, 5, 12, 3, 10, 6, 13]

    def __init__(
        self,
        node_dim: int,
        node_base_dim: int,
        gcn_hidden_dim: int,
        gcn_out_dim: int,
    ) -> None:
        super().__init__()
        if node_dim != 3:
            raise ValueError(f"EquivGCNTemporalEncoder expects three joint features, got {node_dim}.")
        if node_base_dim != 9:
            raise ValueError(f"EquivGCNTemporalEncoder expects nine base features, got {node_base_dim}.")

        self.node_dim = node_dim
        self.node_base_dim = node_base_dim
        self.num_joints = 12

        self.register_buffer("joint_permutation", torch.tensor(self._JOINT_PERMUTATION, dtype=torch.long))
        self.register_buffer("joint_node_ids", torch.tensor(self._JOINT_NODE_IDS, dtype=torch.long))

        # Sagittal reflection signs for [angular velocity, gravity, command].
        self.register_buffer(
            "base_reflection_sign",
            torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0]),
        )

        adjacency = self._build_adjacency()
        self.register_buffer("adj_norm", self._normalize_adjacency(adjacency))

        self.mlp_joint = nn.Sequential(
            nn.LazyLinear(128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 32)
        )
        self.mlp_base = nn.Sequential(
            nn.LazyLinear(128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 32)
        )
        # Shared node weights and a reflection-invariant adjacency make these
        # graph convolutions equivariant to the left/right node permutation.
        self.gcn1 = GCNLayer(32, gcn_hidden_dim)
        self.gcn2 = GCNLayer(gcn_hidden_dim, gcn_out_dim)

    @staticmethod
    def _build_adjacency() -> torch.Tensor:
        edges = [
            (0, 1), (1, 2), (2, 3),
            (0, 4), (4, 5), (5, 6),
            (0, 7),
            (7, 8), (8, 9), (9, 10),
            (7, 11), (11, 12), (12, 13),
        ]
        adjacency = torch.eye(14)
        for source, target in edges:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        return adjacency

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        degree_inv_sqrt = adjacency.sum(dim=1).pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    def _encode_temporal(self, joints: torch.Tensor, bases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joints = joints.permute(0, 2, 1, 3).flatten(2, 3)
        joints = self.mlp_joint(joints)
        bases = bases.permute(0, 2, 1, 3).flatten(2, 3)
        bases = self.mlp_base(bases)
        return joints, bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 45:
            raise ValueError(f"Expected history shape (batch, history, 45), got {tuple(x.shape)}.")

        position = x[:, :, :12]
        velocity = x[:, :, 12:24]
        previous_action = x[:, :, 24:36]
        base = x[:, :, 36:45]

        joints = torch.stack(
            (
                position[:, :, self.joint_permutation],
                velocity[:, :, self.joint_permutation],
                previous_action[:, :, self.joint_permutation],
            ),
            dim=-1,
        )

        # Canonicalize right hips (FR and RR); thigh/calf coordinates already
        # share the left-side convention for the Go2 joint definitions.
        joints = joints.clone()
        joints[:, :, (3, 9), :] *= -1.0
        bases = torch.stack((base, base * self.base_reflection_sign), dim=2)
        joints, bases = self._encode_temporal(joints, bases)

        graph = torch.stack(
            (
                bases[:, 0], joints[:, 0], joints[:, 1], joints[:, 2],
                joints[:, 6], joints[:, 7], joints[:, 8], bases[:, 1],
                joints[:, 3], joints[:, 4], joints[:, 5], joints[:, 9],
                joints[:, 10], joints[:, 11],
            ),
            dim=1,
        )
        hidden = F.elu(self.gcn1(graph, self.adj_norm))
        return F.elu(self.gcn2(hidden, self.adj_norm))


class EquivGCNActor(GCNActor):
    """GCN actor variant using the duplicated-base 14-node Equ-GNN encoder."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gcn_encoder = EquivGCNTemporalEncoder(
            node_dim=3,
            node_base_dim=9,
            gcn_hidden_dim=self.gcn_hidden_dim,
            gcn_out_dim=self.gcn_out_dim,
        )

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        if self.setup != 1:
            return super().forward(obs, masks, hidden_state, stochastic_output)

        obs_policy = self.obs_normalizer(obs["policy"])
        obs_history = self.obs_hist_normalizer(obs["history"])
        gcn_code = self.gcn_encoder(obs_history)
        joint_code = gcn_code.index_select(1, self.gcn_encoder.joint_node_ids)
        fault_logits = self.fault_predictor(joint_code).squeeze(-1)
        actor_input = torch.cat(
            (obs_policy, torch.sigmoid(fault_logits), gcn_code.mean(dim=1)), dim=-1
        )
        mlp_output = self.actor_mlp(actor_input)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                action = self.distribution.sample()
            else:
                action = self.distribution.deterministic_output(mlp_output)
        else:
            action = mlp_output
        return action, (self.setup, fault_logits, None)
