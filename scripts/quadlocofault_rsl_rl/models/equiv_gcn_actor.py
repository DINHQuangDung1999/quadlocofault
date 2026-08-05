from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from modules import GCNLayer, TemporalConvBlock
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


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
    # Reorder Isaac joint features from
    # [FL/FR/RL/RR hip, FL/FR/RL/RR thigh, FL/FR/RL/RR calf]
    # directly into graph order:
    # [FL hip/thigh/calf, RL hip/thigh/calf, FR hip/thigh/calf,
    # RR hip/thigh/calf].
    _JOINT_PERMUTATION = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

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

        # Canonicalize right hips (FR and RR, at graph-joint indices 6 and 9);
        # thigh/calf coordinates already share the left-side convention for
        # the Go2 joint definitions.
        joints = joints.clone()
        joints[:, :, (6, 9), :] *= -1.0
        bases = torch.stack((base, base * self.base_reflection_sign), dim=2)
        joints, bases = self._encode_temporal(joints, bases)

        graph = torch.stack(
            (
                bases[:, 0], 
                joints[:, 0], joints[:, 1], joints[:, 2],
                joints[:, 3], joints[:, 4], joints[:, 5], 
                bases[:, 1],
                joints[:, 6], joints[:, 7], joints[:, 8], 
                joints[:, 9], joints[:, 10], joints[:, 11],
            ),
            dim=1,
        )
        hidden = F.elu(self.gcn1(graph, self.adj_norm))
        return F.elu(self.gcn2(hidden, self.adj_norm))


class FaultResidualTCN(nn.Module):
    """Encode global history into joint-fault logits and FiLM parameters."""

    _LEFT_JOINT_IDS = [0, 2, 4, 6, 8, 10]
    _RIGHT_JOINT_IDS = [1, 3, 5, 7, 9, 11]
    _RIGHT_HIP_IDS = [1, 3]

    def __init__(
        self,
        hidden_dim: int,
        film_dim: int,
        film_scale: float = 0.5,
        num_fault_classes: int = 12,
    ) -> None:
        super().__init__()
        if num_fault_classes not in (12, 13):
            raise ValueError(
                f"num_fault_classes must be 12 or 13, got {num_fault_classes}."
            )
        self.film_scale = film_scale
        self.num_fault_classes = num_fault_classes
        self.register_buffer("left_joint_ids", torch.tensor(self._LEFT_JOINT_IDS, dtype=torch.long))
        self.register_buffer("right_joint_ids", torch.tensor(self._RIGHT_JOINT_IDS, dtype=torch.long))
        self.register_buffer("right_hip_ids", torch.tensor(self._RIGHT_HIP_IDS, dtype=torch.long))

        self.tcn = nn.Sequential(
            TemporalConvBlock(45, 2*hidden_dim, kernel_size=3, dilation=1),
            TemporalConvBlock(2*hidden_dim, 2*hidden_dim, kernel_size=3, dilation=2),
            TemporalConvBlock(2*hidden_dim, 2*hidden_dim, kernel_size=3, dilation=4),
            TemporalConvBlock(2*hidden_dim, hidden_dim, kernel_size=3, dilation=8),
        )
        self.fault_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, num_fault_classes),
        )
        self.film_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2 * film_dim),
        )
        final_film_layer = self.film_head[-1]
        nn.init.zeros_(final_film_layer.weight)
        nn.init.zeros_(final_film_layer.bias)

    def _symmetry_features(self, history: torch.Tensor) -> torch.Tensor:
        """Return pair means/differences and base features with shape [B, H, 45]."""
        position = history[:, :, :12].clone()
        velocity = history[:, :, 12:24].clone()
        previous_action = history[:, :, 24:36].clone()
        base = history[:, :, 36:45]

        # Express right hip signals in the corresponding left-hip coordinates.
        position[:, :, self.right_hip_ids] *= -1.0
        velocity[:, :, self.right_hip_ids] *= -1.0
        previous_action[:, :, self.right_hip_ids] *= -1.0

        means = []
        differences = []
        for feature in (position, velocity, previous_action):
            left = feature.index_select(2, self.left_joint_ids)
            right = feature.index_select(2, self.right_joint_ids)
            means.append(0.5 * (left + right))
            differences.append(0.5 * (left - right))

        return torch.cat((*means, *differences, base), dim=-1)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[-1] != 45:
            raise ValueError(f"Expected history shape (batch, history, 45), got {tuple(history.shape)}.")

        tcn_input = self._symmetry_features(history).permute(0, 2, 1)
        fault_features = self.tcn(tcn_input)[:, :, -1]
        fault_logits = self.fault_head(fault_features)
        raw_gamma, raw_beta = self.film_head(fault_features.detach()).chunk(2, dim=-1)
        gamma = self.film_scale * torch.tanh(raw_gamma)
        beta = self.film_scale * torch.tanh(raw_beta)
        return fault_logits, gamma, beta


class EquivGCNActor(nn.Module):
    """Actor with a symmetry-biased temporal GCN history encoder."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        output_dim: int = 12,
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        gcn_hidden_dim: int = 16,
        gcn_out_dim: int = 16,
        tcn_hidden_dim: int = 16,
        setup: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        if setup != 1:
            raise ValueError("EquivGCNActor currently supports only setup=1.")

        self.obs_hist_length, self.obs_dim = obs["history"].shape[1:]
        self.action_dim = output_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn_out_dim = gcn_out_dim
        self.setup = setup

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.obs_hist_normalizer = EmpiricalNormalization((self.obs_hist_length, self.obs_dim))
        else:
            self.obs_normalizer = nn.Identity()
            self.obs_hist_normalizer = nn.Identity()

        if distribution_cfg is not None:
            distribution_cfg = distribution_cfg.copy()
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.gcn_encoder = EquivGCNTemporalEncoder(
            node_dim=3,
            node_base_dim=9,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_out_dim=gcn_out_dim,
        )
        self.fault_residual_encoder = FaultResidualTCN(
            hidden_dim=tcn_hidden_dim,
            film_dim=gcn_out_dim,
        )
        # tcn_path = "/home/dung-admin/quadloco_ws/quadlocofault/datasets/prop_fault/FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0/2026-07-27_11-45-57/fault_tcn_runs/2026-07-27_11-53-18/best.pt"
        # self.fault_residual_encoder.load_state_dict(
        #     torch.load(tcn_path, weights_only=False)['classifier_state_dict'], 
        #     strict=False
        #     )
        # # Freeze the temporal feature extractor.
        # self.fault_residual_encoder.tcn.requires_grad_(False)

        # # Explicitly keep both output heads trainable.
        # self.fault_residual_encoder.fault_head.requires_grad_(True)
        # self.fault_residual_encoder.film_head.requires_grad_(True)
        
        actor_input_dim = self.obs_dim + self.action_dim + gcn_out_dim
        self.actor_mlp = MLP(actor_input_dim, mlp_output_dim, actor_hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[int, torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ]:
        obs_policy = self.obs_normalizer(obs["policy"])
        obs_history = self.obs_hist_normalizer(obs["history"])
        gcn_code = self.gcn_encoder(obs_history)
        gcn_latent = gcn_code.mean(dim=1)
        fault_logits, gamma, beta = self.fault_residual_encoder(obs_history)
        # fault_probability = torch.sigmoid(fault_logits).detach()
        pos_weight = fault_logits.new_full(
                    (fault_logits.shape[-1],), 1.0
                ).to(fault_logits.device)
        calibrated_logits = fault_logits - torch.log(
            torch.as_tensor(pos_weight, device=fault_logits.device)
        )
        fault_probability = torch.sigmoid(calibrated_logits).detach()
        fault_gate = fault_probability.amax(dim=1, keepdim=True)
        fused_latent = (1.0 + fault_gate * gamma) * gcn_latent + fault_gate * beta
        actor_input = torch.cat(
            (obs_policy, fault_probability, fused_latent), dim=-1
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
        return action, (self.setup, fault_logits, (gamma, beta))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(
        self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("EquivGCNActor has no output distribution.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("EquivGCNActor has no output distribution.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("EquivGCNActor has no output distribution.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.distribution is None:
            raise RuntimeError("EquivGCNActor has no output distribution.")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("EquivGCNActor has no output distribution.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("EquivGCNActor has no output distribution.")
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(obs["policy"])
            self.obs_hist_normalizer.update(obs["history"])
