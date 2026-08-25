# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from modules import CNN1D, GCNLayer, TemporalConvBlock
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable
import torch.nn.functional as F


class GCNTemporalEncoder(nn.Module):
    """Encode a ``[q, qdot, previous_action, omega, gravity, command]`` history."""

    ISAAC_JOINT_ORDER = (
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    )
    LEG_GROUPED_JOINT_IDS = (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)

    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        node_base_dim: int,
        projection_dim: int,
        gcn_hidden_dim: int,
        tcn_hidden_dim: int,
        tcn_out_dim: int,
        gcn_out_dim: int,
        edges: list[tuple[int, int]],
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.node_base_dim = node_base_dim
        self.num_nodes = num_nodes
        self.num_joints = num_nodes - 1
        if self.num_joints != 12 or node_dim != 3 or node_base_dim != 9:
            raise ValueError(
                "GCNTemporalEncoder requires 12 joints with three features per joint "
                "and nine base/command features."
            )
        self.projection_dim = projection_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.tcn_out_dim = tcn_out_dim
        self.gcn_out_dim = gcn_out_dim
        adj = self._build_adj(num_nodes, edges)
        self.register_buffer("adj_norm", self._normalize_adj(adj))
        self.register_buffer(
            "joint_permutation",
            torch.tensor(self.LEG_GROUPED_JOINT_IDS, dtype=torch.long),
        )
        # ``joint_permutation`` converts Isaac Lab joint order into GCN node
        # order (grouped by leg). Predictions returned to PPO must use the
        # original Isaac Lab order so they align with ``faulty_joint_idx``.
        self.register_buffer("inverse_joint_permutation", torch.argsort(self.joint_permutation))

        self.mlp_joint = nn.Sequential(
            nn.LazyLinear(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, tcn_out_dim)
        )

        self.mlp_base = nn.Sequential(
            nn.LazyLinear(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, tcn_out_dim)
        )

        self.gcn1 = GCNLayer(tcn_out_dim, gcn_hidden_dim)
        self.gcn2 = GCNLayer(gcn_hidden_dim, gcn_out_dim)
        # self.gcn3 = GCNLayer(gcn_hidden_dim, gcn_out_dim)

    @staticmethod
    def _build_adj(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
        adj = torch.zeros(num_nodes, num_nodes)
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = adj + torch.eye(num_nodes)
        return adj

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        d_inv_sqrt = torch.diag(degree_inv_sqrt)
        return d_inv_sqrt @ adj @ d_inv_sqrt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_length, feature_dim = x.shape
        expected_dim = self.node_base_dim + self.num_joints * self.node_dim
        if feature_dim != expected_dim:
            raise ValueError(f"Expected input feature_dim={expected_dim}, got {feature_dim}")

        position = x[:, :, :self.num_joints]
        velocity = x[:, :, self.num_joints:2 * self.num_joints]
        previous_action = x[:, :, 2 * self.num_joints:3 * self.num_joints]
        x_base = x[:, :, 3 * self.num_joints:3 * self.num_joints + self.node_base_dim]
        x_joints = torch.stack(
            [
                position[:, :, self.joint_permutation],
                velocity[:, :, self.joint_permutation],
                previous_action[:, :, self.joint_permutation],
            ],
            dim=-1,
        )
        B, H, D = x.shape

        x_joints = x_joints.permute(0,2,1,3).flatten(2,3)
        x_joints = self.mlp_joint(x_joints)

        x_base = self.mlp_base(x_base.flatten(1,2)).unsqueeze(1)
        x_graphs = torch.concat([x_joints, x_base], dim = 1)

        # breakpoint()
        hid = F.elu(self.gcn1(x_graphs, self.adj_norm))
        hid = F.elu(self.gcn2(hid, self.adj_norm))
        # hid = F.elu(self.gcn3(hid, self.adj_norm))
        # breakpoint()
        return hid

    
class GCNActor(nn.Module):

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    expected_joint_order = GCNTemporalEncoder.ISAAC_JOINT_ORDER

    def __init__(
        self,
        obs: TensorDict,
        output_dim: int = 12,
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        projection_dim: int = 8,
        gcn_hidden_dim: int = 16,
        tcn_hidden_dim: int = 8,
        tcn_out_dim: int = 32,
        gcn_out_dim: int = 16,
        film_scale: float = 0.5,
        **_: object,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            output_dim: Dimension of the output.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            actor_hidden_dims: Hidden dimensions of the MLP.
        """
        super().__init__()
        # Resolve observation groups and dimensions
        self.obs_hist_length, self.obs_dim = obs['history'].shape[1:]
        self.action_dim = output_dim

        self.projection_dim = projection_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.tcn_out_dim = tcn_out_dim
        self.gcn_out_dim = gcn_out_dim
        self.film_scale = film_scale

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.obs_hist_normalizer = EmpiricalNormalization((self.obs_hist_length,self.obs_dim))
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.obs_hist_normalizer = torch.nn.Identity()
        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
            
        # Encoders
        edges = [
            (0, 1), (1, 2),
            (3, 4), (4, 5),
            (6, 7), (7, 8),
            (9, 10), (10, 11),
            (12, 0),
            (12, 3),
            (12, 6),
            (12, 9),
        ]

        self.gcn_encoder = GCNTemporalEncoder(
            num_nodes=self.action_dim + 1,
            node_dim=3,
            node_base_dim=9,
            projection_dim=projection_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            tcn_hidden_dim=tcn_hidden_dim,
            gcn_out_dim=gcn_out_dim,
            tcn_out_dim=tcn_out_dim,
            edges=edges,
        )

        # The same classifier is applied independently to every joint
        # embedding, so joint permutations induce the same permutation of the
        # fault logits rather than selecting unrelated output weights.
        self.fault_predictor = nn.Sequential(
            nn.Linear(gcn_out_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        # Fault probabilities pool abnormal joint features into a context that
        # generates bounded residual FiLM parameters.
        self.fault_modulation_head = nn.Sequential(
            nn.Linear(gcn_out_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2 * gcn_out_dim),
        )
        final_modulation_layer = self.fault_modulation_head[-1]
        nn.init.zeros_(final_modulation_layer.weight)
        nn.init.zeros_(final_modulation_layer.bias)

        actor_dim = self.obs_dim + self.action_dim + gcn_out_dim
        self.actor_mlp = MLP(actor_dim, 
                             mlp_output_dim, 
                             actor_hidden_dims, 
                             activation)
        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        """Compute fault-gated graph features and the policy action."""
        obs_policy = self.obs_normalizer(obs['policy'])
        obs_hist = self.obs_hist_normalizer(obs['history'])

        # The graph encoder produces per-joint dynamics embeddings plus a base
        # embedding. The shared head localizes faults before global pooling.
        gcn_code = self.gcn_encoder(obs_hist)
        joint_code = gcn_code[:, :self.action_dim, :]
        fault_logits_gcn_order = self.fault_predictor(joint_code).squeeze(-1)
        fault_logits = fault_logits_gcn_order[
            :, self.gcn_encoder.inverse_joint_permutation
        ]
        fault_probability = torch.sigmoid(fault_logits).detach()

        fault_weight_sum = fault_probability.sum(dim=1, keepdim=True)
        fault_context = (
            fault_probability.unsqueeze(-1) * joint_code
        ).sum(dim=1) / fault_weight_sum.clamp_min(1.0e-6)
        raw_gamma, raw_beta = self.fault_modulation_head(fault_context).chunk(2, dim=-1)
        gamma = self.film_scale * torch.tanh(raw_gamma)
        beta = self.film_scale * torch.tanh(raw_beta)

        dynamics_latent = gcn_code.mean(dim=1)
        fault_gate = fault_probability.amax(dim=1, keepdim=True)
        fused_latent = (
            (1.0 + fault_gate * gamma) * dynamics_latent
            + fault_gate * beta
        )
        actor_input = torch.cat(
            [obs_policy, fault_probability, fused_latent], dim=-1
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
        return action, (fault_logits, (gamma, beta))
    
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach therecurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchGCNActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxGCNModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Update the normalizer parameters
            self.obs_normalizer.update(obs['policy'])  # type: ignore
            self.obs_hist_normalizer.update(obs['history'])



class _TorchGCNActor(nn.Module):
    """TorchScript wrapper for the fault-gated GCN inference path."""

    def __init__(self, model: GCNActor) -> None:
        """Copy the graph encoder, shared fault head, modulation, and actor."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_hist_normalizer = copy.deepcopy(model.obs_hist_normalizer)
        self.gcn_encoder = copy.deepcopy(model.gcn_encoder)
        self.fault_predictor = copy.deepcopy(model.fault_predictor)
        self.fault_modulation_head = copy.deepcopy(model.fault_modulation_head)
        self.actor_mlp = copy.deepcopy(model.actor_mlp)
        self.film_scale = model.film_scale
        self.action_dim = model.action_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        obs = self.obs_normalizer(obs)
        obs_hist = self.obs_hist_normalizer(obs_hist)

        gcn_code = self.gcn_encoder(obs_hist)
        joint_code = gcn_code[:, : self.action_dim, :]
        fault_logits_gcn_order = self.fault_predictor(joint_code).squeeze(-1)
        fault_probability_gcn_order = torch.sigmoid(fault_logits_gcn_order)
        fault_probability = fault_probability_gcn_order[
            :, self.gcn_encoder.inverse_joint_permutation
        ]

        fault_weight_sum = fault_probability_gcn_order.sum(dim=1, keepdim=True)
        fault_context = (
            fault_probability_gcn_order.unsqueeze(-1) * joint_code
        ).sum(dim=1) / fault_weight_sum.clamp_min(1.0e-6)
        raw_gamma, raw_beta = self.fault_modulation_head(fault_context).chunk(2, dim=-1)
        gamma = self.film_scale * torch.tanh(raw_gamma)
        beta = self.film_scale * torch.tanh(raw_beta)

        dynamics_latent = gcn_code.mean(dim=1)
        fault_gate = fault_probability.amax(dim=1, keepdim=True)
        fused_latent = (
            (1.0 + fault_gate * gamma) * dynamics_latent
            + fault_gate * beta
        )
        actor_input = torch.cat([obs, fault_probability, fused_latent], dim=-1)
        out = self.actor_mlp(actor_input)
        return self.deterministic_output(out)

    
    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxGCNModel(_TorchGCNActor):
    """ONNX wrapper for the fault-gated GCN inference path."""

    def __init__(self, model: GCNActor, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    @property
    def input_names(self) -> list[str]:
        return ["obs", "obs_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
