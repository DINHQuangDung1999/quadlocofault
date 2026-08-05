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
        encoder_type: str = 'mlp'
    ) -> None:
        super().__init__()
        tcn_out_dim = 32
        self.node_dim = node_dim
        self.node_base_dim = node_base_dim
        self.num_nodes = num_nodes
        self.num_joints = num_nodes - 1
        self.projection_dim = projection_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.tcn_out_dim = tcn_out_dim
        self.gcn_out_dim = gcn_out_dim
        self.encoder_type = encoder_type
        adj = self._build_adj(num_nodes, edges)
        self.register_buffer("adj_norm", self._normalize_adj(adj))
        self.register_buffer("joint_permutation", torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=torch.long))
        # ``joint_permutation`` converts Isaac Lab joint order into GCN node
        # order (grouped by leg). Predictions returned to PPO must use the
        # original Isaac Lab order so they align with ``faulty_joint_idx``.
        self.register_buffer("inverse_joint_permutation", torch.argsort(self.joint_permutation))

        # obs_dim = self.node_base_dim + self.num_joints * self.node_dim
        # dim_node_joint_feats = self.num_joints * self.node_dim # 36
        if self.encoder_type == 'tcn':
            self.temporal_conv_joint = nn.Sequential(
                TemporalConvBlock(self.node_dim, tcn_hidden_dim, kernel_size=3, dilation=1),
                TemporalConvBlock(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, dilation=2),
                TemporalConvBlock(tcn_hidden_dim, tcn_out_dim, kernel_size=3, dilation=4),
            )

            self.temporal_conv_base = nn.Sequential(
                TemporalConvBlock(self.node_base_dim, tcn_hidden_dim, kernel_size=3, dilation=1),
                TemporalConvBlock(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, dilation=2),
                TemporalConvBlock(tcn_hidden_dim, tcn_out_dim, kernel_size=3, dilation=4),
            )
        elif self.encoder_type == 'mlp':
            self.mlp_joint = nn.Sequential(
                nn.LazyLinear(128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 32)
            )

            self.mlp_base = nn.Sequential(
                nn.LazyLinear(128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 32)
            )
        # self.node_base_projection = nn.Linear(self.node_base_dim, self.projection_dim)
        # self.node_joint_projection = nn.Linear(self.node_dim, self.projection_dim)

        self.gcn1 = GCNLayer(32, gcn_hidden_dim)
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
        #['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 
        # 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 
        # 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 
        # 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

        x_base = x[:, :, 36:]
        pos, vel, a_prev = x[:, :, :12], x[:, :, 12:24], x[:, :, 24:36]
        x_joints = torch.stack(
            [
                pos[:, :, self.joint_permutation],
                vel[:, :, self.joint_permutation],
                a_prev[:, :, self.joint_permutation],
            ],
            dim=-1,
        )
        B, H, D = x.shape
        if self.encoder_type == 'tcn':
            x_joints = x_joints.permute(0,2,3,1).flatten(0,1)
            x_joints = self.temporal_conv_joint(x_joints)
            x_joints = x_joints.view(B, self.num_joints, self.tcn_out_dim, H).mean(dim = -1)

            x_base = x_base.permute(0,2,1)
            x_base = self.temporal_conv_base(x_base).mean(dim = -1).unsqueeze(1)
        elif self.encoder_type == 'mlp':
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
        tcn_out_dim: int = 4,
        gcn_out_dim: int = 16,
        latent_dim: int = 16,
        setup=1,
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
        self.obs_critic_dim = obs['critic'].shape[1]
        self.latent_dim = latent_dim
        self.action_dim = output_dim
        self.setup = setup 

        self.projection_dim = projection_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.tcn_out_dim = tcn_out_dim
        self.gcn_out_dim = gcn_out_dim

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.obs_hist_normalizer = EmpiricalNormalization((self.obs_hist_length,self.obs_dim))
            self.obs_critic_normalizer = EmpiricalNormalization(self.obs_critic_dim)
            self.obs_scandots_normalizer = EmpiricalNormalization(187)
            self.obs_priv_phys_normalizer = EmpiricalNormalization(32)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.obs_hist_normalizer = torch.nn.Identity()
            self.obs_critic_normalizer = torch.nn.Identity()
            self.obs_scandots_normalizer = torch.nn.Identity()
            self.obs_priv_phys_normalizer = torch.nn.Identity()
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

        if setup == 1:
            # self.scandots_code_dim = 16
            # self.scandots_encoder = nn.Sequential(nn.Linear(187, 128),
            #                                     nn.ELU(),
            #                                     nn.Linear(128,64),
            #                                     nn.ELU(),
            #                                     nn.Linear(64,16)
            #                                     )
            # self.history_to_scandots_encoder = nn.Sequential(
            #     TemporalConvBlock(self.obs_dim, 32, kernel_size=3, dilation=1),
            #     TemporalConvBlock(32, 32, kernel_size=3, dilation=2),
            #     TemporalConvBlock(32, 32, kernel_size=3, dilation=4),
            # )
            # self.history_to_scandots_encoder_final_mlp = nn.LazyLinear(self.scandots_code_dim)
            # self.history_to_scandots_mlp = nn.Sequential(nn.Linear(32 * self.obs_hist_length, 128),
            #                                       nn.ELU(),
            #                                       nn.Linear(128, self.scandots_code_dim),
            #                                     #   nn.ELU(),
            #                                     #   nn.Linear(128, self.scandots_code_dim))
            # )
            self.fault_predictor = nn.Linear(gcn_out_dim, 1)
            self.motors_strength_predictor = nn.Linear(gcn_out_dim * 13, self.action_dim)
            self.fault_affine_gate_regressor = nn.Sequential(
                nn.Linear(self.action_dim + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, gcn_out_dim * 2),
            )
            # actor_dim = self.obs_dim + 12 + self.scandots_code_dim + gcn_out_dim
            # actor_dim = self.obs_dim + self.action_dim + 1 + self.action_dim + gcn_out_dim
            actor_dim = self.obs_dim + self.action_dim + gcn_out_dim 
        if setup == 2:
            self.latent_head = nn.Linear(gcn_hidden_dim * self.obs_hist_length, latent_dim)
            self.scandots_encoder = nn.Sequential(nn.Linear(187, 128),
                                                  nn.ELU(),
                                                  nn.Linear(128,64),
                                                  nn.ELU(),
                                                  nn.Linear(64,16))
            self.history_to_scandots_encoder = nn.Sequential(nn.Linear(self.obs_hist_length * self.obs_dim, 128),
                                                             nn.ELU(),
                                                             nn.Linear(128, 64),
                                                             nn.ELU(),
                                                             nn.Linear(64,16))
            self.priv_phys_encoder = nn.Sequential(nn.Linear(32, 64),
                                                   nn.ELU(),
                                                   nn.Linear(64, 64),
                                                   nn.ELU(),
                                                   nn.Linear(64, latent_dim))
            actor_dim = self.obs_dim + 16 + 16
            # self.mean_latent_head = nn.Linear(latent_dim, latent_dim)
            # self.logvar_latent_head = nn.Linear(latent_dim, latent_dim)
        
            # self.vae_decoder = nn.Sequential(
            #     nn.Linear(self.latent_dim, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 187),
            # )
        # self.modulator = nn.Sequential(
        #     nn.Linear(self.latent_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.action_dim * 2),
        # )
        self.actor_mlp = MLP(actor_dim, 
                             mlp_output_dim, 
                             actor_hidden_dims, 
                             activation)
        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_mlp)

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        if self.setup == 1:
            obs_policy = self.obs_normalizer(obs['policy'])
            obs_hist = self.obs_hist_normalizer(obs['history'])
            obs_critic = self.obs_hist_normalizer(obs['critic'])

            # # TCN encodes the terrain awareness via VAE style
            # scandots_code = self.history_to_scandots_encoder(obs_hist.flatten(1,2))
            # breakpoint()
            # scandots_code = self.history_to_scandots_encoder(obs_hist.permute(0,2,1)).flatten(1,2)
            # scandots_code = self.history_to_scandots_mlp(scandots_code)
            # scandots_target = self.scandots_encoder(obs_critic[:,48:48+187])
            # scandots_error = scandots_code - scandots_target.detach()
            # mean_scandots_code, var_scandots_co   de = scandots_code[:,:scandots_code.shape[-1]//2], scandots_code[:,scandots_code.shape[-1]//2:]
            # scandots_code = self.reparameterise(mean_scandots_code, var_scandots_code)
            # scandots_pred = self.history_to_scandots_decoder(scandots_code)
            # GCN encodes the dynamics and is gated by the fault prediction
            # breakpoint()
            gcn_code = self.gcn_encoder(obs_hist)
            # breakpoint()
            # fault_logits = self.fault_predictor(gcn_code.flatten(1,2))
            fault_logits_gcn_order = self.fault_predictor(gcn_code[:, :12, :]).squeeze(-1)
            fault_logits = fault_logits_gcn_order[:, self.gcn_encoder.inverse_joint_permutation]
            # motors_strength = self.motors_strength_predictor(gcn_code.flatten(1,2))
            # gamma = self.fault_affine_gate_regressor(fault_logits)
            # gamma1, gamma2 = gamma[:,:gcn_code.shape[-1]], gamma[:,gcn_code.shape[-1]:]
            # gcn_code = gamma1 + gamma2 * gcn_code.mean(1)
            
            # breakpoint()
            # actor_input = torch.cat([obs_policy, torch.sigmoid(fault_logits), gcn_code, scandots_code], dim = -1)
            # actor_input = torch.cat([obs_policy, gcn_code, motors_strength], dim = -1)
            actor_input = torch.cat([obs_policy, torch.sigmoid(fault_logits), gcn_code.mean(1)], dim = -1)
            # if self.training:
            #     actor_input = torch.cat([obs_policy, torch.sigmoid(fault_logits), gcn_code.mean(1), scandots_target], dim = -1)
            # else:
            #     actor_input = torch.cat([obs_policy, torch.sigmoid(fault_logits), gcn_code.mean(1), scandots_code], dim = -1)
            mlp_output = self.actor_mlp(actor_input)
            if self.distribution is not None:
                if stochastic_output:
                    self.distribution.update(mlp_output)
                    action = self.distribution.sample()
                else:
                    action = self.distribution.deterministic_output(mlp_output)
            # return action, (pred_vel, fault_logits, code_latent, mean_latent, logvar_latent, code_phys, code_terrain, pred_height_map)
            # return action, (self.setup, scandots_pred, scandots_code, mean_scandots_code, var_scandots_code, \
            #                 fault_logits, gcn_code)
            return action, (self.setup, fault_logits, None)
        
        elif self.setup == 2:
            obs_policy = self.obs_normalizer(obs['policy'])
            obs_hist = self.obs_hist_normalizer(obs['history'])
            obs_scandots = self.obs_scandots_normalizer(obs['critic'][:,49:49+187])
            obs_priv_phys = self.obs_priv_phys_normalizer(obs['critic'][:,-32:])
            priv_scandots_z = self.scandots_encoder(obs_scandots)
            hist_to_scandots_z = self.history_to_scandots_encoder(obs_hist.reshape(obs_hist.shape[0], -1))
            priv_phys_z = self.priv_phys_encoder(obs_priv_phys)
            gcn_z = self.latent_head(self.gcn_encoder(obs_hist))
            # breakpoint()
            actor_input = torch.cat([obs_policy, hist_to_scandots_z, gcn_z], dim = -1)
            mlp_output = self.actor_mlp(actor_input)
            if self.distribution is not None:
                if stochastic_output:
                    self.distribution.update(mlp_output)
                    action = self.distribution.sample()
                else:
                    action = self.distribution.deterministic_output(mlp_output)
            # return action, (pred_vel, fault_logits, code_latent, mean_latent, logvar_latent, code_phys, code_terrain, pred_height_map)
            return action, (self.setup, priv_phys_z, priv_scandots_z, hist_to_scandots_z, gcn_z)
    
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
            self.obs_critic_normalizer.update(obs['critic'])
            self.obs_scandots_normalizer.update(obs['critic'][:,49:49+187])
            self.obs_priv_phys_normalizer.update(obs['critic'][:,-32:])



class _TorchGCNActor(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: GCNActor) -> None:
        """Create a TorchScript-friendly copy of a CNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obs_hist_normalizer = copy.deepcopy(model.obs_hist_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.actor_mlp = copy.deepcopy(model.actor_mlp)
        self.hist_encoder_mlp = copy.deepcopy(model.hist_encoder_mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        obs = self.obs_normalizer(obs)
        obs_hist = self.obs_hist_normalizer(obs_hist)
        hist_latent = self.hist_encoder_cnn(obs_hist)
        actor_input = torch.cat([hist_latent, obs], dim = -1)
        out = self.actor_mlp(actor_input)
        return self.deterministic_output(out)

    
    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxGCNModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: GCNActor, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_1d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_1d = model.obs_groups_1d
        self.obs_dims_1d = model.obs_dims_1d
        self.obs_channels_1d = model.obs_channels_1d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs: torch.Tensor, *obs_hist: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_hist[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_1d)):
            h, w = self.obs_dims_1d[i]
            c = self.obs_channels_1d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_1d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
