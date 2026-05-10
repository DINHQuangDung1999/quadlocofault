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
from modules import CNN1D
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class DreamFLEXActor(nn.Module):

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
        hist_encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        latent_decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        fault_decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        latent_dim: int = 16,
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
        # breakpoint()
        # Resolve observation groups and dimensions
        self.obs_hist_length, self.obs_dim = obs['history'].shape[1:]
        self.priv_obs_dim = obs['critic'].shape[1]
        self.latent_dim = latent_dim
        self.action_dim = output_dim
        self.encoder_out_dim = 3 + self.action_dim + self.latent_dim # 3 + 12 + 16,

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.critic_obs_normalizer = EmpiricalNormalization(self.priv_obs_dim)
            self.obs_hist_normalizer = EmpiricalNormalization([self.obs_hist_length, self.obs_dim])
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()
            self.obs_hist_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP
        # self.actor_mlp = MLP(self.obs_dim * 2, mlp_output_dim, actor_hidden_dims, activation)
        self.actor_mlp = MLP(self.obs_dim + self.encoder_out_dim, mlp_output_dim, actor_hidden_dims, activation)
        # Encoders
        self.hist_encoder_mlp = MLP(self.obs_dim * self.obs_hist_length, 
                                    hist_encoder_hidden_dims[-1], 
                                    hist_encoder_hidden_dims[:-1], 
                                    activation)
        code_dim = hist_encoder_hidden_dims[-1]
        self.mean_vel_encoder_mlp = nn.Linear(code_dim, 3)
        self.logvar_vel_encoder_mlp = nn.Linear(code_dim, 3)
        self.mean_latent_encoder_mlp = nn.Linear(code_dim, self.latent_dim)
        self.logvar_latent_encoder_mlp = nn.Linear(code_dim, self.latent_dim)
        self.fault_logit_encoder_mlp = nn.Linear(code_dim, self.action_dim)
        # Decoders
        self.latent_decoder_mlp = MLP(self.latent_dim, self.obs_dim, latent_decoder_hidden_dims, activation)
        self.fault_decoder_mlp = MLP(self.action_dim, self.latent_dim * 2, fault_decoder_hidden_dims, activation)
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
        stochastic_output: bool = False
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # breakpoint()
        # Get MLP input latent
        obs_policy = self.obs_normalizer(obs['policy'])
        latent_outputs = self.get_latent(obs)
        hist_latent = latent_outputs[0]
        # hist_latent = latent_outputs[2]
        actor_input = torch.cat([hist_latent, obs_policy], dim = -1)

        # MLP forward pass
        # breakpoint()
        mlp_output = self.actor_mlp(actor_input)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample(), latent_outputs
            return self.distribution.deterministic_output(mlp_output), latent_outputs
        return mlp_output, latent_outputs
    
    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        # Normalize observations
        obs_hist = self.obs_hist_normalizer(obs['history']).flatten(1,2)
        distribution = self.hist_encoder_mlp(obs_hist)

        mean_latent = self.mean_latent_encoder_mlp(distribution)
        logvar_latent = self.logvar_latent_encoder_mlp(distribution)

        mean_vel = self.mean_vel_encoder_mlp(distribution)
        # logvar_vel = self.logvar_vel_encoder_mlp(distribution)

        code_latent = self.reparameterise(mean_latent,logvar_latent)
        # code_vel = self.reparameterise(mean_vel,logvar_vel)
        code_vel = mean_vel
        logvar_vel = torch.zeros_like(mean_vel)

        
        fault_logit = self.fault_logit_encoder_mlp(distribution)
        # fault_label = torch.max(fault_logit, dim = -1)[1]
        # fault_binarylabel = torch.zeros_like(fault_logit).to(fault_logit.device)
        # fault_binarylabel[torch.arange(fault_binarylabel.shape[0]), fault_label] = 1.

        gamma = self.fault_decoder_mlp(fault_logit)

        decode = self.latent_decoder_mlp(code_latent)
        
        code_latent = gamma[:,:self.latent_dim] * code_latent + gamma[:,self.latent_dim:]

        code = torch.cat((code_vel, fault_logit, code_latent),dim=-1)

        # decode = self.latent_decoder_mlp(code_latent)

        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent, fault_logit


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
        return _TorchDreamFLEXActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Update the normalizer parameters
            self.obs_normalizer.update(obs['policy'])  # type: ignore
            self.critic_obs_normalizer.update(obs['critic'])
            self.obs_hist_normalizer.update(obs['history'])



class _TorchDreamFLEXActor(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: DreamFLEXActor) -> None:
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


class _OnnxMLPModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: DreamFLEXActor, verbose: bool) -> None:
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

