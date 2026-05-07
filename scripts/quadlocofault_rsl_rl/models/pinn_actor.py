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
from rsl_rl.utils import resolve_callable, unpad_trajectories, resolve_nn_activation

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import numpy as np 

class PINNActor(nn.Module):

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
        # latent_decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        # fault_decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
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
        self.latent_dim = latent_dim
        self.action_dim = output_dim

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.obs_hist_normalizer = EmpiricalNormalization(self.obs_dim)
            self.obs_critic_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.obs_hist_normalizer = torch.nn.Identity()
            self.obs_critic_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP
        # self.actor_mlp = MLP(self.obs_dim * 2 + 12, mlp_output_dim, actor_hidden_dims, activation)
        # self.actor_mlp = MLP(self.obs_dim + 24 + 12, mlp_output_dim, actor_hidden_dims, activation)
        self.actor_mlp = MLP(self.obs_dim, mlp_output_dim, actor_hidden_dims, activation)
        # Encoders
        self.hist_encoder_mlp = MLP(self.obs_dim * self.obs_hist_length, 
                                    latent_dim, 
                                    hist_encoder_hidden_dims, 
                                    activation)
        self.priv_encoder_mlp = MLP(obs['critic'].shape[1], 
                                    latent_dim, 
                                    hist_encoder_hidden_dims, 
                                    activation)
        # self.hist_encoder_mlp = nn.Sequential(
        #               nn.Linear(self.obs_dim, latent_dim),
        #               resolve_nn_activation(activation),
        #               nn.GRU(latent_dim, latent_dim, num_layers=1, batch_first=True),
        # )

        # self.func = LatentODEfunc(latent_dim, 20)
        # self.rec = RecognitionRNN(latent_dim, self.obs_dim, latent_dim, obs['history'].shape[0])
        # self.dec = Decoder(latent_dim, self.obs_dim, 20)
    
        # self.mean_vel_encoder_mlp = nn.Linear(latent_dim, 3)
        # self.logvar_vel_encoder_mlp = nn.Linear(latent_dim, 3)
        self.motor_strengths_encoder_mlp = nn.Linear(latent_dim, self.action_dim) 
        self.mean_latent_encoder_mlp = nn.Linear(latent_dim, latent_dim)
        self.logvar_latent_encoder_mlp = nn.Linear(latent_dim, latent_dim)
        self.phys_latent_encoder_mlp = nn.Linear(latent_dim, latent_dim)
        # self.fault_logit_encoder_mlp = nn.Linear(latent_dim, self.action_dim)
        # Decoders
        self.Minv_head = nn.Linear(latent_dim, self.action_dim*(self.action_dim + 1)//2)
        self.fn_head = nn.Linear(latent_dim, self.action_dim)  
        self.torque_head = nn.Linear(latent_dim + 2 * self.action_dim, self.action_dim) 
        # self.partial_next_obs_head = MLP(latent_dim, 
        #                             25, 
        #                             [256, 128], 
        #                             activation)
        # self.fault_head = nn.Linear(latent_dim, self.action_dim)
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
        obs_policy = self.obs_normalizer(obs['policy'])

  
        # Get MLP input latent
        latent_outputs = self.get_latent(obs)
        priv_latent, hist_latent, state_latent, _, _, phys_latent, pred_motors_strength = latent_outputs

        # im_next_partial_obs = self.partial_next_obs_head(state_latent)
        # im_next_action = im_next_partial_obs[:,:12]
        # obs_ = torch.concat([obs_policy, torch.zeros((obs_policy.shape[0], self.latent_dim), device = obs.device)], dim = -1)
        # im_action = self.actor_mlp(obs_)
        relu = resolve_nn_activation("relu")
        Lterms = self.Minv_head(phys_latent)
        L = torch.zeros((Lterms.shape[0], self.action_dim, self.action_dim), device=obs.device)
        idx = torch.tril_indices(self.action_dim, self.action_dim, offset=-1)
        L[:, idx[0], idx[1]] = Lterms[:, self.action_dim:]
        L += torch.diag_embed(nn.Sigmoid()(Lterms[:, :self.action_dim]))
        # L += torch.diag_embed(relu(Lterms[:, :self.action_dim]) + 0.1)
        Minv = L @ L.transpose(1,2)
        # breakpoint()

        # breakpoint()
        # fn and delta q
        fn = self.fn_head(phys_latent)
        # tau = self.torque_head(torch.cat([phys_latent, pred_motors_strength, im_action], dim = -1))
        obs_critic = self.obs_critic_normalizer(obs['critic'])
        # breakpoint()
        tau = obs_critic[:,-36:-24]
        # integrate
        # breakpoint()
        dt = 0.005
        q, qdot = obs['history'][:,-2,:self.action_dim], obs['history'][:,-2,self.action_dim:self.action_dim*2]
        # breakpoint()
        pred_qddot = Minv @ (tau + fn).unsqueeze(-1)
        pred_qdot = qdot + pred_qddot.squeeze(-1) * dt 
        pred_q    =  q + pred_qdot * dt

        pred_state = torch.cat([pred_q, pred_qdot], dim = -1)
        error = obs['policy'][:,:24] - pred_state
        # MLP forward pass
        # breakpoint()
        
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        # Action
        obs_policy = self.obs_normalizer(obs['policy'])
        actor_input = torch.cat([obs_policy, error, pred_motors_strength], dim = -1)

        mlp_output = self.actor_mlp(actor_input)
        if self.distribution is not None:
            try:
                if stochastic_output:
                    self.distribution.update(mlp_output)
                    action = self.distribution.sample()
                else:
                    action = self.distribution.deterministic_output(mlp_output)
            except:
                breakpoint()
        return action, (latent_outputs, error, tau)
    
    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        # Normalize observations
        obs_hist = self.obs_hist_normalizer(obs['history'].flatten(1,2))
        hist_latent = self.hist_encoder_mlp(obs_hist)
        # obs_hist = self.obs_hist_normalizer(obs['history'])
        # o_n, h_n = self.hist_encoder_mlp(obs_hist)
        # hist_latent = o_n.mean(1)
        priv_latent = self.get_priv_latent(obs)

        mean_latent = self.mean_latent_encoder_mlp(hist_latent)
        logvar_latent = self.logvar_latent_encoder_mlp(hist_latent)
        state_latent = self.reparameterise(mean_latent,logvar_latent)

        phys_latent = self.phys_latent_encoder_mlp(hist_latent)

        motors_strength = self.motor_strengths_encoder_mlp(hist_latent)
        motors_strength = 1 + nn.Tanh()(motors_strength)
        # fault_logit = self.fault_logit_encoder_mlp(distribution)
        # fault_label = torch.max(fault_logit, dim = -1)[1]
        # fault_binarylabel = torch.zeros_like(fault_logit).to(fault_logit.device)
        # fault_binarylabel[torch.arange(fault_binarylabel.shape[0]), fault_label] = 1.

        # gamma = self.fault_decoder_mlp(fault_logit)

        # decode = self.latent_decoder_mlp(code_latent)

        # code_latent = gamma[:,:self.latent_dim] * code_latent + gamma[:,self.latent_dim:]

        # code = torch.cat((code_vel, fault_logit, code_latent),dim=-1)
        return priv_latent, hist_latent, state_latent, mean_latent, logvar_latent, phys_latent, motors_strength

    def get_priv_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs_critic_hist = self.obs_critic_normalizer(obs['critic'])
        priv_latent = self.priv_encoder_mlp(obs_critic_hist)
        return priv_latent
    
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
        return _TorchPINNActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Update the normalizer parameters
            self.obs_normalizer.update(obs['policy'])  # type: ignore
            self.obs_hist_normalizer.update(obs['history'])



class _TorchPINNActor(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: PINNActor) -> None:
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

    def __init__(self, model: PINNActor, verbose: bool) -> None:
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



class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        # breakpoint()
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, batch, hidden):
        return torch.zeros(batch, hidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl