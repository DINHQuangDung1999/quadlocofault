# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

#########################
# Model configurations #
#########################


@configclass
class RslRlFTNetActorCfg:
    """Configuration for the FTNetActor model."""

    class_name: str = "FTNetActor"
    """The model class name. Defaults to MLPModel."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP Actor network."""
    
    priv_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP privilege encoder network."""
    
    hist_encoder_output_channels: int | tuple[int] | list[int] = MISSING
    """The kernel size for the CNN."""

    hist_encoder_kernel_sizes: int | tuple[int] | list[int] = MISSING
    """The kernel size for the CNN."""

    hist_encoder_strides: int | tuple[int] | list[int] = 1
    """The stride for the CNN. Defaults to 1."""

    hist_encoder_dilations: int | tuple[int] | list[int] = 1
    """The dilation for the CNN. Defaults to 1."""

    hist_encoder_padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
    """The padding for the CNN. Defaults to none."""

    hist_encoder_norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
    """The normalization for the CNN. Defaults to none."""

    # hist_encoder_activation: str = MISSING
    # """The activation function for the CNN."""

    hist_encoder_max_pool: bool | tuple[bool] | list[bool] = False
    """Whether to use max pooling for the CNN. Defaults to False."""

    hist_encoder_global_pool: Literal["none", "max", "avg"] = "none"
    """The global pooling for the CNN. Defaults to none."""

    hist_encoder_flatten: bool = True
    """Whether to flatten the output of the CNN. Defaults to True."""
    
    encoder_out_dim: int = MISSING
    """The dimension of the latent space."""
    
    activation: str = MISSING
    """The activation function for the MLP Actor network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution of Actor. Defaults to None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""


        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""



@configclass
class RslRlDreamFLEXActorCfg:
    """Configuration for the DreamFLEXActor model."""

    class_name: str = "DreamFLEXActor"
    """The model class name. Defaults to MLPModel."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP Actor network."""

    hist_encoder_hidden_dims: int | tuple[int] | list[int] = MISSING
    """The hidden dimensions of the MLP history encoder network."""
    
    latent_decoder_hidden_dims: int | tuple[int] | list[int] = MISSING
    """The hidden dimensions of the MLP next state regressor network."""
    
    fault_decoder_hidden_dims: int | tuple[int] | list[int] = MISSING
    """The hidden dimensions of the MLP fault modulation network."""

    latent_dim: int = MISSING
    """The dimension of the latent space."""

    activation: str = MISSING
    """The activation function for the MLP Actor network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution of Actor. Defaults to None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""


        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

@configclass
class RslRlPINNActorCfg:
    """Configuration for the PINNActor model."""

    class_name: str = "PINNActor"
    """The model class name. Defaults to MLPModel."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP Actor network."""

    hist_encoder_hidden_dims: int | tuple[int] | list[int] = MISSING
    """The hidden dimensions of the MLP history encoder network."""
    
    # latent_decoder_hidden_dims: int | tuple[int] | list[int] = MISSING
    # """The hidden dimensions of the MLP next state regressor network."""
    
    # fault_decoder_hidden_dims: int | tuple[int] | list[int] = MISSING
    # """The hidden dimensions of the MLP fault modulation network."""

    latent_dim: int = MISSING
    """The dimension of the latent space."""

    activation: str = MISSING
    """The activation function for the MLP Actor network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution of Actor. Defaults to None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""


        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

