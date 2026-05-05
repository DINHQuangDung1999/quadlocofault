# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# from ...rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg
from isaaclab_quadlocofault_rl.rsl_rl.rl_cfg import RslRlFTNetActorCfg, RslRlDreamFLEXActorCfg, RslRlPINNActorCfg

@configclass
class UnitreeGo2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_go2_base"
    empirical_normalization = False

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class UnitreeGo2RoughPPORunnerCfg(UnitreeGo2PPORunnerCfg):
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0,
                                                                    std_type="log")
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = None
    )
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPO"
        self.experiment_name = "unitree_go2_rough_base"

@configclass
class UnitreeGo2RoughPPOFTNetRunnerCfg(UnitreeGo2PPORunnerCfg):
    actor = RslRlFTNetActorCfg(
        actor_hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = RslRlFTNetActorCfg.GaussianDistributionCfg(init_std=1.0,std_type="log"),
        priv_encoder_hidden_dims =[256, 128],
        hist_encoder_output_channels = [32,32,32], 
        hist_encoder_kernel_sizes=[9,5,5], 
        hist_encoder_strides=[2,1,1],        
        hist_encoder_dilations = [1,1,1],
        hist_encoder_padding = "none",
        hist_encoder_norm = "none",
        # hist_encoder_activation = "elu",
        hist_encoder_max_pool = False,
        hist_encoder_global_pool = "none",
        hist_encoder_flatten = True,
        encoder_out_dim = 8
                        )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = None
    )
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPOFTNet"
        self.experiment_name = "unitree_go2_rough_ftnet"

@configclass
class UnitreeGo2RoughPPOFLEXRunnerCfg(UnitreeGo2PPORunnerCfg):
    actor = RslRlDreamFLEXActorCfg(
        actor_hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = RslRlDreamFLEXActorCfg.GaussianDistributionCfg(init_std=1.0,
                                                                    std_type="log"),
        hist_encoder_hidden_dims=[512, 256, 128],
        latent_decoder_hidden_dims=[64, 128],
        fault_decoder_hidden_dims=[64, 64],
        latent_dim = 16
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = None
    )
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPODreamFLEX"
        self.experiment_name = "unitree_go2_rough_flex"
        
@configclass
class UnitreeGo2RoughPPOPINNRunnerCfg(UnitreeGo2PPORunnerCfg):
    actor = RslRlPINNActorCfg(
        actor_hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = RslRlPINNActorCfg.GaussianDistributionCfg(init_std=1.0,
                                                                    std_type="log"),
        hist_encoder_hidden_dims=[512, 256, 128],
        # latent_decoder_hidden_dims=[64, 128],
        # fault_decoder_hidden_dims=[64, 64],
        latent_dim = 32
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = None
    )
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPOPINN"
        self.experiment_name = "unitree_go2_rough_pinn"


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "unitree_go2_flat_base"

@configclass
class UnitreeGo2FlatPPOFTNetRunnerCfg(UnitreeGo2RoughPPOFTNetRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "unitree_go2_flat_ftnet"

@configclass
class UnitreeGo2FlatPPOFLEXRunnerCfg(UnitreeGo2RoughPPOFLEXRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "unitree_go2_flat_flex"


@configclass
class UnitreeGo2FlatPPOPINNRunnerCfg(UnitreeGo2RoughPPOPINNRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "unitree_go2_flat_pinn"

