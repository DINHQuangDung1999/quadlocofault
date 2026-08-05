# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab_quadlocofault.terrains import CustomHfRandomUniformTerrainCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Approximate DreamWaQ/legged_gym terrain mix:
        # [smooth slope, rough slope, stairs up, stairs down, discrete obstacles].
        # Isaac Lab does not expose the rough-slope composite primitive directly,
        # so we approximate it with a separate random-rough terrain type.
        # "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.1, 
        #                                      num_waves=4, 
        #                                      amplitude_range=(0.0,0.25)
        # ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15
        ),        
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=3.0,
        ),
        "random_rough": CustomHfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.05, 0.05),
            noise_step=0.01,
            downsampled_scale=0.15,
            border_width=0.25,
        ),        
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, 
            slope_range=(0.0, 0.4), 
            platform_width=3.0, 
            border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.1,
        #     obstacle_height_mode="fixed",
        #     obstacle_width_range=(1.0, 2.0),
        #     obstacle_height_range=(0.05, 0.25),
        #     num_obstacles=20,
        #     platform_width=3.0,
        #     border_width=0.25,
        # ),
    },
)
"""Rough terrains configuration."""
