# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from .hf_terrains import custom_random_uniform_terrain


@configclass
class CustomHfRandomUniformTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = custom_random_uniform_terrain

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""

    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""

    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """