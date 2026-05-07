# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate as interpolate

from .utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def custom_random_uniform_terrain(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(
        (cfg.noise_range[0] + difficulty * (cfg.noise_range[1] - cfg.noise_range[0])) / cfg.vertical_scale
    )
    height_step = max(1, int(cfg.noise_step / cfg.vertical_scale))

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)
