from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        # self.conv2 = nn.Conv1d(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     padding=0,
        #     dilation=dilation,
        # )
        # self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.norm1 = nn.BatchNorm1d(out_channels)
        # self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual = self.skip(x)
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv1(x)
        return F.elu(x)
        # x = self.norm1(self.conv1(x))
        # x = F.elu(self.norm1(self.conv1(x)))
        # x = F.pad(x, (self.left_padding, 0))
        # x = self.norm2(self.conv2(x))
        # return F.elu(x + residual)

