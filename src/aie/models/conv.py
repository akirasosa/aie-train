from typing import Dict

import torch
from ray.rllib.utils.typing import TensorType
from torch import nn as nn

from aie.models.const import KEYS_3D, KEYS_2D, KEYS_1D
from libs.torch import init_weights


class ConvModel(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.conv_2d = nn.Sequential(
            nn.Conv2d(9, dim, kernel_size=7),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Conv2d(dim, dim, kernel_size=5),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Flatten(),
        )
        self.conv_1d = nn.Sequential(
            nn.Conv1d(10, dim, kernel_size=7),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Conv1d(dim, dim, kernel_size=5),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Flatten(),
        )
        self.fc_others = nn.Sequential(
            nn.Linear(11, dim),
            nn.LayerNorm(dim),
            nn.Hardswish(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Hardswish(),
            nn.Flatten(),
        )
        self.fc_all = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.Hardswish(),
        )
        self.apply(init_weights)

    def forward(self, input_dict: Dict[str, TensorType]):
        self.conv_2d.train(mode=input_dict.get("is_training", False))
        self.conv_1d.train(mode=input_dict.get("is_training", False))
        self.fc_others.train(mode=input_dict.get("is_training", False))
        self.fc_all.train(mode=input_dict.get("is_training", False))

        x_0 = torch.cat([input_dict['obs'][k] for k in KEYS_3D], dim=1)
        x_0 = self.conv_2d(x_0)

        x_1 = torch.stack([input_dict['obs'][k] for k in KEYS_2D], dim=1)
        x_1 = self.conv_1d(x_1)

        x_2 = torch.stack([input_dict['obs'][k] for k in KEYS_1D], dim=-1)
        x_2 = torch.cat((x_2, input_dict['obs']['time']), dim=-1)
        x_2 = self.fc_others(x_2)

        x = torch.cat((x_0, x_1, x_2), dim=-1)
        x = self.fc_all(x)

        return x
