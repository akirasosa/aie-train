from typing import Dict

from ray.rllib.utils.typing import TensorType
from torch import nn as nn


class FlatModel(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        input_dim = 1260
        self.dense = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.Hardswish(),
            nn.Linear(input_dim // 2, dim),
            nn.LayerNorm(dim),
            nn.Hardswish(),
        )

    def forward(self, input_dict: Dict[str, TensorType]):
        x = self.dense(input_dict["obs_flat"])
        return x
