from typing import List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, sizes: List[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, flatten_start_dim: Optional[int] = None) -> torch.Tensor:
        if flatten_start_dim is not None:
            x = torch.flatten(x, start_dim=flatten_start_dim)
        return self.layers(x)
