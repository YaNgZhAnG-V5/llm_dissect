"""
This module implements the `ConcreteMask` class for differentiable masking using
the Concrete (Gumbel-Softmax) distribution.

Inspired by: Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class L0Mask(nn.Module):
    """
    A mask sampling module that uses the Concrete distribution to generate continuous
    mask values, enabling gradient-based learning. The mask is "soft" during training
    and can be hardened to binary during inference.

    Attributes:
    -----------
    - temperature: Controls the sharpness of mask values.
    - droprate_init: Initial dropout rate to configure mask logits.
    - z_loga: Learnable mask logits.
    - limit_a, limit_b: Bounds for mask values.
    - epsilon: Small constant to prevent numerical issues.
    """

    def __init__(
        self, shape, temperature=2.0 / 3.0, droprate_init=0.5, limit_a=-0.1, limit_b=1.1, epsilon=1e-6, device="cuda"
    ):
        """
        Initializes the mask with a given shape, dropout rate, and mask value limits.

        Parameters:
        -----------
        - shape: Shape of the mask.
        - temperature: Controls the sharpness of mask values.
        - droprate_init: Initial dropout rate to configure mask logits.
        - limit_a, limit_b: Bounds for mask values.
        - epsilon: Small constant to prevent numerical issues.
        - device: Device to place the mask on.
        """
        super().__init__()
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.shape = shape
        self.device = device
        self.z_loga = self.initialize_mask()
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

    def initialize_mask(self):
        """
        Sets up the mask logits (`z_loga`) as learnable parameters.

        Returns:
        --------
        - z_loga: Learnable mask logits.
        """
        mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        z_loga = nn.Parameter(torch.ones(*self.shape, device=self.device) * mean)
        return z_loga

    # def cdf_qz(self, z_loga):
    #     """
    #     Computes the CDF of the Concrete distribution for the mask logits.

    #     Parameters:
    #     -----------
    #     - z_loga: Mask logits.

    #     Returns:
    #     --------
    #     - CDF values for the mask logits.
    #     """
    #     # Normalize 0 to the range defined by limit_a and limit_b
    #     xn = (0 - self.limit_a) / (self.limit_b - self.limit_a)
    #     logits = math.log(xn) - math.log(1 - xn)
    #     return torch.sigmoid(logits * self.temperature - z_loga).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, eps):
        """
        Samples soft mask values using the inverse CDF and random noise.

        Parameters:
        -----------
        - eps: Random noise.

        Returns:
        --------
        - Soft mask values.
        """
        # Gumbel-Softmax Transformation
        y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + self.z_loga) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def forward(self):
        """
        During training, samples from the Concrete distribution. During inference,
        outputs a deterministic mask based on learned logits.

        Returns:
        --------
        - Mask values.
        """
        if self.training:
            eps = torch.rand(self.z_loga.shape, device=self.device)
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = torch.sigmoid(self.z_loga)
            return F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    # def forward(self):
    #     if self.training:
    #         eps = torch.rand(self.z_loga.shape, device=self.device)
    #         z = self.quantile_concrete(eps)
    #         return F.hardtanh(z, min_val=0, max_val=1)
    #     else:
    #         z = torch.sigmoid(self.z_loga / self.temperature * 0.8)
    #         return z

    def l0_norm(self):
        # return 1 - self.cdf_qz(self.z_loga)
        return torch.sigmoid(self.z_loga - self.temperature * (math.log(-self.limit_a) - math.log(self.limit_b)))
