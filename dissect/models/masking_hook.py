from typing import Any, Optional

import torch
import torch.nn as nn


class MaskingHook:

    def __init__(self, mask: torch.Tensor, prior: Optional[torch.Tensor] = None, mask_input: bool = False) -> None:
        self.mask = mask.to(torch.float32)
        self.prior = prior
        self.mask_input = mask_input

    def __call__(self, m: nn.Module, inputs: Any, outputs: Any) -> Any:
        if self.mask_input:
            if not isinstance(inputs, torch.Tensor):
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]
                else:
                    raise ValueError("Unsupported input type.")
            return m.forward(inputs * self.mask)
        if self.prior is None:
            return outputs * self.mask
        else:
            return outputs * self.mask + (1 - self.mask) * self.prior
