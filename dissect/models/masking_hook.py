from typing import Any, Optional

import torch
import torch.nn as nn


class MaskingHook:

    def __init__(self, mask: torch.Tensor, prior: Optional[torch.Tensor] = None, mask_input: bool = False) -> None:
        self.mask = mask
        self.prior = prior
        self.mask_input = mask_input

    def __call__(self, m: nn.Module, inputs: Any, outputs: Any) -> Any:
        if isinstance(outputs, torch.Tensor):
            device = outputs.device
            dtype = outputs.dtype
        elif isinstance(outputs, (tuple, list)):
            device = outputs[0].device
            dtype = outputs[0].dtype
        else:
            raise TypeError(f"Unsupported outputs type: {outputs.__class__.__name__}")
        self.mask = self.mask.to(dtype=dtype, device=device)
        if self.mask_input:
            if not isinstance(inputs, torch.Tensor):
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]
                else:
                    raise ValueError("Unsupported input type.")
            return m.forward(inputs * self.mask)
        if self.prior is None:
            if isinstance(outputs, torch.Tensor):
                return outputs * self.mask
            else:
                # The first element is usually the hidden_states
                return (outputs[0] * self.mask,) + outputs[1:]
        else:
            if isinstance(outputs, torch.Tensor):
                return outputs * self.mask + (1 - self.mask) * self.prior
            else:
                return (outputs[0] * self.mask + (1 - self.mask) * self.prior,) + outputs[1:]
