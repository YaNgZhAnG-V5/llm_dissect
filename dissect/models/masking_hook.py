from typing import Any, Optional

import torch
import torch.nn as nn


class MaskingHook:

    def __init__(self, mask: torch.Tensor, prior: Optional[torch.Tensor] = None) -> None:
        self.mask = mask.to(torch.float32)
        self.prior = prior

    def __call__(self, m: nn.Module, inputs: Any, outputs: Any) -> Any:
        if self.prior is None:
            return outputs * self.mask
        else:
            return outputs * self.mask + (1 - self.mask) * self.prior
