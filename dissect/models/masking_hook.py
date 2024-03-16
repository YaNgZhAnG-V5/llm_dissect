from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from ..utils import Device


class MaskingHook:

    def __init__(self, mask: torch.Tensor, prior: Optional[torch.Tensor] = None) -> None:
        self.mask = mask.to(torch.float32)
        self.prior = prior

    def __call__(self, m: nn.Module, inputs: Any, outputs: Any) -> Any:
        if self.prior is None:
            return outputs * self.mask
        else:
            return outputs * self.mask + (1 - self.mask) * self.prior


def register_masking_hooks(
    model: nn.Module,
    mask_path: str,
    device: Device,
    exclude_layers: Sequence[str] = (),
    prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, RemovableHandle]:
    mask_state_dict = torch.load(mask_path, map_location=device)
    handle_dict = dict()

    for layer_name, pruning_mask in mask_state_dict.items():
        if any(exclude_layer in layer_name for exclude_layer in exclude_layers):
            continue
        prior = None if prior_state_dict is None else prior_state_dict[layer_name]
        layer = model.get_submodule(layer_name)
        hook = MaskingHook(pruning_mask, prior=prior)
        handle = layer.register_forward_hook(hook)
        handle_dict.update({layer_name: handle})

    return handle_dict
