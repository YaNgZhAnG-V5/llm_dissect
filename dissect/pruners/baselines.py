from typing import List

import mmengine
import torch
import torch.nn as nn

from ..utils import Device, name_contains_keys
from .builder import TESTING_MANAGER

# Wanda does not need a pruner. It only needs a testing manager.


@TESTING_MANAGER.register_module()
class WandaTestingManager:
    """Manager for loading Wanda inputs, applying masks to weights, and restoring the original weights"""

    def __init__(self, inputs_path: str, device: Device) -> None:
        logger = mmengine.MMLogger.get_instance("dissect")
        self.all_input_norms = torch.load(inputs_path, map_location=device)
        logger.info(f"Loaded cached input norms from {inputs_path}")

    @torch.no_grad()
    def prepare_environment(
        self,
        model: nn.Module,
        sparsity: float,
        device: Device,
        exclude_layers: List[str] = (),
    ) -> None:
        """Prepare environment for testing model."""
        # in the script where this method is called, the model is already copied,
        # and the copied model is passed to the method
        pruned_state_dict = model.state_dict()

        for k, v in pruned_state_dict.items():
            if name_contains_keys(k, exclude_layers):
                continue
            if "bias" in k:
                continue
            # weight: (output dim, input dim)
            weight = v.abs()
            # make input_norm: (input dim)
            input_norm = self.all_input_norms[k.replace(".weight", "")]
            metric = weight * input_norm
            _, sorted_idx = torch.sort(metric, dim=1)
            pruned_idx = sorted_idx[:, : int(sorted_idx.shape[1] * sparsity)]
            v.scatter_(dim=1, index=pruned_idx, src=torch.zeros_like(pruned_idx, dtype=v.dtype))
            pruned_state_dict[k] = v
        model.load_state_dict(pruned_state_dict)

    @torch.no_grad()
    def clean_environment(self, model: nn.Module) -> None:
        """Clean environment after testing model."""
        pass


@TESTING_MANAGER.register_module()
class MagnitudeTestingManager:
    """Manager applying masks to weights, and restoring the original weights"""

    @torch.no_grad()
    def prepare_environment(
        self,
        model: nn.Module,
        sparsity: float,
        device: Device,
        exclude_layers: List[str] = (),
    ) -> None:
        """Prepare environment for testing model."""

        # in the script where this method is called, the model is already copied,
        # and the copied model is passed to the method
        # TODO there is a bug in this implementation, split weight is defined over all modules (include embedding)
        pruned_state_dict = model.state_dict()
        all_weights = []
        all_numels = []
        # pruned_state_dict is still original state dict at this moment
        for k, v in pruned_state_dict.items():
            all_weights.append(torch.flatten(v))
            all_numels.append(v.numel())

        all_weights = torch.concat(all_weights, 0)
        abs_all_weights = all_weights.abs()
        _, top_k_inds = torch.topk(abs_all_weights, int(all_weights.numel() * (1 - sparsity)))

        # pruned_mask: 1 for setting weight to 0, 0 for keep original weight
        pruned_mask = torch.ones_like(all_weights, dtype=torch.bool)
        pruned_mask[top_k_inds] = 0
        all_weights[pruned_mask] = 0

        split_weights = all_weights.split(all_numels)
        for i, (k, v) in enumerate(pruned_state_dict.items()):
            pruned_state_dict[k] = split_weights[i].view(v.shape)

        model.load_state_dict(pruned_state_dict)

    @torch.no_grad()
    def clean_environment(self, model: nn.Module) -> None:
        """Clean environment after testing model."""
        pass
