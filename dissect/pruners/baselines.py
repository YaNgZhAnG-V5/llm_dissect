from typing import List

import mmengine
import torch
import torch.nn as nn

from ..utils import Device, name_contains_keys
from .builder import TESTING_MANAGER

# Wanda does not need a pruner. It only needs a testing manager.


@TESTING_MANAGER.register_module()
class WandaTestingManager:
    """Manager for loading Wanda inputs, applying masks to weights"""

    def __init__(self, inputs_path: str, device: Device = "cuda:0") -> None:
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


# Magnitude pruning does not need a pruner. It only needs a testing manager.


@TESTING_MANAGER.register_module()
class MagnitudeTestingManager:
    """Manager applying masks to weights"""

    def __init__(self, device: Device = "cuda:0") -> None:
        # just for API compatibility
        pass

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
        # all weight tensors need to be pruned and their number of elements.
        all_abs_weights = []
        all_numels = []
        all_pruned_param_names = []
        for k, v in pruned_state_dict.items():
            if name_contains_keys(k, exclude_layers):
                continue
            all_pruned_param_names.append(k)
            all_abs_weights.append(torch.flatten(v.abs()))
            all_numels.append(v.numel())

        all_abs_weights = torch.concat(all_abs_weights, 0)
        # top_k_inds point to the top (1 - sparsity) weights, which should be kept
        _, top_k_inds = torch.topk(all_abs_weights, int(all_abs_weights.numel() * (1 - sparsity)))

        # pruned_mask: 1 for setting weight to 0, 0 for keep original weight
        pruned_mask = torch.ones_like(all_abs_weights, dtype=torch.bool)
        pruned_mask[top_k_inds] = 0

        split_pruned_masks = pruned_mask.split(all_numels)
        assert len(split_pruned_masks) == len(all_pruned_param_names)
        for param_name, mask in zip(all_pruned_param_names, split_pruned_masks):
            param = pruned_state_dict[param_name]
            mask = mask.view(param.shape)
            # applying pruned_mask
            param[mask] = 0
            pruned_state_dict[param_name] = param

        model.load_state_dict(pruned_state_dict)

    @torch.no_grad()
    def clean_environment(self, model: nn.Module) -> None:
        """Clean environment after testing model."""
        pass
