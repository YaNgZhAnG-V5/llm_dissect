import logging
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Optional

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from ..utils import Device
from .binary_mask_mixin import BinaryMaskMixin
from .builder import PRUNERS, TESTING_MANAGER


@PRUNERS.register_module()
class WeightGradientsPruner(BinaryMaskMixin):

    def __init__(self, model: nn.Module, criterion: Dict[str, Any], sparsities: List[float]) -> None:
        self.model = model
        self.criterion = criterion
        assert (
            self.criterion["strategy"] == "weight_grads"
        ), "WeightGradientsPruner only support 'weight_grads' strategy."
        self.sparsities = sparsities

    def analyze_model(
        self, data_loader: DataLoader, work_dir: str, device: Device, logger: logging.Logger
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        weight_grads: Dict[str, torch.Tensor] = dict()
        # dummy optimizer, only used for reset gradients
        optimizer = Adam(self.model.parameters())
        num_processed_samples = 0

        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
            # reset gradients to avoid gradient accumulation
            optimizer.zero_grad()
            labels = batch.pop("label").to(device)
            batch = BatchEncoding(batch).to(device)

            loss = self.model(**batch, labels=labels).loss
            loss.backward()

            num_current_samples = labels.shape[0]
            for param_name, param in self.model.named_parameters():
                prev_avg_grad = weight_grads.get(param_name, 0.0)
                new_avg_grad = (prev_avg_grad * num_processed_samples + param.grad.abs() * num_current_samples) / (
                    num_processed_samples + num_current_samples
                )
                weight_grads.update({param_name: new_avg_grad})

            num_processed_samples += num_current_samples

        results = {"weight_grads": weight_grads}

        torch.save(weight_grads, osp.join(work_dir, "weight_grads.pth"))
        logger.info(f"Weight gradients are saved to {osp.join(work_dir, 'weight_grads.pth')}")
        return results

    @staticmethod
    def load_analysis_result(result_dir: str, device: Device) -> Dict:
        file_path = osp.join(result_dir, "weight_grads.pth")
        results = {"weight_grads": torch.load(file_path, map_location=device)}
        return results

    def prune(self, analyze_result: Dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
        mask_save_dir = osp.join(work_dir, "pruning_masks")
        mmengine.mkdir_or_exist(mask_save_dir)

        # get stats based on the strategy
        stats: Dict[str, torch.Tensor] = analyze_result[self.criterion["strategy"]]

        # remove not interested layers
        exclude_layer_names = [
            k for k in stats.keys() if any(exclude_key in k for exclude_key in self.criterion["exclude_layers"])
        ]
        for layer_name in exclude_layer_names:
            stats.pop(layer_name)

        # shape info stores the output's shape and number of neurons
        shape_info = dict()
        flatten_stats = []

        for k, v in stats.items():
            flatten_stats.append(v.flatten())
            shape_info.update({k: (v.shape, v.numel())})

        # concatenate the flattened stats and record length of each chunk
        concat_stats = torch.concat(flatten_stats, dim=0)
        split_size = [v[1] for v in shape_info.values()]
        if self.criterion["scope"] == "global":
            prune_fn = self.global_prune
        elif self.criterion["scope"] == "local":
            prune_fn = self.local_prune
        elif self.criterion["scope"] == "global_thres":
            prune_fn = self.global_thres_prune
        else:
            raise NotImplementedError(f"Unknown pruning scope: {self.criterion['scope']}")
        for sparsity in self.sparsities:
            mask_state_dict = prune_fn(
                sparsity=sparsity,
                flatten_stats=flatten_stats,
                concat_stats=concat_stats,
                split_size=split_size,
                shape_info=shape_info,
            )
            torch.save(
                mask_state_dict,
                osp.join(mask_save_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'),
            )

        logger.info(f"Pruning masks are saved to {mask_save_dir}")

    @staticmethod
    def get_testing_setup_manager():
        return WeightGradientsTestingManager()


@TESTING_MANAGER.register_module()
class WeightGradientsTestingManager:
    """Manager for loading weight masks, applying masks to weights, and restoring the original weights"""

    @torch.no_grad()
    def prepare_environment(
        self,
        model: nn.Module,
        mask_path: str,
        ori_state_dict: Dict[str, torch.Tensor],
        device: Device,
        exclude_layers: List[str] = (),
        prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Prepare environment for testing model."""
        assert prior_state_dict is None, "prior_state_dict is only for API compatibility"
        mask_state_dict = torch.load(mask_path, map_location=device)
        # deep-copy original state dict to avoid in-place modification of original weights
        state_dict = deepcopy(ori_state_dict)
        for param_name, param in state_dict.items():
            if any(exclude_layer in param_name for exclude_layer in exclude_layers):
                continue
            # binary_mask: 1 means keep, 0 means set to 0
            binary_mask = mask_state_dict[param_name]
            param[~binary_mask] = 0.0
        model.load_state_dict(state_dict)
        logger = mmengine.MMLogger.get_instance("dissect")
        logger.info("model loaded pruned state dict.")

    @torch.no_grad()
    def clean_environment(self, model: nn.Module, ori_state_dict: Dict[str, torch.Tensor]) -> None:
        """Clean environment after testing model."""
        logger = mmengine.MMLogger.get_instance("dissect")
        model.load_state_dict(ori_state_dict)
        logger.info("model re-loaded original state dict.")
