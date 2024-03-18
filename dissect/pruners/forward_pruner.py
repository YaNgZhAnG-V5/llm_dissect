import logging
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from ..dissectors import ActivationExtractor, ForwardADExtractor
from ..utils import Device
from .builder import PRUNERS


@PRUNERS.register_module()
class ForwardPruner:

    def __init__(
        self,
        model: nn.Module,
        dual_insert_layer: Optional[str],
        criterion: Dict[str, Any],
        sparsities: List[float],
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.jvp_extractor = ForwardADExtractor(model=self.model, dual_insert_layer=dual_insert_layer)
        self.act_extractor = ActivationExtractor(model=self.model)

        self.sparsities = sparsities

    def analyze_model(
        self, data_loader: DataLoader, work_dir: str, device: Device, logger: logging.Logger
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        all_forward_grads: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_activations: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore

        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):

            # TODO: need to be changed. Current hack only for backward compatibility
            batch = BatchEncoding(batch).to(device)
            input_ids = batch.pop("input_ids")
            # _ = batch.pop('token_type_ids', None)
            forward_grads = self.jvp_extractor.forward_ad(input_ids, forward_kwargs=batch)
            activations = self.act_extractor.extract_activations(input_ids, forward_kwargs=batch)

            for k, v in forward_grads.items():
                # avg over batch dim, accumulate over data loader (will be averaged later)
                v = v.abs()
                # TODO caution, this only works if the output neuron dim is the last dim
                v = v.mean(list(range(v.ndim - 1)))
                all_forward_grads[k] += v
                all_activations[k] += activations[k].mean(0)

        for k, v in all_activations.items():
            all_activations[k] = v / len(data_loader)

        for k, v in all_forward_grads.items():
            all_forward_grads[k] = v / len(data_loader)

        result = {"forward_grads": all_forward_grads, "activations": all_activations}
        # save average forward grads and activations
        torch.save(all_forward_grads, osp.join(work_dir, "forward_grads.pth"))
        logger.info(f"Forward grads are saved to {osp.join(work_dir, 'forward_grads.pth')}")
        torch.save(all_activations, osp.join(work_dir, "activations.pth"))
        logger.info(f"Activations are saved to {osp.join(work_dir, 'activations.pth')}")

        return result

    @staticmethod
    def load_analysis_result(result_dir: str, device: Device) -> Dict:
        forward_grads = torch.load(osp.join(result_dir, "forward_grads.pth"), map_location=device)
        return {"forward_grads": forward_grads}

    def prune(self, analyze_result: Dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
        # get stats based on the strategy
        stats = analyze_result[self.criterion["strategy"]]
        mask_save_dir = osp.join(work_dir, "pruning_masks")
        mmengine.mkdir_or_exist(mask_save_dir)

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
    def _get_global_binary_mask(
        concat_stats: torch.Tensor, sparsity: float, split_size: List[int]
    ) -> List[torch.Tensor]:
        """get global binary masks for the model given the sparsity rate"""
        top_k = int(concat_stats.numel() * (1 - sparsity))
        _, top_k_inds = torch.topk(concat_stats, top_k, sorted=False, largest=True)
        binary_mask = torch.zeros_like(concat_stats, dtype=torch.bool)
        binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
        global_binary_masks = binary_mask.split(dim=-1, split_size=split_size)
        return global_binary_masks

    @staticmethod
    def _get_local_binary_masks(flatten_stats: List[torch.Tensor], sparsity: float) -> List[torch.Tensor]:
        """get local binary masks for each layer given the sparsity rate"""
        local_binary_masks = []
        for stat in flatten_stats:
            top_k = int(stat.numel() * (1 - sparsity))
            _, top_k_inds = torch.topk(stat, top_k, sorted=False, largest=True)
            local_binary_mask = torch.zeros_like(stat, dtype=torch.bool)
            local_binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
            local_binary_masks.append(local_binary_mask)
        return local_binary_masks

    def global_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """prune the model globally (rank all neurons diregard their belonging layers) with the sparsity rate"""
        mask_state_dict = dict()
        global_binary_masks = self._get_global_binary_mask(
            concat_stats=concat_stats, sparsity=sparsity, split_size=split_size
        )
        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: global_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict

    def local_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """prune the model locally (rank neurons within each layer) with the sparsity rate"""
        mask_state_dict = dict()
        local_binary_masks = self._get_local_binary_masks(flatten_stats=flatten_stats, sparsity=sparsity)
        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: local_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict

    def global_thres_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """global prune with pruning rate thresholding at each layer"""
        mask_state_dict = dict()

        # gat global and local binary masks
        global_binary_masks = self._get_global_binary_mask(
            concat_stats=concat_stats, sparsity=sparsity, split_size=split_size
        )
        local_binary_masks = self._get_local_binary_masks(flatten_stats=flatten_stats, sparsity=sparsity)

        final_binary_masks = []
        for global_binary_mask, local_binary_mask in zip(global_binary_masks, local_binary_masks):
            actual_sparsity = 1 - global_binary_mask.float().sum() / global_binary_mask.numel()
            # enforce per-layer actual sparsity is no greater than the specified sparsity
            if actual_sparsity > sparsity:
                final_binary_mask = local_binary_mask
            else:
                final_binary_mask = global_binary_mask

            final_binary_masks.append(final_binary_mask)

        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: final_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict
