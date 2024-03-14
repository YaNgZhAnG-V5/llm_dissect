import logging
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Sequence

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
        sparsities: List[float],
    ) -> None:
        self.model = model
        self.jvp_extractor = ForwardADExtractor(model=self.model)
        self.act_extractor = ActivationExtractor(model=self.model)

        self.sparsities = sparsities

    def analyze_model(
        self,
        data_loader: DataLoader,
        device: Device,
        model_input_keys: Sequence[str],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        all_forward_grads: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_activations: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore

        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
            for k in batch.keys():
                if k not in model_input_keys:
                    _ = batch.pop(k)

                # TODO: need to be changed. Current hack only for backward compatibility
                batch = BatchEncoding(batch).to(device)
                input_ids = batch.pop("input_ids")
                forward_grads = self.jvp_extractor.forward_ad(input_ids, **batch)
                activations = self.act_extractor.extract_activations(input_ids, **batch)

                for k, v in forward_grads.items():
                    # avg over batch dim, accumulate over data loader (will be averaged later)
                    all_forward_grads[k] = v.abs().mean(0)
                    all_activations[k] += activations[k].mean(0)

        for k, v in all_activations.items():
            all_activations[k] = v / len(data_loader)

        for k, v in all_forward_grads.items():
            all_forward_grads[k] = v / len(data_loader)

        result = {"forward_grads": all_forward_grads, "activations": all_activations}
        return result

    def prune(self, analyze_result: Dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
        forward_grads = analyze_result["forward_grads"]
        activations = analyze_result["activations"]

        # save average activations
        torch.save(activations, osp.join(work_dir, "activations.pth"))
        logger.info(f"Activations are saved to {osp.join(work_dir, 'activations.pth')}")

        # shape info stores the output's shape and number of neurons
        shape_info = dict()
        flatten_forward_grads = []

        for k, v in forward_grads.items():
            flatten_forward_grads.append(v.flatten())
            shape_info.update({k: (v.shape, v.numel())})

        # concatenate the flattened forward grads and record length of each chunk
        flatten_forward_grads = torch.concat(flatten_forward_grads, dim=0)
        split_size = [v[1] for v in shape_info.values()]
        mask_state_dict = dict()

        mask_save_dir = osp.join(work_dir, "pruning_masks")

        for sparsity in self.sparsities:
            top_k = int(flatten_forward_grads.numel() * (1 - sparsity))
            _, top_k_inds = torch.topk(flatten_forward_grads, top_k, sorted=False, largest=True)
            binary_mask = torch.zeros_like(flatten_forward_grads, dtype=torch.bool)
            binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
            split_binary_masks = binary_mask.split(dim=-1, split_size=split_size)

            for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
                mask_state_dict.update({layer_name: split_binary_masks[i].reshape(forward_grad_shape)})

            torch.save(
                mask_state_dict,
                osp.join(mask_save_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'),
            )

        logger.info(f"Pruning masks are saved to {mask_save_dir}")
