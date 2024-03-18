import logging
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from ..dissectors import Dissector
from ..utils import Device
from .builder import PRUNERS


@PRUNERS.register_module()
class ForwardPruner:

    def __init__(
        self,
        model: nn.Module,
        dual_insert_layer: Optional[str],
        sparsities: List[float],
    ) -> None:
        self.model = model
        self.dissector = Dissector(model=self.model, dual_insert_layer=dual_insert_layer)

        self.sparsities = sparsities

    def analyze_model(
        self, data_loader: DataLoader, work_dir: str, device: Device, logger: logging.Logger
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        all_forward_grads: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_activations: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_weights: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_biases: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_backward_grads: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore

        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):

            # TODO: need to be changed. Current hack only for backward compatibility
            batch = BatchEncoding(batch).to(device)
            input_ids = batch.pop("input_ids")
            dissect_results = self.dissector.dissect(input_ids, forward_kwargs=batch)
            forward_grads = dissect_results["forward_grads"]
            activations = dissect_results["activations"]
            weights = dissect_results["weights"]
            biases = dissect_results["biases"]
            backward_grads = dissect_results["backward_grads"]

            for k, forward_grad in forward_grads.items():
                # TODO caution, this only works if the output neuron dim is the last dim
                # avg over batch dim, accumulate over data loader (will be averaged later)
                forward_grad = forward_grad.abs().mean(list(range(forward_grad.ndim - 1)))
                all_forward_grads[k] += forward_grad

                backward_grad = backward_grads[k]
                backward_grad = backward_grad.abs().mean(list(range(backward_grad.ndim - 1)))
                all_backward_grads[k] += backward_grad

                all_activations[k] += activations[k].mean(0)

                # Weights and biases retrieval are repeated for all batches.
                # Therefore, they can be directly saved from the first batch.
                if batch_index < 1:
                    all_weights = weights
                    all_biases = biases

        for k, v in all_activations.items():
            all_activations[k] = v / len(data_loader)

        for k, v in all_forward_grads.items():
            all_forward_grads[k] = v / len(data_loader)

        for k, v in all_backward_grads.items():
            all_backward_grads[k] = v / len(data_loader)

        result = {
            "forward_grads": all_forward_grads,
            "activations": all_activations,
            "backward_grads": all_backward_grads,
            "weights": all_weights,
            "biases": all_biases,
        }

        torch.save(all_forward_grads, osp.join(work_dir, "forward_grads.pth"))
        logger.info(f"Forward grads are saved to {osp.join(work_dir, 'forward_grads.pth')}")
        torch.save(all_activations, osp.join(work_dir, "activations.pth"))
        logger.info(f"Activations are saved to {osp.join(work_dir, 'activations.pth')}")
        torch.save(all_backward_grads, osp.join(work_dir, "backward_grads.pth"))
        logger.info(f"Backward grads are saved to {osp.join(work_dir, 'backward_grads.pth')}")
        torch.save(all_weights, osp.join(work_dir, "weights.pth"))
        logger.info(f"Weights are saved to {osp.join(work_dir, 'weights.pth')}")
        torch.save(all_biases, osp.join(work_dir, "biases.pth"))
        logger.info(f"Biases are saved to {osp.join(work_dir, 'biases.pth')}")

        return result

    @staticmethod
    def load_analysis_result(result_dir: str, device: Device) -> Dict:
        all_keys = ["forward_grads", "backward_grads", "activations", "weights", "biases"]
        results = {k: torch.load(osp.join(result_dir, f"{k}.pth"), map_location=device) for k in all_keys}
        return results

    def prune(self, analyze_result: Dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
        forward_grads = analyze_result["forward_grads"]
        backward_grads = analyze_result["backward_grads"]
        activations = analyze_result["activations"]
        weights = analyze_result["weights"]
        biases = analyze_result["biases"]

        mask_save_dir = osp.join(work_dir, "pruning_masks")
        mmengine.mkdir_or_exist(mask_save_dir)

        # shape info stores the output's shape and number of neurons
        shape_info = dict()
        flatten_forward_grads = []

        for k, v in forward_grads.items():
            flatten_forward_grads.append(v.flatten())
            shape_info.update({k: (v.shape, v.numel())})

        # concatenate the flattened forward grads and record length of each chunk
        concat_forward_grads = torch.concat(flatten_forward_grads, dim=0)
        split_size = [v[1] for v in shape_info.values()]
        mask_state_dict = dict()

        for sparsity in self.sparsities:
            # global binary masks
            top_k = int(concat_forward_grads.numel() * (1 - sparsity))
            _, top_k_inds = torch.topk(concat_forward_grads, top_k, sorted=False, largest=True)
            binary_mask = torch.zeros_like(concat_forward_grads, dtype=torch.bool)
            binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
            global_binary_masks = binary_mask.split(dim=-1, split_size=split_size)

            # local binary masks
            local_binary_masks = []
            for forward_grad in flatten_forward_grads:
                top_k = int(forward_grad.numel() * (1 - sparsity))
                _, top_k_inds = torch.topk(forward_grad, top_k, sorted=False, largest=True)
                local_binary_mask = torch.zeros_like(forward_grad, dtype=torch.bool)
                local_binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
                local_binary_masks.append(local_binary_mask)

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

            torch.save(
                mask_state_dict,
                osp.join(mask_save_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'),
            )

        logger.info(f"Pruning masks are saved to {mask_save_dir}")
