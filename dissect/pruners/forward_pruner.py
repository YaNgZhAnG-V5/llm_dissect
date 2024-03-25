import logging
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from transformers import BatchEncoding

from ..dissectors import Dissector
from ..models import MaskingHook
from ..utils import Device, name_contains_keys
from .binary_mask_mixin import BinaryMaskMixin
from .builder import PRUNERS, TESTING_MANAGER


@PRUNERS.register_module()
class ForwardPruner(BinaryMaskMixin):

    def __init__(
        self,
        model: nn.Module,
        dual_insert_layer: Optional[str],
        criterion: Dict[str, Any],
        sparsities: List[float],
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.dissector = Dissector(model=self.model, dual_insert_layer=dual_insert_layer)

        self.sparsities = sparsities

    def analyze_model(
        self, data_loader: DataLoader, work_dir: str, device: Device, logger: logging.Logger
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        all_forward_grads: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_inputs: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_activations: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_weights: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_biases: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_backward_grads: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore
        all_backward_grads_activations: Dict[str, torch.Tensor] = defaultdict(float)  # type: ignore

        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):

            # TODO: need to be changed. Current hack only for backward compatibility
            batch = BatchEncoding(batch).to(device)
            input_ids = batch.pop("input_ids")
            dissect_results = self.dissector.dissect(input_ids, forward_kwargs=batch)
            forward_grads = dissect_results["forward_grads"]
            backward_grads = dissect_results["backward_grads"]
            activations = dissect_results["activations"]
            inputs = dissect_results["inputs"]

            # Weights and biases retrieval are repeated for all batches.
            # Therefore, they can be directly saved from the first batch.
            all_weights = dissect_results["weights"]
            all_biases = dissect_results["biases"]

            for k, forward_grad in forward_grads.items():
                # TODO caution, this only works if the output neuron dim is the last dim
                # avg over batch dim, accumulate over data loader (will be averaged later)
                forward_grad = forward_grad.abs().mean(list(range(forward_grad.ndim - 1)))
                all_forward_grads[k] += forward_grad

                backward_grad = backward_grads[k]
                backward_grad = backward_grad.abs().mean(list(range(backward_grad.ndim - 1)))
                all_backward_grads[k] += backward_grad

                activation = activations[k]
                activation = activation.abs().mean(list(range(activation.ndim - 1)))
                all_activations[k] += activation

                # save backward_grad * activation
                all_backward_grads_activations[k] += backward_grad * activation

                # for now we take the l2 norm of input tensor across N*L
                # follows exactly the original wanda implementation
                all_inputs[k] += inputs[k]

        for k, v in all_activations.items():
            all_activations[k] = v / len(data_loader)

        for k, v in all_forward_grads.items():
            all_forward_grads[k] = v / len(data_loader)

        for k, v in all_backward_grads.items():
            all_backward_grads[k] = v / len(data_loader)

        for k, v in all_activations.items():
            all_activations[k] = v / len(data_loader)

        for k, v in all_backward_grads_activations.items():
            all_backward_grads_activations[k] = v / len(data_loader)

        for k, v in all_inputs.items():
            all_inputs[k] = v / len(data_loader)

        # for weights, the first dim is the output dim, so we need to average over the rest dims
        for k, v in all_weights.items():
            all_weights[k] = v.abs().mean(list(range(1, v.ndim)))

        result = {
            "forward_grads": all_forward_grads,
            "activations": all_activations,
            "inputs": all_inputs,
            "backward_grads": all_backward_grads,
            "backward_grads_activations": all_backward_grads_activations,
            "weights": all_weights,
            "biases": all_biases,
        }

        torch.save(all_forward_grads, osp.join(work_dir, "forward_grads.pth"))
        logger.info(f"Forward grads are saved to {osp.join(work_dir, 'forward_grads.pth')}")
        torch.save(all_activations, osp.join(work_dir, "activations.pth"))
        logger.info(f"Activations are saved to {osp.join(work_dir, 'activations.pth')}")
        torch.save(all_inputs, osp.join(work_dir, "inputs.pth"))
        logger.info(f"Inputs are saved to {osp.join(work_dir, 'inputs.pth')}")
        torch.save(all_backward_grads, osp.join(work_dir, "backward_grads.pth"))
        logger.info(f"Backward grads are saved to {osp.join(work_dir, 'backward_grads.pth')}")
        torch.save(all_backward_grads_activations, osp.join(work_dir, "backward_grads_activations.pth"))
        logger.info(
            f"Backward grads times activations are saved to {osp.join(work_dir, 'backward_grads_activations.pth')}"
        )
        torch.save(all_weights, osp.join(work_dir, "weights.pth"))
        logger.info(f"Weights are saved to {osp.join(work_dir, 'weights.pth')}")
        torch.save(all_biases, osp.join(work_dir, "biases.pth"))
        logger.info(f"Biases are saved to {osp.join(work_dir, 'biases.pth')}")

        return result

    @staticmethod
    def load_analysis_result(result_dir: str, device: Device) -> Dict:
        all_keys = [
            "forward_grads",
            "backward_grads",
            "backward_grads_activations",
            "activations",
            "inputs",
            "weights",
            "biases",
        ]
        results = {k: torch.load(osp.join(result_dir, f"{k}.pth"), map_location=device) for k in all_keys}
        return results

    def prune(self, analyze_result: Dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
        mask_save_dir = osp.join(work_dir, "pruning_masks")
        mmengine.mkdir_or_exist(mask_save_dir)

        # get stats based on the strategy
        strategy = self.criterion["strategy"]
        if isinstance(strategy, (list, tuple)):
            # the case of (abs(activation) * abs(forward_grad)) or (abs(activation) * abs(backward_grad))
            # TODO: it introduces overhead if first computing product for all layers and then filtering layers
            assert len(strategy) == 2, f"strategy (List) should be have length of 2, but got {len(strategy)}"
            result_dict_0 = analyze_result[strategy[0]]
            result_dict_1 = analyze_result[strategy[1]]
            stats: Dict[str, torch.Tensor] = {k: v * result_dict_1[k] for k, v in result_dict_0.items()}
        else:
            # the case where each value in analyze_result is a Tensor
            stats: Dict[str, torch.Tensor] = analyze_result[self.criterion["strategy"]]

        # remove not interested layers
        exclude_layer_names = [k for k in stats.keys() if name_contains_keys(k, self.criterion["exclude_layers"])]
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


@TESTING_MANAGER.register_module()
class ForwardPrunerTestingManager:
    """Manager for loading masks, applying masking hooks, and cleaning hooks."""

    def __init__(self):
        self.test_handle_dict = dict()

    def prepare_environment(
        self,
        model: nn.Module,
        mask_path: str,
        device: Device,
        exclude_layers: List[str] = (),
        prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Prepare environment for testing model."""
        mask_state_dict = torch.load(mask_path, map_location=device)
        handle_dict: Dict[str, RemovableHandle] = dict()

        for layer_name, pruning_mask in mask_state_dict.items():
            if name_contains_keys(layer_name, exclude_layers):
                continue
            prior = None if prior_state_dict is None else prior_state_dict[layer_name]
            layer = model.get_submodule(layer_name)
            hook = MaskingHook(pruning_mask, prior=prior)
            handle = layer.register_forward_hook(hook)
            handle_dict.update({layer_name: handle})

        self.test_handle_dict = handle_dict

    def clean_environment(self, model: nn.Module) -> None:
        """Clean environment after testing model."""
        for handle in self.test_handle_dict.values():
            handle.remove()
        self.test_handle_dict.clear()
