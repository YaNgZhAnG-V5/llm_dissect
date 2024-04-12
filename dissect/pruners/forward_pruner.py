import logging
import os.path as osp
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from transformers import BatchEncoding
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv

from ..dissectors import Dissector
from ..models import MaskingHook, build_model_and_tokenizer
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
        use_loss: bool,
        dissector_options: dict,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.use_loss = use_loss
        self.dissector = Dissector(
            model=self.model, dual_insert_layer=dual_insert_layer, dissector_options=dissector_options
        )

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
            batch = BatchEncoding(batch).to(device)
            input_ids = batch.pop("input_ids")
            dissect_results = self.dissector.dissect(input_ids, forward_kwargs=batch, use_loss=self.use_loss)
            forward_grads = dissect_results["forward_grads"]
            backward_grads = dissect_results["backward_grads"]
            activations = dissect_results["activations"]
            inputs = dissect_results["inputs"]

            # Weights and biases retrieval are repeated for all batches.
            # Therefore, they can be directly saved from the first batch.
            all_weights = dissect_results["weights"]
            all_biases = dissect_results["biases"]
            layers = forward_grads.keys() if forward_grads is not None else backward_grads.keys()
            for layer in layers:
                # TODO caution, this only works if the output neuron dim is the last dim
                # avg over batch dim, accumulate over data loader (will be averaged later)
                if backward_grads is not None:
                    backward_grad = backward_grads[layer]
                    backward_grad = backward_grad.abs().mean(list(range(backward_grad.ndim - 1)))
                    all_backward_grads[layer] += backward_grad

                if forward_grads is not None:
                    forward_grad = forward_grads[layer]
                    forward_grad = forward_grad.abs().mean(list(range(forward_grad.ndim - 1)))
                else:
                    forward_grad = torch.zeros_like(backward_grad)
                all_forward_grads[layer] += forward_grad

                if activations is not None:
                    activation = activations[layer]
                    activation = activation.abs().mean(list(range(activation.ndim - 1)))
                else:
                    activation = torch.zeros_like(backward_grad)
                all_activations[layer] += activation

                # save backward_grad * activation
                all_backward_grads_activations[layer] += backward_grad * activation

                # for now we take the l2 norm of input tensor across N*L
                # follows exactly the original wanda implementation
                if inputs is not None:
                    all_inputs[layer] += inputs[layer]
                else:
                    all_inputs[layer] += torch.zeros_like(backward_grad)

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
        if all_weights is not None:
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
            "activations",
            "backward_grads_activations",
            "inputs",
            "weights",
            "biases",
        ]
        results = {k: torch.load(osp.join(result_dir, f"{k}.pth"), map_location=device) for k in all_keys}
        return results

    @staticmethod
    def _get_target_layers(
        stats: Dict[str, torch.Tensor], target_name: str, group: List[str], ori_exclude_layers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """exclude layers from stats that are not target layers in the group and are excluded layers."""
        copy_stats = deepcopy(stats)
        exclude_layers = deepcopy(ori_exclude_layers)
        for name in group:
            if name != target_name:
                exclude_layers.append(name)
        exclude_layer_names = [k for k in copy_stats.keys() if name_contains_keys(k, exclude_layers)]
        for layer_name in exclude_layer_names:
            copy_stats.pop(layer_name)
        return copy_stats

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

        # get pruning function based on the scope
        if self.criterion["scope"] == "global":
            prune_fn = self.global_prune
        elif self.criterion["scope"] == "local":
            prune_fn = self.local_prune
        elif self.criterion["scope"] == "global_thres":
            prune_fn = self.global_thres_prune
        elif self.criterion["scope"] == "global_thres_per_head":
            prune_fn = self.global_thres_per_head_prune
        else:
            raise NotImplementedError(f"Unknown pruning scope: {self.criterion['scope']}")

        for sparsity in self.sparsities:
            all_mask_state_dict = dict()
            for target_name in self.criterion["group"]:
                # remove not interested layers
                copy_stats = self._get_target_layers(
                    stats, target_name, self.criterion["group"], self.criterion["exclude_layers"]
                )

                # shape info stores the output's shape and number of neurons, head info stores the # of heads
                shape_info = dict()
                head_info = dict()
                flatten_stats = []

                # merge Q and K stats
                if self.criterion["identical_prune_k_q"]:
                    for k, v in copy_stats.items():
                        if "k_proj" in k and k.replace("k_proj", "q_proj") in copy_stats:
                            copy_stats[k] *= copy_stats[k.replace("k_proj", "q_proj")]
                            copy_stats[k.replace("k_proj", "q_proj")] = copy_stats[k]

                for k, v in copy_stats.items():
                    flatten_stats.append(v.flatten())
                    shape_info.update({k: (v.shape, v.numel())})

                # concatenate the flattened stats and record length of each chunk
                concat_stats = torch.concat(flatten_stats, dim=0)
                split_size = [v[1] for v in shape_info.values()]

                # perform prune
                mask_state_dict = prune_fn(
                    sparsity=sparsity,
                    flatten_stats=flatten_stats,
                    concat_stats=concat_stats,
                    split_size=split_size,
                    shape_info=shape_info,
                    **self.criterion["params"],
                )
                all_mask_state_dict.update(mask_state_dict)
            torch.save(
                all_mask_state_dict,
                osp.join(mask_save_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'),
            )

        logger.info(f"Pruning masks are saved to {mask_save_dir}")


@TESTING_MANAGER.register_module()
class ForwardPrunerTestingManager:
    """Manager for loading masks, applying masking hooks, and cleaning hooks."""

    def __init__(self, prune_input: List[str]):
        self.test_handle_dict = dict()
        self.mask_state_dict = None
        self.backup_forward = None  # to store the original forward function
        self.prune_input = prune_input

    @staticmethod
    def merge_mask(mask_state_dict):
        for k in sorted(mask_state_dict.keys()):
            # TODO: later might need to pass q and k name to account for different models
            if "q_proj" in k:
                k_proj_mask = mask_state_dict[k.replace("q_proj", "k_proj")]
                mask_state_dict[k] = torch.logical_and(mask_state_dict[k], k_proj_mask)
            elif "k_proj" in k:
                q_proj_mask = mask_state_dict[k.replace("k_proj", "q_proj")]
                mask_state_dict[k] = torch.logical_and(mask_state_dict[k], q_proj_mask)
        return mask_state_dict

    def calc_pruned_parameters(
        self, model: nn.Module, mask_state_dict: Dict, ori_param_count_dict: Dict, in_place: bool
    ) -> Tuple[List[Dict[str, float]], float, float]:
        """
        calculate the number of pruned parameters in each layer and the total number of pruned parameters
        assume neuron pruning
        param model: the model to be pruned
        param mask_state_dict: the mask state dict that is used to mask the model
        return: a list of dictionaries, each dictionary contains the layer name, neuron sparsity, and parameter sparsity
        return: the total sparsity ratio in the target layers
        return: the total sparsity ratio in the entire model
        """
        # get total number of parameters, ignore bias
        total_params_model = sum(p for p in ori_param_count_dict.values())
        pruned_parameters = 0

        # calculate total_params_target_layers only once since it wont change
        total_params_target_layers = sum(ori_param_count_dict[k + ".weight"] for k in mask_state_dict.keys())

        log_tabulate = []
        for k in sorted(mask_state_dict.keys()):
            # calculation for inplace prune and masking prune are different
            total_params_layer = ori_param_count_dict[k + ".weight"]
            if in_place:
                actual_params_layer = model.get_submodule(k).weight.data.numel()
                pruned_parameters_layer = total_params_layer - actual_params_layer
            else:
                mask_input = True if any([target_name in k for target_name in self.prune_input]) else False
                size_untouched_neurons = (
                    model.get_submodule(k).weight.data.shape[0]
                    if mask_input
                    else model.get_submodule(k).weight.data.shape[1]
                )
                pruned_parameters_layer = (
                    total_params_layer - mask_state_dict[k].float().sum().item() * size_untouched_neurons
                )
            pruned_parameters += pruned_parameters_layer

            log_tabulate.append(
                {
                    "layer": k,
                    "neuron_sparsity": 1 - mask_state_dict[k].float().mean().item(),
                    "param_sparsity": pruned_parameters_layer / total_params_layer,
                }
            )

        # get global pruning ratio
        sparsity_target_layers = pruned_parameters / total_params_target_layers
        sparsity_whole_model = pruned_parameters / total_params_model
        return log_tabulate, sparsity_target_layers, sparsity_whole_model

    def prepare_environment(
        self,
        model: nn.Module,
        mask_path: str,
        device: Device,
        in_place: bool = False,
        prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Prepare environment for testing model."""
        if in_place:
            self.prepare_environment_inplace(
                model=model,
                mask_path=mask_path,
                device=device,
            )
        else:
            self.prepare_environment_mask_hook(
                model=model, mask_path=mask_path, device=device, prior_state_dict=prior_state_dict
            )

    def prepare_environment_mask_hook(
        self,
        model: nn.Module,
        mask_path: str,
        device: Device,
        prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Prepare environment for testing model by adding neuron mask."""
        self.mask_state_dict = torch.load(mask_path, map_location=device)
        handle_dict: Dict[str, RemovableHandle] = dict()

        for layer_name, pruning_mask in self.mask_state_dict.items():
            prior = None if prior_state_dict is None else prior_state_dict[layer_name]
            layer = model.get_submodule(layer_name)
            mask_input = True if any([target_name in layer_name for target_name in self.prune_input]) else False
            hook = MaskingHook(pruning_mask, prior=prior, mask_input=mask_input)
            handle = layer.register_forward_hook(hook)
            handle_dict.update({layer_name: handle})

        self.test_handle_dict = handle_dict

    def prepare_environment_inplace(
        self,
        model: nn.Module,
        mask_path: str,
        device: Device,
    ) -> None:
        """Prepare environment for testing model by performing inplace neuron pruning on models"""
        self.backup_forward = LlamaSdpaAttention.forward
        LlamaSdpaAttention.forward = pruned_forward
        self.mask_state_dict = torch.load(mask_path, map_location=device)
        for layer_name, pruning_mask in self.mask_state_dict.items():
            # Apply resized weight and bias
            layer = model.get_submodule(layer_name)
            assert isinstance(layer, nn.Linear), "only support linear layer for now."
            prune_input = True if any([target_name in layer_name for target_name in self.prune_input]) else False
            if prune_input:
                layer = self.reduce_linear_input(layer, pruning_mask)
            else:
                layer = self.reduce_linear_output(layer, pruning_mask)

    @staticmethod
    def reduce_linear_output(layer: nn.Module, pruning_mask: torch.Tensor):
        """reduce the output neuron of a linear layer"""
        layer.out_features = pruning_mask.sum().item()
        non_zero_indices = torch.sort(pruning_mask.float(), descending=True)[1][: layer.out_features]
        layer.weight.data = layer.weight.data[non_zero_indices, :]
        if layer.bias:
            layer.bias.data = layer.bias.data[non_zero_indices]
        return layer

    @staticmethod
    def reduce_linear_input(layer: nn.Module, pruning_mask: torch.Tensor):
        """reduce the input neuron of a linear layer"""
        layer.in_features = pruning_mask.sum().item()
        non_zero_indices = torch.sort(pruning_mask.float(), descending=True)[1][: layer.in_features]
        layer.weight.data = layer.weight.data[:, non_zero_indices]
        return layer

    def clean_environment_inplace(self, model_cfg, device) -> None:
        """Clean environment by reinitialize the model and recover the forward function."""
        model, _ = build_model_and_tokenizer(model_cfg, device=device)
        LlamaSdpaAttention.forward = self.backup_forward
        return model

    def clean_environment_hook(self) -> None:
        """Clean environment by removing hooks."""
        for handle in self.test_handle_dict.values():
            handle.remove()
        self.test_handle_dict.clear()


# monkey patch to make in-place prune work
# Adapted from LlamaAttention.forward
def pruned_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        return super(LlamaSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Customized code ################################
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    # Customized code ################################

    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # In case static cache is used, it is an instance attribute.
    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    # if attention_mask is not None and cache_position is not None:
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2)
    # bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    # Customized code ################################
    attn_output = attn_output.view(bsz, q_len, -1)
    # Customized code ################################

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
