from typing import Dict, List, Optional, Tuple

import mmengine
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv

from ..models import MaskingHook, build_model_and_tokenizer
from ..utils import Device
from .builder import TESTING_MANAGER


@TESTING_MANAGER.register_module()
class ForwardPrunerTestingManager:
    """Manager for loading masks, applying masking hooks, and cleaning hooks."""

    def __init__(self, prune_input: List[str], in_place: bool) -> None:
        self.test_handle_dict = dict()
        self.mask_state_dict = None
        self.backup_forward = None  # to store the original forward function
        self.prune_input = prune_input
        self.in_place = in_place
        if self.in_place:
            logger = mmengine.MMLogger.get_current_instance()
            logger.info("TestingManager: In-place testing environment will be used.")

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
        self,
        model: nn.Module,
        mask_state_dict: Dict,
        ori_param_count_dict: Dict,
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
        logger = mmengine.MMLogger.get_current_instance()
        logger.warning("calc_pruned_parameters is deprecated. Dummy values will be returned.")
        return [], 0.0, 0.0

        # get total number of parameters, ignore bias
        total_params_model = sum(p for p in ori_param_count_dict.values())
        pruned_parameters = 0

        # calculate total_params_target_layers only once since it wont change
        total_params_target_layers = sum(ori_param_count_dict[k + ".weight"] for k in mask_state_dict.keys())

        log_tabulate = []
        for k in sorted(mask_state_dict.keys()):
            # calculation for inplace prune and masking prune are different
            total_params_layer = ori_param_count_dict[k + ".weight"]
            if self.in_place:
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

    def load_mask_state_dict(self, mask_path: str, device):
        self.mask_state_dict = torch.load(mask_path, map_location=device)

    def prepare_environment(
        self,
        model: nn.Module,
        model_cfg: Dict,
        mask_path: Optional[str] = None,
        device: Optional[Device] = None,
        prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """Prepare environment for testing model."""
        if mask_path is not None and device is not None:
            self.load_mask_state_dict(mask_path, device)
        if self.in_place:
            return self.prepare_environment_inplace(
                model=model,
                model_cfg=model_cfg,
                device=device,
            )
        else:
            return self.prepare_environment_mask_hook(
                model=model, model_cfg=model_cfg, prior_state_dict=prior_state_dict
            )

    def prepare_environment_mask_hook(
        self,
        model: nn.Module,
        model_cfg: Dict,
        prior_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """Prepare environment for testing model by adding neuron mask."""
        handle_dict: Dict[str, RemovableHandle] = dict()

        for layer_name, pruning_mask in self.mask_state_dict.items():
            prior = None if prior_state_dict is None else prior_state_dict[layer_name]
            layer = model.get_submodule(layer_name)
            mask_input = True if any([target_name in layer_name for target_name in self.prune_input]) else False
            hook = MaskingHook(pruning_mask, prior=prior, mask_input=mask_input)
            handle = layer.register_forward_hook(hook)
            handle_dict.update({layer_name: handle})

        self.test_handle_dict = handle_dict
        return model

    def prepare_environment_inplace(
        self,
        model: nn.Module,
        model_cfg: Dict,
        device: Device,
    ) -> nn.Module:
        """Prepare environment for testing model by performing inplace neuron pruning on models"""
        # First swap forward fn for self-attn, and then rebuild the model,
        # otherwise this swap will not work in multi-gpu inference, which is implemented via huggingface.accelerate
        self.backup_forward = LlamaSdpaAttention.forward
        LlamaSdpaAttention.forward = pruned_forward
        model, _ = build_model_and_tokenizer(model_cfg, device=device)
        for layer_name, pruning_mask in self.mask_state_dict.items():
            # Apply resized weight and bias
            layer = model.get_submodule(layer_name)
            # Move pruning_mask to the layer's device. In multi-gpu inference, layers are located on different GPUs.
            layer_device = next(iter(layer.parameters())).device
            pruning_mask = pruning_mask.to(layer_device)

            assert isinstance(layer, nn.Linear), "only support linear layer for now."
            prune_input = True if any([target_name in layer_name for target_name in self.prune_input]) else False
            if prune_input:
                layer = self.reduce_linear_input(layer, pruning_mask)
            else:
                layer = self.reduce_linear_output(layer, pruning_mask)

        return model

    @staticmethod
    def reduce_linear_output(layer: nn.Module, pruning_mask: torch.Tensor) -> nn.Module:
        """reduce the output neuron of a linear layer"""
        layer.out_features = pruning_mask.sum().item()
        non_zero_indices = torch.sort(pruning_mask.float(), descending=True)[1][: layer.out_features]
        layer.weight.data = layer.weight.data[non_zero_indices, :]
        if layer.bias:
            layer.bias.data = layer.bias.data[non_zero_indices]
        return layer

    @staticmethod
    def reduce_linear_input(layer: nn.Module, pruning_mask: torch.Tensor) -> nn.Module:
        """reduce the input neuron of a linear layer"""
        layer.in_features = pruning_mask.sum().item()
        non_zero_indices = torch.sort(pruning_mask.float(), descending=True)[1][: layer.in_features]
        layer.weight.data = layer.weight.data[:, non_zero_indices]
        return layer

    def clean_environment(self, model: nn.Module, model_cfg: Dict, device: Device) -> nn.Module:
        if self.in_place:
            return self.clean_environment_in_place(model=model, model_cfg=model_cfg, device=device)
        else:
            return self.clean_environment_hook(model=model, model_cfg=model_cfg, device=device)

    def clean_environment_in_place(self, model: nn.Module, model_cfg: Dict, device: Device) -> nn.Module:
        """Clean environment by recovering the forward function."""
        LlamaSdpaAttention.forward = self.backup_forward
        return model

    def clean_environment_hook(self, model: nn.Module, model_cfg: Dict, device: Device) -> nn.Module:
        """Clean environment by removing hooks."""
        for handle in self.test_handle_dict.values():
            handle.remove()
        self.test_handle_dict.clear()
        return model


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
