import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from ptflops import get_model_complexity_info
from ptflops.pytorch_ops import pool_flops_counter_hook
from torch import nn
from torch.nn import SiLU
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention

from ..utils import Device, name_contains_keys
from .builder import EVALUATORS


@EVALUATORS.register_module()
class MacsEvaluator:
    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    @torch.inference_mode()
    def evaluate(
        self,
        model: PreTrainedModel,
        sparsity: float,
        data_loader: DataLoader,
        device: Device,
        logger: logging.Logger,
        method_name: str,
    ) -> float:
        self.check_model_name(model, logger)

        if data_loader is not None:
            logger.warning("MacsEvaluator.evaluate: data_loader is not needed for this method. ")

        def construct_inputs(shape: Tuple[int]) -> Dict[str, torch.Tensor]:
            return {"input_ids": torch.ones(shape, dtype=torch.long, device=device)}

        logger.info(f"MacsCounter: Using input shape {(1, self.seq_len)}")

        # macs: e.g. "71800.51 GMac"
        macs, _ = get_model_complexity_info(
            model,
            input_res=(1, self.seq_len),
            input_constructor=construct_inputs,
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True,
            custom_modules_hooks={  # type: ignore
                LlamaSdpaAttention: self.llama_attn_counter_hook,
                LlamaRMSNorm: self.rmsnorm_flops_counter_hook,
                SiLU: pool_flops_counter_hook,
            },
        )
        logger.info(f"Method: {method_name}, Sparsity: {sparsity}, MACs: {macs}")
        macs_value = float(macs.split(" ")[0])
        return macs_value

    @classmethod
    def check_model_name(cls, model: PreTrainedModel, logger: logging.Logger) -> None:
        model_name = model.__class__.__name__
        lower_model_name = model_name.lower()
        if not name_contains_keys(lower_model_name, ["llama", "vicuna"]):
            logger.warning(
                f"MacsCounter does not support model: {model_name}. " "Check the custom hooks for counting the MACs."
            )

    @staticmethod
    def llama_attn_counter_hook(module: nn.Module, input: Any, output: Any) -> Any:
        # (1) Ignore past-key values
        # (2) Assume there is no attention mask
        # Input will be empty in some pytorch version. use output here since input.shape == output.shape
        flops = 0
        q_len = output[0].shape[1]
        linear_dim = output[0].shape[-1]
        num_heads = module.num_heads
        head_dim = module.head_dim

        rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
        # QK^T + softmax + AttentionV
        attention_flops = num_heads * (q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim)
        # 4 for q, k, v, o.
        linear_flops = 4 * (q_len * linear_dim * num_heads * head_dim)
        flops += rotary_flops + attention_flops + linear_flops
        # Here the __flops__ is actually MACs
        # https://github.com/sovrasov/flops-counter.pytorch/issues/40#issuecomment-625264888
        module.__flops__ += int(flops)

    @staticmethod
    def rmsnorm_flops_counter_hook(module: nn.Module, input: Any, output: Any) -> Any:
        input = input[0]
        batch_flops = np.prod(input.shape)
        batch_flops *= 2
        module.__flops__ += int(batch_flops)
