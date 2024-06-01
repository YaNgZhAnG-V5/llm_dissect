from argparse import ArgumentParser
from time import perf_counter, sleep
from typing import Any, Dict, Tuple

import mmengine
import numpy as np
import torch
from ptflops import get_model_complexity_info
from ptflops.pytorch_ops import pool_flops_counter_hook
from torch import nn
from torch.nn import SiLU
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention

from dissect.datasets import build_dataset
from dissect.models import build_model_and_tokenizer


class RuntimeHook:
    def __init__(self) -> None:
        self.in_time = 0.0
        self.out_time = 0.0
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0

    def pre_hook(self, module, input):
        self.in_time = perf_counter()
        self.starter.record()

    def hook(self, module, input, output):
        self.out_time = perf_counter()
        self.ender.record()
        self.ender.synchronize()
        self.elapsed_time = self.starter.elapsed_time(self.ender)

    def __call__(self, pre: bool):
        if pre:
            return self.pre_hook
        else:
            return self.hook

    def get_inference_time(self):
        # return self.out_time - self.in_time
        return self.elapsed_time


def register_runtime_hook(module):
    hook = RuntimeHook()
    module.register_forward_pre_hook(hook(pre=True))
    module.register_forward_hook(hook(pre=False))
    return hook


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


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/llama3_70b_per_attn.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    device = torch.device(f"cuda:{args.gpu_id}")
    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    model.eval()
    for length in [64, 8192]:
        cfg.test_dataset.update({"max_length": length})
        dataset = build_dataset(cfg.test_dataset, tokenizer=tokenizer)
        data_loader = DataLoader(dataset, **cfg.data_loader)

        # measure macs
        def construct_inputs(shape: Tuple[int]) -> Dict[str, torch.Tensor]:
            return {"input_ids": torch.ones(shape, dtype=torch.long, device=device)}

        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                model,
                input_res=(1, length),
                input_constructor=construct_inputs,
                as_strings=True,
                print_per_layer_stat=True,
                verbose=True,
                custom_modules_hooks={  # type: ignore
                    LlamaSdpaAttention: llama_attn_counter_hook,
                    LlamaRMSNorm: rmsnorm_flops_counter_hook,
                    SiLU: pool_flops_counter_hook,
                },
            )

        # register runtime hooks for interested modules
        interest_modules = ["self_attn", "mlp", "overall"]
        hooks_dict = dict()
        for module in interest_modules:
            hooks_dict.update({module: []})
        hooks_dict.update({"overall": [register_runtime_hook(model)]})
        for name, module in model.named_modules():
            module_name = name.split(".")[-1]
            if module_name in interest_modules:
                hooks_dict[module_name].append(register_runtime_hook(module))

        with torch.no_grad():
            # run inference on one single batch
            data = BatchEncoding(next(iter(data_loader))).to(device)
            _ = model(**data)
            # warm up
            if length == 64:
                continue

        # get inference time for interested modules
        print(f"Context length: {length}")
        for module_name, hooks in hooks_dict.items():
            inference_time = 0.0
            for hook in hooks:
                inference_time += hook.get_inference_time()
            print(f"Total inference time for all {module_name}: {inference_time:.4f}")
        # pause for a while to give margin in time measurement
        sleep(10)


if __name__ == "__main__":
    main()
