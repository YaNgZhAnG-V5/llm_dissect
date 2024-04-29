from argparse import ArgumentParser
from time import perf_counter, sleep

import mmengine
import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding

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


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_vicuna.yaml", help="Path to config file.")
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
    model.to(device).eval()
    for length in [64, 128, 256, 512, 1024, 2048, 4096]:
        cfg.test_dataset.update({"max_length": length})
        dataset = build_dataset(cfg.test_dataset, tokenizer=tokenizer)
        data_loader = DataLoader(dataset, **cfg.data_loader)

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
