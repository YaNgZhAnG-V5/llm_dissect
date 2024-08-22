# This script removes the middle attention layers from the model.
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat
from typing import List

import mmengine
import torch
from tabulate import tabulate

from dissect.models import build_model_and_tokenizer


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/llama3_8b.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument("--length", "-l", type=int, default=2, help="init length of the pruning.")
    parser.add_argument("--stride", "-s", type=int, default=2, help="stride of the pruning.")
    parser.add_argument(
        "--workdir", "-w", type=str, default="workdirs/remove_middle_attn", help="Path to save the result."
    )
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


def middle_attn_layer_pruning(
    target_layers: List[str],
    length: int,
):
    """prune attention blocks in the middle."""
    assert length % 2 == 0, "length should be even."

    # remove all non attention layers
    not_target_layers = []
    for layer in target_layers:
        if "self_attn" not in layer:
            not_target_layers.append(layer)
    for layer in not_target_layers:
        target_layers.remove(layer)

    # stop if length is longer than the number of target layers
    if length > len(target_layers):
        return None

    # find middle layers
    middle_layer_idx = len(target_layers) // 2
    start_idx = middle_layer_idx - length // 2
    end_idx = middle_layer_idx + length // 2
    pruned_layers = target_layers[start_idx:end_idx]
    return pruned_layers


class OrderedLayerNameHook:
    """get orders of all the target layers via hook."""

    def __init__(self, target_layers: dict):
        self.target_layers = target_layers
        self.ordered_target_layers = []

    def __call__(self, module, input, output):
        self.ordered_target_layers.append(
            list(self.target_layers.keys())[list(self.target_layers.values()).index(module)]
        )


def get_target_layers(model: torch.nn.Module, target_modules: List[str]):
    target_layers = {}
    hooks = []
    hook_callback = OrderedLayerNameHook(target_layers)

    # get target layers
    for target_module in target_modules:
        for name, layer in model.named_modules():
            if target_module in name.split(".")[-1] and name not in target_layers:
                target_layers[name] = layer
                hooks.append(layer.register_forward_hook(hook_callback))
    with torch.no_grad():
        _ = model(model.dummy_inputs["input_ids"].to(model.device))

    # remove hooks
    while hooks:
        hooks.pop().remove()
    target_layers = hook_callback.ordered_target_layers
    return target_layers


def main():
    args = parse_args()
    if "mixtral" in args.config:
        target_modules = ["self_attn", "block_sparse_moe"]
    else:
        target_modules = ["o_proj", "down_proj"]
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    exist_warning = True if os.path.exists(args.workdir) else False
    mmengine.mkdir_or_exist(args.workdir)
    device = torch.device(f"cuda:{args.gpu_id}")
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(args.workdir, f"{time_stamp}.log"),
    )
    if exist_warning:
        logger.warning(f"workdir {args.workdir} already exists, consider save it in another place.")
    logger.info(f"Model: \n{pformat(cfg.model)}")
    logger.info(f"Prune dataset: \n{pformat(cfg.pruning_dataset)}")
    logger.info(f"Test dataset: \n{pformat(cfg.test_dataset)}")
    logger.info(f"Testing manager: \n{pformat(cfg.test_cfg.testing_manager)}")
    logger.info(f"Evaluator:\n{pformat(cfg.test_cfg.evaluator)}")
    logger.info(f"parsed arguments: \n{pformat(vars(args))}")
    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", [])
    if len(cuda_visible_devices) == 0:
        model.to(device).eval()
    else:
        model.eval()
    target_layers = get_target_layers(model, target_modules)
    tabulate_target_layers = tabulate([layer.split(".") for layer in target_layers])
    logger.info(f"target layers are {tabulate_target_layers}")

    continue_pruning = True
    length = args.length
    while continue_pruning:
        pruned_layers = middle_attn_layer_pruning(
            target_layers=target_layers,
            length=length,
        )

        # stop if no more layers to prune
        if pruned_layers is None:
            continue_pruning = False
            continue

        # save pruned layers
        mmengine.mkdir_or_exist(osp.join(args.workdir, "pruning_masks"))
        mask_state_dict = {
            pruned_layer: torch.zeros(args.layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers
        }
        assert len(mask_state_dict) == length
        string_ratio = "_".join(str(length).split("."))
        file_name = f"length_{string_ratio}_pruning_masks.pth"
        save_path = osp.join(args.workdir, "pruning_masks", file_name)
        torch.save(mask_state_dict, save_path)
        length += args.stride


if __name__ == "__main__":
    main()
