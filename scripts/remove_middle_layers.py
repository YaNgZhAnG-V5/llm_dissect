# This script removes the middle attention layers from the model.
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat

import mmengine
import torch
from tabulate import tabulate

from dissect.models import build_model_and_tokenizer
from dissect.pruners import middle_layer_pruning
from dissect.utils import get_target_layers_ordered


def parse_args():
    parser = ArgumentParser("Pruned middle layers")
    parser.add_argument("--config", default="./configs/llama3_8b.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument("--attn", "-a", action="store_true", help="prune attention layers.")
    parser.add_argument("--ffn", "-f", action="store_true", help="prune ffn layers.")
    parser.add_argument(
        "--start", "-st", type=int, default=-1, help="init layer of the pruning. If not set, start from last layer."
    )
    parser.add_argument(
        "--length", "-l", type=int, default=-1, help="maximal layers of the pruning. If not set, end at first layer."
    )
    parser.add_argument("--stride", "-s", type=int, default=2, help="stride of the pruning.")
    parser.add_argument(
        "--workdir", "-w", type=str, default="workdirs/remove_middle_attn_debug", help="Path to save the result."
    )
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
    assert (args.attn and not args.ffn) or (not args.attn and args.ffn), "only one of attn or ffn must be set."
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
    target_layers = get_target_layers_ordered(model, target_modules)
    tabulate_target_layers = tabulate([layer.split(".") for layer in target_layers])
    logger.info(f"target layers are {tabulate_target_layers}")

    continue_pruning = True
    expected_length = args.length if args.length != -1 else len(target_layers)
    length = 2  # start from removing 2 layers
    while continue_pruning:
        prune_module = "attn" if args.attn else "ffn"
        pruned_layers = middle_layer_pruning(
            target_layers=target_layers,
            start=args.start,
            length=length,
            pruned_module=prune_module,
        )

        # stop if no more layers to prune
        if pruned_layers is None or length > expected_length:
            continue_pruning = False
            continue

        # save pruned layers
        print(f"prune {len(pruned_layers)} layers, starting layer: {pruned_layers[0]}, last layer: {pruned_layers[-1]}")
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
