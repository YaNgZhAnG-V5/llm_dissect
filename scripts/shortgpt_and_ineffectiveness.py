import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
from dissect.models import build_model_and_tokenizer
from dissect.pruners import PRUNERS
from dissect.utils import get_cuda_visible_devices


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("config", help="Path to config file.")
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/prune_vicuna/", help="Working directory to save the output files."
    )
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
    set_random_seed(42)
    work_dir = args.work_dir
    sparsity_table_save_dir = osp.join(work_dir, "sparsity_tables")
    mmengine.mkdir_or_exist(work_dir)
    mmengine.mkdir_or_exist(sparsity_table_save_dir)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(work_dir, f"{time_stamp}.log"),
    )
    # Pre-process config
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir
    if cfg.test_dataset.split == "train":
        logger.warning("cfg.test_dataset.split is 'train'. Automatically override it to 'test'.")
        cfg.test_dataset["split"] = "test"
    if not cfg.test_dataset.use_label:
        logger.warning(
            "Testing models requires ground truth labels, but cfg.test_dataset.use_label: "
            f"{cfg.test_dataset.use_label}. This config value will be automatically set to True."
        )
        cfg.test_dataset.use_label = True
    logger.info("Using config:\n" + "=" * 60 + f"\n{cfg.pretty_text}\n" + "=" * 60)
    cfg.dump(osp.join(work_dir, f"{time_stamp}_{osp.splitext(osp.basename(cfg.filename))[0]}.yaml"))

    cuda_visible_devices = get_cuda_visible_devices()
    if len(cuda_visible_devices) > 1:
        logger.info(
            f"Running multi-gpu inference on GPUs: {cuda_visible_devices}. The argument: "
            f"--gpu-id {args.gpu_id} is automatically set to 0, indicating that the inference starts from "
            f"GPU 0."
        )
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    model.eval()

    dataset = build_dataset(cfg.test_dataset, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, **cfg.data_loader)

    bi_based_pruner = PRUNERS.build(
        cfg.pruner, default_args=dict(model=model, tokenizer=tokenizer, logger=logger, device=device)
    )
    pruned_layers_dict = dict()
    for sparsity in bi_based_pruner.sparsities:
        logger.info(f"Start pruning for sparsity: {sparsity}")
        pruned_layers = bi_based_pruner.analyze_model(data_loader, sparsity)
        pruned_layers_dict.update({sparsity: pruned_layers})
        bi_based_pruner.prune(pruned_layers=pruned_layers, work_dir=work_dir, sparsity=sparsity)
        bi_based_pruner.reset()

    mmengine.dump(pruned_layers_dict, osp.join(work_dir, "pruned_layers_table.yaml"))


if __name__ == "__main__":
    main()
