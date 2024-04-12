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


def parse_args():
    parser = ArgumentParser("Analyze and prune model.")
    parser.add_argument("--config", default="./configs/prune_vicuna.yaml", help="Path to yaml config file.")
    parser.add_argument(
        "--prev-result-dir",
        "-p",
        help="Directory of previous analysis result. If it is given, the analysis step will be skipped, "
        "and only pruning step is performed.",
    )
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
    mmengine.mkdir_or_exist(work_dir)

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'),
    )
    logger.info("Using config:\n" + "=" * 60 + f"\n{cfg.pretty_text}\n" + "=" * 60)
    device = torch.device(f"cuda:{args.gpu_id}")

    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    model.eval()

    logger.info(f"Using {cfg.pruning_dataset.dataset_name} dataset for pruning.")
    dataset = build_dataset(cfg.pruning_dataset, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, **cfg.data_loader)

    pruner = PRUNERS.build(cfg.pruner, default_args={"model": model})
    if args.prev_result_dir is not None:
        analysis_result = pruner.load_analysis_result(args.prev_result_dir, device=device)
        logger.info(f"Loaded analysis result from {args.prev_result_dir}")
    else:
        analysis_result = pruner.analyze_model(data_loader=data_loader, work_dir=work_dir, device=device, logger=logger)

    pruner.prune(analyze_result=analysis_result, work_dir=work_dir, logger=logger)


if __name__ == "__main__":
    main()
