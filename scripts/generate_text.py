import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
from datasets import Dataset
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader

from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER
from dissect.utils import get_cuda_visible_devices


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_vicuna.yaml", help="Path to config file.")
    parser.add_argument(
        "--pruning-dir",
        "-p",
        required=True,
        help="Directory where the pruning results were stored. "
        'It should contain a sub-directory "pruning_masks/" storing the pruning masks.',
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
    cfg.pruning_dir = args.pruning_dir
    cfg.work_dir = args.work_dir
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

    test_set = Dataset.from_dict({"input_prompt": cfg.input_prompt})
    test_set.set_format(type="torch", columns=["input_prompt"])
    data_loader = DataLoader(test_set, **cfg.data_loader)

    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"], default_args={"tokenizer": tokenizer})
    if cfg.test_cfg.get("eval_original", True):
        main_performance = evaluator.evaluate(
            model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
        )
    else:
        logger.warning('cfg.test_cfg.get("eval_original", True) is False. The original model will not be evaluated.')

    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)
    # perform evaluation on pruned models
    for sparsity in cfg.test_cfg.sparsities:
        mask_path = osp.join(
            args.pruning_dir, "pruning_masks", f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'
        )
        logger.info(f"Loading mask from {mask_path}")

        # prepare the testing environment, e.g. attach masking hook etc.
        model = testing_manager.prepare_environment(
            model=model,
            model_cfg=cfg.model,
            mask_path=mask_path,
            device=device,
            prior_state_dict=None,
        )
        # perform evaluation
        main_performance = evaluator.evaluate(
            model=model,
            sparsity=sparsity,
            data_loader=data_loader,
            device=device,
            logger=logger,
            method_name="Ours",
        )

        model = testing_manager.clean_environment(model=model, model_cfg=cfg.model, device=device)


if __name__ == "__main__":
    main()
