import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import mmengine
import torch
from mmengine.runner import set_random_seed
from tabulate import tabulate
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER
from dissect.utils import calc_pruned_parameters


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_vicuna.yaml", help="Path to config file.")
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
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'),
    )
    # Pre-process config
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.dataset.split == "train":
        logger.warning("cfg.dataset.split is 'train'. Automatically override it to 'test'.")
        cfg.dataset["split"] = "test"
    if not cfg.dataset.use_label:
        logger.warning(
            "Testing models requires ground truth labels, but cfg.dataset.use_label: "
            f"{cfg.dataset.use_label}. This config value will be automatically set to True."
        )
        cfg.dataset.use_label = True

    logger.info("Using config:\n" + "=" * 60 + f"\n{cfg.pretty_text}\n" + "=" * 60)
    device = torch.device(f"cuda:{args.gpu_id}")

    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    model.eval()

    dataset = build_dataset(cfg.dataset, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, **cfg.data_loader)

    if cfg.test_cfg.use_prior:
        prior_state_dict = torch.load(osp.join(cfg.pruning_dir, "activations.pth"), map_location=device)
    else:
        prior_state_dict = None

    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"])
    performance = evaluator.evaluate(
        model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
    )
    dump_data_dict = [
        {"sparsity": 0.0, "performance": performance},
    ]

    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)

    # perform evaluation on pruned models
    for sparsity in cfg.test_cfg.sparsities:
        mask_path = osp.join(
            cfg.pruning_dir, "pruning_masks", f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'
        )
        logger.info(f"Loading mask from {mask_path}")
        logger.info("Deep-copied original model.")

        # get mask ratio at each layer and the parameter prune rate
        mask_state_dict = torch.load(mask_path)
        log_tabulate, sparsity_target_layers, sparsity_whole_model = calc_pruned_parameters(model, mask_state_dict)
        sparsity_table = tabulate(log_tabulate, headers="keys", tablefmt="grid", floatfmt=".2f")
        mmengine.put_text(sparsity_table, osp.join(sparsity_table_save_dir, f"sparsity_{sparsity}.txt"))
        if cfg.test_cfg.print_table:
            logger.info("Sparsity table:\n" f"{sparsity_table}")
        logger.info(f"Total parameter sparsity within considered layers: {sparsity_target_layers:.4f}")
        logger.info(f"Total parameter sparsity in model: {sparsity_whole_model:.4f}")

        # prepare the testing environment, e.g. attach masking hook etc.
        # always operate on pruned_model (e.g. deep-copy from original model)
        # deep-copy original model to avoid in-place changes.
        pruned_model = deepcopy(model)
        exclude_layers = cfg.pruner.criterion.exclude_layers
        testing_manager.prepare_environment(
            model=pruned_model,
            mask_path=mask_path,
            device=device,
            exclude_layers=exclude_layers,
            prior_state_dict=prior_state_dict,
        )
        performance = evaluator.evaluate(
            model=pruned_model,
            sparsity=sparsity,
            data_loader=data_loader,
            device=device,
            logger=logger,
            method_name="Ours",
        )
        dump_data_dict.append({"sparsity": sparsity, "performance": performance, "layer_stats": log_tabulate})
        testing_manager.clean_environment(model=pruned_model)

    mmengine.dump(dump_data_dict, osp.join(work_dir, "test_results.yaml"))


if __name__ == "__main__":
    main()
