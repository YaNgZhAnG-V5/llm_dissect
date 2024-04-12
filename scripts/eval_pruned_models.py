import os.path as osp
from argparse import ArgumentParser
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
    ori_num_params = sum(p.numel() for p in model.parameters())
    ori_model_params_dict = {k: p.numel() for k, p in model.named_parameters()}
    logger.info(f"Total number of parameters in the original model: {ori_num_params}")

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

        # prepare the testing environment, e.g. attach masking hook etc.
        testing_manager.prepare_environment(
            model=model,
            mask_path=mask_path,
            device=device,
            prior_state_dict=prior_state_dict,
            in_place=cfg.test_cfg.in_place,
        )

        # get mask ratio at each layer and the parameter prune rate
        log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
            model, testing_manager.mask_state_dict, ori_model_params_dict, cfg.test_cfg.in_place
        )
        num_params = sum(p.numel() for p in model.parameters())

        # assertions to make sure the pruning is working correctly
        if not cfg.test_cfg.in_place:
            assert (
                num_params / ori_num_params
            ) == 1.0, "For pruning using mask, the actual pruning ratio should never change, check implementation."
        else:
            assert (
                num_params / ori_num_params != 1.0
            ), "In-place pruning should have non-zero actual sparsity, check implementation."

        # log parameter information
        logger.info(
            f"Total number of parameters in the pruned model: {num_params}, "
            f"pruned ratio: {(1 - num_params / ori_num_params):2f}"
        )
        if cfg.test_cfg.print_table:
            sparsity_table = tabulate(log_tabulate, headers="keys", tablefmt="grid", floatfmt=".2f")
            mmengine.put_text(sparsity_table, osp.join(sparsity_table_save_dir, f"sparsity_{sparsity}.txt"))
            logger.info("Sparsity table:\n" f"{sparsity_table}")
        logger.info(f"Total parameter sparsity within considered layers: {sparsity_target_layers:.4f}")
        logger.info(f"Total parameter sparsity in model: {sparsity_whole_model:.4f}")

        # perform evaluation
        performance = evaluator.evaluate(
            model=model,
            sparsity=sparsity,
            data_loader=data_loader,
            device=device,
            logger=logger,
            method_name="Ours",
        )
        dump_data_dict.append(
            {
                "desired\n sparsity": sparsity,
                "performance": performance.item(),
                "sparsity within\n considered layers": sparsity_target_layers,
                "sparsity\n in model": sparsity_whole_model,
            }
        )
        if cfg.test_cfg.in_place:
            model = testing_manager.clean_environment_inplace(model_cfg=cfg.model, device=device)
        else:
            testing_manager.clean_environment_hook()
    logger.info("Evaluation finished.\n" f"{tabulate(dump_data_dict, headers='keys', floatfmt='.4f')}")
    mmengine.dump(dump_data_dict, osp.join(work_dir, "test_results.yaml"))


if __name__ == "__main__":
    main()
