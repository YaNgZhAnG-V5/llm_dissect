import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import Any

import mmengine
import torch
from mmengine.runner import set_random_seed
from tabulate import tabulate
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
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


def sanity_check_actual_sparsity(num_params: int, ori_total_param_count: int, testing_manager: Any) -> None:
    if hasattr(testing_manager, "in_place"):
        # assertions to make sure the pruning is working correctly
        if not testing_manager.in_place:
            assert (
                num_params / ori_total_param_count == 1.0
            ), "For pruning using mask, the actual pruning ratio should never change, check implementation."
        else:
            assert (
                num_params / ori_total_param_count != 1.0
            ), "In-place pruning should have non-zero actual sparsity, check implementation."


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
    cfg.dump(osp.join(work_dir, f"{osp.splitext(osp.basename(cfg.filename))[0]}_{time_stamp}.yaml"))

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
    ori_total_param_count = sum(p.numel() for p in model.parameters())
    ori_param_count_dict = {k: p.numel() for k, p in model.named_parameters()}
    logger.info(f"Total number of parameters in the original model: {ori_total_param_count}")
    logger.info(f"Using {cfg.test_dataset.dataset_name} dataset for test.")
    dataset = build_dataset(cfg.test_dataset, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, **cfg.data_loader)

    if cfg.test_cfg.use_prior:
        prior_state_dict = torch.load(osp.join(args.pruning_dir, "activations.pth"), map_location=device)
    else:
        prior_state_dict = None

    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"])
    runtime_evaluator = EVALUATORS.build(cfg.test_cfg["runtime_evaluator"])
    if cfg.test_cfg.get("eval_original", True):
        main_performance = evaluator.evaluate(
            model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
        )
        original_mean_time, _ = runtime_evaluator.evaluate(
            model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
        )
        dump_data_list = [
            {
                "desired\nsparsity": 0.0,
                "main\nperformance": main_performance,
                "mean\ntime": original_mean_time,
                "speedup": 1.0,
            },
        ]
    else:
        logger.warning('cfg.test_cfg.get("eval_original", True) is False. The original model will not be evaluated.')
        original_mean_time = None
        dump_data_list = []
    # Evaluate the zero-shot performance on various tasks
    if cfg.test_cfg.get("second_evaluator", None) is not None:
        if cfg.test_cfg.second_evaluator["type"] == "LMEvalHarness":
            default_args = {"tokenizer": tokenizer}
        else:
            default_args = None
        second_evaluator = EVALUATORS.build(cfg.test_cfg["second_evaluator"], default_args=default_args)
        if cfg.test_cfg.get("eval_original", True):
            second_eval_result = second_evaluator.evaluate(
                model=model, sparsity=0.0, data_loader=None, device=device, logger=logger, method_name="Original Model"
            )
            dump_data_list[0].update(second_eval_result)
    else:
        second_evaluator = None
    # Evaluate MACs
    if cfg.test_cfg.get("macs_evaluator", None) is not None:
        macs_evaluator = EVALUATORS.build(cfg.test_cfg["macs_evaluator"])
        total_macs = macs_evaluator.evaluate(
            model=model, sparsity=0.0, data_loader=None, device=device, logger=logger, method_name="Original Model"
        )
        dump_data_list[0].update({"total\nmacs": total_macs})
    else:
        macs_evaluator = None

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
            prior_state_dict=prior_state_dict,
        )

        # get mask ratio at each layer and the parameter prune rate
        log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
            model, testing_manager.mask_state_dict, ori_param_count_dict
        )
        num_params = sum(p.numel() for p in model.parameters())
        sanity_check_actual_sparsity(num_params, ori_total_param_count, testing_manager)
        logger.info(
            f"Total #params in pruned model: {num_params}, pruning ratio: {(1 - num_params / ori_total_param_count):2f}"
        )
        if cfg.test_cfg.print_table:
            sparsity_table = tabulate(log_tabulate, headers="keys", tablefmt="grid", floatfmt=".2f")
            mmengine.put_text(sparsity_table, osp.join(sparsity_table_save_dir, f"sparsity_{sparsity}.txt"))
            logger.info("Sparsity table:\n" f"{sparsity_table}")
        logger.info(f"Total parameter sparsity within considered layers: {sparsity_target_layers:.4f}")
        logger.info(f"Total parameter sparsity in model: {sparsity_whole_model:.4f}")

        # perform evaluation
        main_performance = evaluator.evaluate(
            model=model,
            sparsity=sparsity,
            data_loader=data_loader,
            device=device,
            logger=logger,
            method_name="Ours",
        )
        # TODO: Check why here the sparsity is 0.0 and why the method_name is "Original Model"
        mean_time, _ = runtime_evaluator.evaluate(
            model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
        )
        curr_result_dict = {
            "desired\nsparsity": sparsity,
            "sparsity within\nconsidered layers": sparsity_target_layers,
            "sparsity\nin model": sparsity_whole_model,
            "mean\ntime": mean_time,
            "main\nperformance": main_performance.item(),
        }
        if cfg.test_cfg.get("eval_original", True):
            curr_result_dict.update({"speedup": original_mean_time / mean_time})
        if second_evaluator is not None:
            # Zero-shot performance on various tasks. LMEvalHarness will load data, so no data_loader is needed
            second_eval_result = second_evaluator.evaluate(
                model=model, sparsity=sparsity, data_loader=None, device=device, logger=logger, method_name="Ours"
            )
            curr_result_dict.update(second_eval_result)
        if macs_evaluator is not None:
            total_macs = macs_evaluator.evaluate(
                model=model, sparsity=sparsity, data_loader=None, device=device, logger=logger, method_name="Ours"
            )
            curr_result_dict.update({"total\nmacs": total_macs})
        dump_data_list.append(curr_result_dict)

        model = testing_manager.clean_environment(model=model, model_cfg=cfg.model, device=device)
    logger.info("Evaluation finished.\n" f"{tabulate(dump_data_list, headers='keys', floatfmt='.4f', tablefmt='grid')}")
    mmengine.dump(dump_data_list, osp.join(work_dir, "test_results.yaml"))


if __name__ == "__main__":
    main()
