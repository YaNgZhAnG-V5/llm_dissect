import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import mmengine
import torch
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER


def parse_args():
    parser = ArgumentParser("Test pruned models with baseline pruning methods.")
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


# @torch.no_grad()
# def baseline_magnitude_prune(
#     model: nn.Module,
#     sparsity: float,
# ) -> nn.Module:
#     # TODO there is a bug in this implementation, split weight is defined over all modules (include embedding)
#     pruned_model = deepcopy(model)
#     pruned_state_dict = pruned_model.state_dict()
#     all_weights = []
#     all_numels = []
#     # pruned_state_dict is still original state dict at this moment
#     for k, v in pruned_state_dict.items():
#         all_weights.append(torch.flatten(v))
#         all_numels.append(v.numel())
#
#     all_weights = torch.concat(all_weights, 0)
#     abs_all_weights = all_weights.abs()
#     _, top_k_inds = torch.topk(abs_all_weights, int(all_weights.numel() * (1 - sparsity)))
#
#     # pruned_mask: 1 for setting weight to 0, 0 for keep original weight
#     pruned_mask = torch.ones_like(all_weights, dtype=torch.bool)
#     pruned_mask[top_k_inds] = 0
#     all_weights[pruned_mask] = 0
#
#     split_weights = all_weights.split(all_numels)
#     for i, (k, v) in enumerate(pruned_state_dict.items()):
#         pruned_state_dict[k] = split_weights[i].view(v.shape)
#
#     pruned_model.load_state_dict(pruned_state_dict)
#     return pruned_model


# @torch.no_grad()
# def baseline_wanda_prune(
#     model: nn.Module,
#     sparsity: float,
#     cfg: mmengine.Config,
# ) -> nn.Module:
#     """reproduce the wanda method, replace based on weight times input, on a per output basis"""
#     pruned_model = deepcopy(model)
#
#     all_inputs = torch.load(osp.join(cfg.pruning_dir, "inputs.pth"), map_location=model.device)
#     for k, v in pruned_state_dict.items():
#         # ignore not prunable parts
#         exclude_layers = ["embeddings", "classifier", "LayerNorm", "pooler"]
#         if name_contains_keys(k, exclude_layers):
#             continue
#         if "bias" in k:
#             continue
#         # weight: (output dim, input dim)
#         weight = v.abs()
#         # make input_norm: (input dim)
#         input_norm = all_inputs[k.replace(".weight", "")]
#         metric = weight * input_norm
#         _, sorted_idx = torch.sort(metric, dim=1)
#         pruned_idx = sorted_idx[:, : int(sorted_idx.shape[1] * sparsity)]
#         v.scatter_(dim=1, index=pruned_idx, src=torch.zeros_like(pruned_idx, dtype=v.dtype))
#         pruned_state_dict[k] = v
#
#     pruned_model.load_state_dict(pruned_state_dict)
#     return pruned_model


def main():
    args = parse_args()
    set_random_seed(42)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
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

    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"])
    performance = evaluator.evaluate(
        model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
    )
    dump_data_dict = [
        {"sparsity": 0.0, "performance": performance},
    ]

    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager, default_args={"device": device})
    for sparsity in cfg.test_cfg.sparsities:
        # deep-copy original model to avoid in-place changes.
        pruned_model = deepcopy(model)
        logger.info("Deep-copied original model.")

        # prepare the testing environment, e.g. attach masking hook etc.
        # always operate on pruned_model (e.g. deep-copy from original model)
        testing_manager.prepare_environment(
            model=pruned_model,
            sparsity=sparsity,
            device=device,
            exclude_layers=cfg.pruner.criterion.exclude_layers,
        )
        performance = evaluator.evaluate(
            model=pruned_model,
            sparsity=sparsity,
            data_loader=data_loader,
            device=device,
            logger=logger,
            method_name=cfg.get("method_name", "Baseline"),
        )
        # TODO: compute the sparsity by taking all excluded layers into account.
        dump_data_dict.append({"sparsity": sparsity, "performance": performance})
        testing_manager.clean_environment(model=pruned_model)

    mmengine.dump(dump_data_dict, osp.join(work_dir, "test_results.yaml"))


if __name__ == "__main__":
    main()
