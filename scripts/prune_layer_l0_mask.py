# prune_layer_l0_mask.py
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat
from typing import List

import mmengine
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from alive_progress import alive_it
from torch import nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.models.l0_mask import L0Mask
from dissect.pruners import TESTING_MANAGER
from dissect.pruners.mask_optimizer import MaskOptimizer
from dissect.utils import Device, get_target_layers_no_exclude, suppress_output, suppress_tqdm


def parse_args():
    parser = ArgumentParser("Optimize masks for layer pruning using L0Mask and Lagrangian Optimization")
    parser.add_argument("--config", default="./configs/prune_llama.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="Layer dimension in the model.")
    parser.add_argument(
        "--workdir", "-w", type=str, default="workdirs/layer_prune_mask", help="Path to save the result."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override config entries with format xxx=yyy or xxx.zzz=qqq=yyy.",
    )
    # Hyperparameters for mask optimization
    parser.add_argument("--lr-mask", type=float, default=1e-3, help="Learning rate for the mask optimizer.")
    parser.add_argument(
        "--lr-mu", type=float, default=1e-3, help="Learning rate for the Lagrangian multipliers optimizer."
    )
    parser.add_argument("--gamma", type=float, default=5e-7, help="Gamma for Lagrangian's first-order term.")
    parser.add_argument("--eta", type=float, default=5e-7, help="Eta for Lagrangian's second-order term.")
    parser.add_argument("--alpha", "-a", type=float, default=5e-7, help="Alpha for sparsity regularization.")
    parser.add_argument("--beta", "-b", type=float, default=1e-8, help="Beta for polar regularization.")
    parser.add_argument("--num-iterations", "-it", type=int, default=500, help="Number of optimization iterations.")
    parser.add_argument("--use-lagrangian", action="store_true", help="Enable Lagrangian multiplier optimization.")
    parser.add_argument(
        "--use-lagrangian-proxy", action="store_true", help="Enable Lagrangian loss with regular optimization."
    )
    parser.add_argument("--target-sparsity", type=float, default=0.25, help="Target sparsity level for pruning (0-1).")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory."
    )
    parser.add_argument("--warm-up", type=int, default=0, help="Number of iterations before using regularization.")
    # New argument for skipping layers
    parser.add_argument(
        "--skip-layers",
        type=int,
        default=0,
        help="Number of initial layers to skip from optimization and pruning.",
    )
    # New argument for weighting language modeling loss
    parser.add_argument(
        "--weight-lm-loss",
        type=float,
        default=1e-6,
        help="Weight for the language modeling loss in the combined distance metric.",
    )
    return parser.parse_args()


def collect_output_data(data_loader: DataLoader, model: nn.Module, device: torch.device, logger) -> torch.Tensor:
    """
    Collects output logits from the original model for comparison.

    Parameters:
    -----------
    - data_loader: DataLoader for the pruning dataset.
    - model: The original model.
    - device: Device to perform computations on.
    - logger: Logger for logging information.

    Returns:
    --------
    - Tensor containing all collected output logits.
    """
    logger.info("Collecting output data for comparison...")
    outputs = []
    with torch.no_grad():
        for data in alive_it(data_loader, total=len(data_loader), enrich_print=False):
            data = BatchEncoding(data).to(device)
            output = model(**data)
            outputs.append(output.logits.detach().cpu())
    outputs = torch.cat(outputs, dim=0).to('cpu')
    logger.info("Finished collecting output data.")
    return outputs


def main():
    args = parse_args()

    # Load configuration
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Ensure that use_label is enabled for pruning and test datasets
    cfg.pruning_dataset["use_label"] = True
    cfg.test_dataset["use_label"] = True

    # Setup work directory and logger
    exist_warning = os.path.exists(args.workdir)
    mmengine.mkdir_or_exist(args.workdir)
    if exist_warning:
        print(f"Warning: workdir {args.workdir} already exists, consider saving in another location.")
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    log_file = osp.join(args.workdir, f"{time_stamp}.log")
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=log_file,
    )

    # Log initial configuration
    logger.info(f"Model Configuration:\n{pformat(cfg.model)}")
    logger.info(f"Pruning Dataset Configuration:\n{pformat(cfg.pruning_dataset)}")
    logger.info(f"Test Dataset Configuration:\n{pformat(cfg.test_dataset)}")
    logger.info(f"Testing Manager Configuration:\n{pformat(cfg.test_cfg.testing_manager)}")
    logger.info(f"Evaluator Configuration:\n{pformat(cfg.test_cfg.evaluator)}")
    logger.info(f"Parsed Arguments:\n{pformat(vars(args))}")

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg.model, device=f"cuda:{args.gpu_id}")
    model.eval()

    # Build datasets and dataloaders
    prune_dataset = build_dataset(cfg.pruning_dataset, tokenizer=tokenizer)
    prune_data_loader = DataLoader(prune_dataset, **cfg.data_loader)

    # Collect original outputs
    outputs = collect_output_data(
        data_loader=prune_data_loader, model=model, device=torch.device(f"cuda:{args.gpu_id}"), logger=logger
    )

    # Identify target layers
    target_modules = ["o_proj", "down_proj"]  # Example target modules; adjust as needed
    target_layers = get_target_layers_no_exclude(model, target_modules)
    logger.info(f"Identified {len(target_layers)} target layers for pruning.")

    # Determine layers to skip
    num_skip = args.skip_layers
    if num_skip > len(target_layers):
        logger.warning(
            f"Requested to skip {num_skip} layers, but only {len(target_layers)} target layers available. Skipping all target layers."
        )
        num_skip = len(target_layers)
    skipped_layers = target_layers[:num_skip]
    optimized_layers = target_layers[num_skip:]
    logger.info(f"Skipping the first {num_skip} layers: {skipped_layers}")
    logger.info(f"Optimizing the remaining {len(optimized_layers)} layers.")

    # Initialize Mask Optimizer
    mask_optimizer = MaskOptimizer(
        model=model,
        target_modules=target_modules,
        outputs=outputs,
        data_loader=prune_data_loader,
        distance_metric=cfg.test_cfg.evaluator.get("distance_metric", "js_divergence"),
        num_iterations=args.num_iterations,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
        lr_mask=args.lr_mask,
        lr_mu=args.lr_mu,
        device=torch.device(f"cuda:{args.gpu_id}"),
        logger=logger,
        lamb_init="random",
        target_sparsity=args.target_sparsity,
        use_lagrangian=args.use_lagrangian,
        use_lagrangian_proxy=args.use_lagrangian_proxy,
        gradient_checkpointing=args.gradient_checkpointing,
        warm_up=args.warm_up,
        skipped_layers=skipped_layers,  # Pass skipped layers
        weight_lm_loss=args.weight_lm_loss,  # Pass the weight for LM loss
    )

    # Optimize masks
    mask_optimizer.optimize_masks()

    # Retrieve pruned layers
    binary_mask = mask_optimizer.get_binary_mask()
    logger.info(f"Kept Layers: {binary_mask}")

    target_layers = get_target_layers_no_exclude(model, target_modules)
    assert len(binary_mask) == len(target_layers), "Mask should have the same length as the target layers."
    pruned_layers = [layer for layer, m in zip(target_layers, binary_mask) if m == 0]

    # Save pruning masks
    mmengine.mkdir_or_exist(osp.join(args.workdir, "pruning_masks"))
    mask_state_dict = {pruned_layer: torch.zeros(args.layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers[:]}
    file_name = f"{time_stamp}_pruning_masks.pth"
    save_path = osp.join(args.workdir, "pruning_masks", file_name)
    torch.save(mask_state_dict, save_path)
    logger.info(f"Pruning masks saved to {save_path}")


if __name__ == "__main__":
    main()
