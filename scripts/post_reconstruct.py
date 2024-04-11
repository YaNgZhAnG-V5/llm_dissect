import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import mmengine
import torch
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER, reconstruct_layer


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


def merge_and_sort_layer_indices(list_of_indices: List[List[int]]) -> List[int]:
    """This function does two things. First, it performs sanity check to the list of indices and check if two indices
    (List[int]) have overlapped layer index (int). Then, it merges the list of indices, and sort the merged list."""
    seen = set()
    for indices in list_of_indices:
        indices_set = set(indices)
        if len(indices_set) < len(indices):
            raise ValueError(f"layer_indices {indices} have overlapped elements.")
        # Check if any element in the current list is already in the 'seen' set
        if any(element in seen for element in indices):
            raise ValueError("In lr_options, two list layer_indices have overlapped layer indices")
        # Add the elements of the current list to 'seen'
        seen.update(indices)
    return sorted(seen)


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
    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)

    # TODO: remove hard code sparsity
    sparsity = 0.4
    mask_path = osp.join(
        cfg.pruning_dir, "pruning_masks", f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'
    )
    logger.info(f"Loaded mask from {mask_path}")

    # prepare the testing environment, e.g. attach masking hook etc.
    testing_manager.prepare_environment(
        model=model,
        mask_path=mask_path,
        device=device,
        prior_state_dict=prior_state_dict,
        in_place=cfg.test_cfg.in_place,
    )

    _, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, testing_manager.mask_state_dict
    )

    logger.info(f"Total parameter sparsity within considered layers: {sparsity_target_layers:.4f}")
    logger.info(f"Total parameter sparsity in model: {sparsity_whole_model:.4f}")

    performance = evaluator.evaluate(
        model=model,
        sparsity=sparsity,
        data_loader=data_loader,
        device=device,
        logger=logger,
        method_name="Ours",
    )

    layer_name_templates = cfg.reconstruct.layer_name_templates
    # sanity check of layer indices
    lr_options = cfg.reconstruct.lr_options
    # merge and sort all layer indices
    layer_indices = merge_and_sort_layer_indices([opt["layer_indices"] for opt in lr_options])
    all_layer_inds_and_names = [
        (index, template.format(index)) for template in layer_name_templates for index in layer_indices
    ]

    for layer_index, layer_name in all_layer_inds_and_names:
        # opt has two fields. 1. lr (float); 2. layer_indices: (List[float]).
        # We check in which option group the layer_index is, and then retrieve the corresponding lr.
        lr = None
        for opt in lr_options:
            if layer_index in opt["layer_indices"]:
                # convert to float because "1e-5" in yaml will be parsed as string.
                lr = float(opt["lr"])
                break
        if lr is None:
            raise ValueError(f"Did not find corresponding lr for layer_index: {layer_index}")

        model = reconstruct_layer(
            layer_in_out_dir=cfg.reconstruct.in_out_dir,
            layer_name=layer_name,
            lr=lr,
            model=model,
            device=device,
            logger=logger,
        )

        with torch.no_grad():
            # perform evaluation
            logger.info(f"Reconstruction of layer [{layer_name}] finished. Start evaluating the model.")
            performance = evaluator.evaluate(
                model=model,
                sparsity=sparsity,
                data_loader=data_loader,
                device=device,
                logger=logger,
                method_name=f"Ours-[{layer_name}]-reconstructed",
            )

    # model is reset only when the (iterative) reconstruction process is finished.
    if cfg.test_cfg.in_place:
        model = testing_manager.clean_environment_inplace(model_cfg=cfg.model, device=device)
    else:
        testing_manager.clean_environment_hook()


if __name__ == "__main__":
    main()
