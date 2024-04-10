import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.runner import set_random_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dissect.datasets import LayerInOutDataset, build_dataset
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
    performance = evaluator.evaluate(
        model=model, sparsity=0.0, data_loader=data_loader, device=device, logger=logger, method_name="Origin Model"
    )

    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)

    # TODO: remove hard code sparsity
    sparsity = 0.25
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

    _, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, testing_manager.mask_state_dict
    )

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

    if cfg.test_cfg.in_place:
        model = testing_manager.clean_environment_inplace(model_cfg=cfg.model, device=device)
    else:
        testing_manager.clean_environment_hook()

    # TODO: remove hard code layer_name
    layer_name = "model.layers.0.self_attn"
    dataset_root = osp.join(cfg.layer_in_out_dir, layer_name)
    dataset = LayerInOutDataset(dataset_root)
    indices = np.arange(len(dataset))
    train_size = int(len(dataset) * 0.75)
    train_set = Subset(dataset, indices[:train_size])
    val_set = Subset(dataset, indices[train_size:])
    logger.info(f"Post reconstruction train set size: {len(train_set)}, val set size: {len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=4, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, num_workers=4, shuffle=False)

    target_layer = model.get_submodule(layer_name)
    optimizer = AdamW(target_layer.parameters(), lr=5e-5, weight_decay=5e-5)

    num_epochs = 3
    for epoch_index in range(num_epochs):
        for batch_index, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            targets = targets.to(device)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
                preds = target_layer(inputs)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                preds = target_layer(**inputs)
            else:
                raise TypeError(f"Invalid inputs type: {type(inputs)}")

            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

            if batch_index % 2 == 0:
                logger.info(
                    f"Epoch [{epoch_index+1}/{num_epochs}] Batch [{batch_index+1}/{len(train_loader)}]: "
                    f"training error: {loss:.5f}"
                )

        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(val_loader):
                targets = targets.to(device)
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                    preds = target_layer(inputs)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    preds = target_layer(**inputs)
                else:
                    raise TypeError(f"Invalid inputs type: {type(inputs)}")

                loss = F.mse_loss(preds, targets)
                logger.info(f"Epoch [{epoch_index+1}/{num_epochs}]: validation error: {loss:.5f}")

        with torch.no_grad():
            # perform evaluation
            performance = evaluator.evaluate(
                model=model,
                sparsity=sparsity,
                data_loader=data_loader,
                device=device,
                logger=logger,
                method_name="Ours",
            )


if __name__ == "__main__":
    main()
