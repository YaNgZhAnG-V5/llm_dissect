import logging
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from datasets import load_dataset
from mmengine.runner import set_random_seed
from tabulate import tabulate
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding

from dissect.models import register_masking_hooks
from dissect.utils import Device


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_bert.yaml", help="Path to config file.")
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/debug/", help="Working directory to save the output files."
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


@torch.no_grad()
def test_model_acc(
    model: nn.Module,
    sparsity: float,
    data_loader: DataLoader,
    device: Device,
    logger: logging.Logger,
    method_name: str,
) -> float:
    num_correct = 0
    num_total = 0

    for data in alive_it(data_loader, total=len(data_loader), enrich_print=False):
        target = data.pop("label").to(device)
        batch = BatchEncoding(data).to(device)
        pred = model(**batch)["logits"].argmax(-1)
        num_correct += (pred == target).sum().item()
        num_total += target.shape[0]

    acc = num_correct / num_total
    logger.info(f"Method: {method_name}, sparsity: {sparsity:.2f}, accuracy: {acc:.4f}")
    return acc


@torch.no_grad()
def baseline_magnitude_prune(
    model: nn.Module,
    sparsity: float,
    cfg: mmengine.Config,
) -> nn.Module:
    # TODO there is a bug in this implementation, split weight is defined over all modules (include embedding)
    pruned_model = deepcopy(model)
    pruned_state_dict = pruned_model.state_dict()
    all_weights = []
    all_numels = []
    for k, v in pruned_state_dict.items():
        all_weights.append(torch.flatten(v))
        all_numels.append(v.numel())

    all_weights = torch.concat(all_weights, 0)
    abs_all_weights = all_weights.abs()
    _, top_k_inds = torch.topk(abs_all_weights, int(all_weights.numel() * (1 - sparsity)))

    # pruned_mask: 1 for setting weight to 0, 0 for keep original weight
    pruned_mask = torch.ones_like(all_weights, dtype=torch.bool)
    pruned_mask[top_k_inds] = 0
    all_weights[pruned_mask] = 0

    split_weights = all_weights.split(all_numels)
    for i, (k, v) in enumerate(pruned_state_dict.items()):
        pruned_state_dict[k] = split_weights[i].view(v.shape)

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model


@torch.no_grad()
def baseline_wanda_prune(
    model: nn.Module,
    sparsity: float,
    cfg: mmengine.Config,
) -> nn.Module:
    """reproduce the wanda method, replace based on weight times input, on a per output basis"""
    pruned_model = deepcopy(model)
    pruned_state_dict = pruned_model.state_dict()
    all_inputs = torch.load(osp.join(cfg.pruning_dir, "inputs.pth"), map_location=model.device)
    for k, v in pruned_state_dict.items():
        # ignore not prunable parts
        exclude_layers = ["embeddings", "classifier", "LayerNorm", "pooler"]
        if any(exclude_layer in k for exclude_layer in exclude_layers):
            continue
        if "bias" in k:
            continue
        # weight: (output dim, input dim)
        weight = v.abs()
        # make input_norm: (input dim)
        input_norm = all_inputs[k.replace(".weight", "")]
        metric = weight * input_norm
        _, sorted_idx = torch.sort(metric, dim=1)
        pruned_idx = sorted_idx[:, : int(sorted_idx.shape[1] * sparsity)]
        v.scatter_(dim=1, index=pruned_idx, src=torch.zeros_like(pruned_idx, dtype=v.dtype))
        pruned_state_dict[k] = v

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model


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

    # TODO: extend to more datasets and models
    imdb = load_dataset("imdb")
    test_set = imdb["test"].shuffle(seed=42).select(list(range(300)))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    test_set.set_format("torch")
    test_set = test_set.map(preprocess_function, batched=True).remove_columns(column_names=["text"])
    test_loader = DataLoader(test_set, **cfg.data_loader)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.ckpt_path).to(device)
    model.eval()

    if cfg.test_cfg.use_prior:
        prior_state_dict = torch.load(osp.join(cfg.pruning_dir, "activations.pth"), map_location=device)
    else:
        prior_state_dict = None

    test_acc = test_model_acc(
        model=model, sparsity=0.0, data_loader=test_loader, device=device, logger=logger, method_name="Origin Model"
    )
    dump_data_dict = [
        {"sparsity": 0.0, "accuracy": test_acc},
    ]

    for sparsity in cfg.test_cfg.sparsities:
        mask_path = osp.join(
            cfg.pruning_dir, "pruning_masks", f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'
        )

        # get mask ratio at each layer and the parameter prune rate
        mask_state_dict = torch.load(mask_path)
        exclude_layers = cfg.pruner.criterion.exclude_layers

        log_tabulate = []
        for k in sorted(mask_state_dict.keys()):
            # if the layer name contains any one of the exclude_layers, skip the layer
            if any(exclude_layer in k for exclude_layer in exclude_layers):
                continue
            v = mask_state_dict[k]
            assert v.ndim == 1, "mask should be one-dimensional."
            # TODO: calibrate the weight sparsity per layer
            log_tabulate.append({"layer": k, "neuron_sparsity": 1 - v.float().mean().item()})

        logger.info("Sparsity table:\n" f"{tabulate(log_tabulate, headers='keys', tablefmt='grid', floatfmt='.2f')}")

        # register mask hooks and perform testing
        handle_dict = register_masking_hooks(
            model, mask_path, device=device, exclude_layers=exclude_layers, prior_state_dict=prior_state_dict
        )

        test_acc = test_model_acc(
            model=model, sparsity=sparsity, data_loader=test_loader, device=device, logger=logger, method_name="Ours"
        )
        dump_data_dict.append({"sparsity": sparsity, "accuracy": test_acc, "layer_stats": log_tabulate})
        for k, v in handle_dict.items():
            v.remove()

        # magnitude pruning as baseline
        pruned_model = baseline_magnitude_prune(model, sparsity, cfg=cfg)
        _ = test_model_acc(
            model=pruned_model,
            sparsity=sparsity,
            data_loader=test_loader,
            device=device,
            logger=logger,
            method_name="Magnitude",
        )

        # magnitude pruning as baseline
        pruned_model = baseline_wanda_prune(model, sparsity, cfg=cfg)
        _ = test_model_acc(
            model=pruned_model,
            sparsity=sparsity,
            data_loader=test_loader,
            device=device,
            logger=logger,
            method_name="wanda",
        )

    mmengine.dump(dump_data_dict, osp.join(work_dir, "test_results.yaml"))


if __name__ == "__main__":
    main()
