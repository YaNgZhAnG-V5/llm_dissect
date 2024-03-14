import logging
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from typing import Dict

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from datasets import load_dataset
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dissect.models import register_masking_hooks
from dissect.utils import Device


def parse_args():
    parser = ArgumentParser("Test Pruned MLP")
    # parser.add_argument("sparsities", nargs="+", type=float, help="A sequence of sparsities in range[0, 1].")
    parser.add_argument(
        "--pruning-mask-dir", "-p", default="workdirs/prune_bert/pruning_masks", help="Directory of the pruning masks."
    )
    # parser.add_argument("--ckpt", "-c", required=True, help="Path to checkpoint.")
    parser.add_argument("--prior-path", help="Path to the activation priors.")
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/debug/", help="Working directory to save the output files."
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")

    return parser.parse_args()


@torch.no_grad()
def test_model_acc(
    model: nn.Module,
    sparsity: float,
    data_loader: DataLoader,
    device: Device,
    logger: logging.Logger,
    method_name: str,
) -> None:
    num_correct = 0
    num_total = 0

    for data in alive_it(data_loader, total=len(data_loader), enrich_print=False):
        target = data.pop("label")
        input_tensor = data.pop("input_ids").to(device)
        pred = model(input_tensor, **data)["logits"].argmax(-1)
        num_correct += (pred == target).sum().item()
        num_total += target.shape[0]

    acc = num_correct / num_total
    logger.info(f"Method: {method_name}, sparsity: {sparsity:.2f}, accuracy: {acc:.4f}")


@torch.no_grad()
def baseline_magnitude_prune(
    model: nn.Module,
    sparsity: float,
    ori_state_dict: Dict,
) -> nn.Module:
    pruned_state_dict = deepcopy(ori_state_dict)
    all_weights = []
    all_numels = []
    for k, v in ori_state_dict.items():
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

    model.load_state_dict(pruned_state_dict)
    return model


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

    device = torch.device(f"cuda:{args.gpu_id}")

    imdb = load_dataset("imdb")
    small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    small_test_dataset.set_format("torch", device="cuda")
    tokenized_train = small_test_dataset.map(preprocess_function, batched=True, batch_size=16)
    tokenized_train = tokenized_train.remove_columns(column_names=["text"])
    test_loader = torch.utils.data.DataLoader(tokenized_train, batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("./workdirs/bert-imdb-finetuned/checkpoint-188").to(
        "cuda:0"
    )
    state_dict = model.state_dict()
    model.eval()

    prior_state_dict = torch.load(args.prior_path, map_location=device) if args.prior_path is not None else None

    test_model_acc(
        model=model, sparsity=0.0, data_loader=test_loader, device=device, logger=logger, method_name="Origin Model"
    )
    sparsities = [i / 20 for i in range(1, 20)]
    for sparsity in sparsities:
        mask_path = osp.join(args.pruning_mask_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth')

        # get mask ratio at each layer and the parameter prune rate
        mask_state_dict = torch.load(mask_path)
        total_weights, total_remain_weights = 0, 0
        previous_pruned_neuron = 0
        for k in sorted(mask_state_dict.keys()):
            v = mask_state_dict[k]
            assert v.ndim == 1, "mask should be one-dimensional."
            layer = model.get_submodule(k)

            #     # TODO maybe subject to change, right now assume linear layers
            #     prior_num_params = layer.in_features * layer.out_features
            #     remained_num_params = v.float().sum().item() * (layer.in_features - previous_pruned_neuron)
            #     previous_pruned_neuron = layer.out_features - v.float().sum().item()
            #     total_weights += prior_num_params
            #     total_remain_weights += remained_num_params
            logger.info(f"Layer: {k}, Sparsity: {(1 - v.float().mean()):.2f}")
        #     logger.info(f"Layer: {k}, weight Sparsity: {(1 - (remained_num_params / prior_num_params)):.2f}")
        # logger.info(f"Total Sparsity: {(1 - total_remain_weights / total_weights):.2f}")

        # register mask hooks and perform testing
        handle_dict = register_masking_hooks(model, mask_path, device=device, prior_state_dict=prior_state_dict)
        test_model_acc(
            model=model, sparsity=sparsity, data_loader=test_loader, device=device, logger=logger, method_name="Ours"
        )
        for k, v in handle_dict.items():
            v.remove()

        # try baseline pruning
        model = baseline_magnitude_prune(model, sparsity, ori_state_dict=state_dict)
        test_model_acc(
            model=model,
            sparsity=sparsity,
            data_loader=test_loader,
            device=device,
            logger=logger,
            method_name="Magnitude",
        )

        # # reload origin model state dict
        # model.load_state_dict(state_dict)


if __name__ == "__main__":
    main()
