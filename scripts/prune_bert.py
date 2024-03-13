import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from typing import List

import mmengine
import torch
import torch.nn as nn
from alive_progress import alive_it
from datasets import load_dataset
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dissect.dissectors import ActivationExtractor, ForwardADExtractor, get_layers
from dissect.utils import Device


def parse_args():
    parser = ArgumentParser("Prune MLP")
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/prune_bert/", help="Working directory to save the output files."
    )
    parser.add_argument("--gpu-id", type=int, default=1, help="GPU ID.")

    return parser.parse_args()


def forward_prune(
    model: nn.Module,
    sparsities: List[float],
    data_loader: DataLoader,
    work_dir: str,
    device: Device,
):
    mask_save_dir = osp.join(work_dir, "pruning_masks")
    mmengine.mkdir_or_exist(mask_save_dir)
    all_layers = get_layers(model, return_dict=True)
    layers = {}
    keywords = ["embeddings", "LayerNorm", "classifier"]
    for name, layer in all_layers.items():
        if any(keyword in name for keyword in keywords):
            continue
        layers[name] = layer
    dissector = ForwardADExtractor(model, layers=layers, insert_layer=model.bert.embeddings)
    prior_extractor = ActivationExtractor(model)

    accum_forward_grads = defaultdict(float)
    all_priors = defaultdict(float)
    for batch_index, data in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
        input_tensor = data.pop("input_ids").to(device)

        forward_grads = dissector.forward_ad(input_tensor, forward_kwargs=data)
        priors = prior_extractor.extract_activations(input_tensor, forward_kwargs=data)

        for k, v in forward_grads.items():
            # avg over batch dim, accumulate over data loader (will be averaged later)
            v = v.abs()
            # TODO caution, this only works if the output neuron dim is the last dim
            v = v.mean(list(range(v.ndim - 1)))
            accum_forward_grads[k] += v
            all_priors[k] += priors[k].mean(0)

    for k, v in all_priors.items():
        all_priors[k] = v / len(data_loader)
    torch.save(all_priors, osp.join(work_dir, "priors.pth"))

    # shape info stores the output's shape and number of neurons
    shape_info = dict()
    flatten_forward_grads = []
    for k, v in accum_forward_grads.items():
        # compute the absolute values of the gradients
        avg_forward_grad = v / len(data_loader)
        flatten_forward_grads.append(avg_forward_grad.flatten())
        shape_info.update({k: (avg_forward_grad.shape, avg_forward_grad.numel())})

    # concatenate the flattened forward grads and record length of each chunk
    concat_forward_grads = torch.concat(flatten_forward_grads, dim=0)
    split_size = [v[1] for v in shape_info.values()]
    mask_state_dict = dict()

    for sparsity in sparsities:
        top_k = int(concat_forward_grads.numel() * (1 - sparsity))
        _, top_k_inds = torch.topk(concat_forward_grads, top_k, sorted=False, largest=True)
        binary_mask = torch.zeros_like(concat_forward_grads, dtype=torch.bool)
        binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
        split_binary_masks = binary_mask.split(dim=-1, split_size=split_size)

        # local binary masks
        local_binary_masks = []
        for forward_grad in flatten_forward_grads:
            top_k = int(forward_grad.numel() * (1 - sparsity))
            _, top_k_inds = torch.topk(forward_grad, top_k, sorted=False, largest=True)
            local_binary_mask = torch.zeros_like(forward_grad, dtype=torch.bool)
            local_binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
            local_binary_masks.append(local_binary_mask)

        final_binary_masks = []
        for global_binary_mask, local_binary_mask in zip(split_binary_masks, local_binary_masks):
            actual_sparsity = 1 - global_binary_mask.float().sum() / global_binary_mask.numel()
            # enforce per-layer actual sparsity is no greater than the specified sparsity
            if actual_sparsity > sparsity:
                final_binary_mask = local_binary_mask
            else:
                final_binary_mask = global_binary_mask
            final_binary_masks.append(final_binary_mask)

        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: final_binary_masks[i].reshape(forward_grad_shape)})

        torch.save(
            mask_state_dict, osp.join(mask_save_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth')
        )


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
    small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    small_train_dataset.set_format("torch", device=f"cuda:{args.gpu_id}")
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True, batch_size=16)
    tokenized_train = tokenized_train.remove_columns(column_names=["text", "label"])
    data_loader = torch.utils.data.DataLoader(tokenized_train, batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("./workdirs/bert-imdb-finetuned/checkpoint-188").to(
        f"cuda:{args.gpu_id}"
    )
    model.eval()

    forward_prune(
        model=model,
        sparsities=[i / 20 for i in range(1, 20)],
        work_dir=work_dir,
        data_loader=data_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
