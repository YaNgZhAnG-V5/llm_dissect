import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from typing import List

import mmengine
import torch
import torch.nn as nn
import torchvision.transforms as T
from alive_progress import alive_it
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from dissect.dissectors import ActivationExtractor, ForwardADExtractor
from dissect.models import MLP
from dissect.utils import Device


def parse_args():
    parser = ArgumentParser("Prune MLP")
    parser.add_argument("sparsities", nargs="+", type=float, help="A sequence of sparsities in range[0, 1].")
    parser.add_argument("ckpt", help="Path to checkpoint.")
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/debug/", help="Working directory to save the output files."
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")

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
    dissector = ForwardADExtractor(model)
    prior_extractor = ActivationExtractor(model)

    accum_forward_grads = defaultdict(float)
    all_priors = defaultdict(float)
    for batch_index, (image, target) in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
        image, target = image.to(device), target.to(device)
        forward_kwargs = dict(flatten_start_dim=1)

        forward_grads = dissector.forward_ad(image, forward_kwargs=forward_kwargs)
        priors = prior_extractor.extract_activations(image, forward_kwargs=forward_kwargs)

        for k, v in forward_grads.items():
            # avg over batch dim, accumulate over data loader (will be averaged later)
            accum_forward_grads[k] += v.abs().mean(0)
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
    flatten_forward_grads = torch.concat(flatten_forward_grads, dim=0)
    split_size = [v[1] for v in shape_info.values()]
    mask_state_dict = dict()

    for sparsity in sparsities:
        top_k = int(flatten_forward_grads.numel() * (1 - sparsity))
        _, top_k_inds = torch.topk(flatten_forward_grads, top_k, sorted=False, largest=True)
        binary_mask = torch.zeros_like(flatten_forward_grads, dtype=torch.bool)
        binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
        split_binary_masks = binary_mask.split(dim=-1, split_size=split_size)

        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: split_binary_masks[i].reshape(forward_grad_shape)})

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

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    test_set = MNIST("./data/", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2, pin_memory=True, shuffle=False)

    model = MLP([784, 1024, 1024, 512, 256, 10]).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    logger.info(f"Loaded checkpoint: {args.ckpt}")
    model.load_state_dict(state_dict)
    model.eval()

    forward_prune(model=model, sparsities=args.sparsities, work_dir=work_dir, data_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
