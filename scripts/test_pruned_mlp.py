import logging
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from typing import Dict

import mmengine
import torch
import torch.nn as nn
import torchvision.transforms as T
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from alive_progress import alive_it

from dissect.models import MLP, register_masking_hooks
from dissect.utils import Device


def parse_args():
    parser = ArgumentParser('Test Pruned MLP')
    parser.add_argument('sparsities', nargs='+', type=float, help='A sequence of sparsities in range[0, 1].')
    parser.add_argument('--pruning-mask-dir', '-p', required=True, help='Directory of the pruning masks.')
    parser.add_argument('--ckpt', '-c', required=True, help='Path to checkpoint.')
    parser.add_argument('--prior-path', help='Path to the activation priors.')
    parser.add_argument(
        '--work-dir', '-w', default='workdirs/debug/', help='Working directory to save the output files.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')

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

    for image, target in alive_it(data_loader, total=len(data_loader), enrich_print=False):
        image, target = image.to(device), target.to(device)
        pred = model(image, flatten_start_dim=1).argmax(-1)
        num_correct += (pred == target).sum().item()
        num_total += target.shape[0]

    acc = num_correct / num_total
    logger.info(f'Method: {method_name}, sparsity: {sparsity:.2f}, accuracy: {acc:.4f}')


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
        name='dissect',
        logger_name='dissect',
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'))

    device = torch.device(f'cuda:{args.gpu_id}')

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    test_set = MNIST('./data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2, pin_memory=True, shuffle=False)

    model = MLP([784, 1024, 1024, 512, 256, 10]).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    logger.info(f'Loaded checkpoint: {args.ckpt}')
    model.load_state_dict(state_dict)
    model.eval()

    prior_state_dict = torch.load(args.prior_path, map_location=device) if args.prior_path is not None else None

    test_model_acc(
        model=model, sparsity=0.0, data_loader=test_loader, device=device, logger=logger, method_name='Origin Model')

    for sparsity in args.sparsities:
        mask_path = osp.join(
            args.pruning_mask_dir, f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth')
        handle_dict = register_masking_hooks(model, mask_path, device=device, prior_state_dict=prior_state_dict)
        test_model_acc(
            model=model, sparsity=sparsity, data_loader=test_loader, device=device, logger=logger, method_name='Ours')
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
            method_name='Magnitude')

        # reload origin model state dict
        model.load_state_dict(state_dict)


if __name__ == '__main__':
    main()
