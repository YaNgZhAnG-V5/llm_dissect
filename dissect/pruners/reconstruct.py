import logging
import os.path as osp
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ..datasets import LayerInOutDataset
from ..utils import Device


def layer_forward_fn(
    target_layer: nn.Module, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], targets: torch.Tensor, device: Device
) -> torch.Tensor:
    targets = targets.to(device)
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
        preds = target_layer(inputs)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        preds = target_layer(**inputs)
    else:
        raise TypeError(f"Invalid inputs type: {type(inputs)}")

    if isinstance(preds, tuple):
        preds = preds[0]
    elif isinstance(preds, torch.Tensor):
        preds = preds
    else:
        raise TypeError(f"Invalid type of target layer output: {type(preds)}")

    loss = F.mse_loss(preds, targets)
    return loss


def reconstruct_layer(
    layer_in_out_dir: str,
    layer_name: str,
    lr: float,
    num_epochs: int,
    model: nn.Module,
    device: Device,
    logger: logging.Logger,
) -> nn.Module:
    dataset_root = osp.join(layer_in_out_dir, layer_name)
    dataset = LayerInOutDataset(dataset_root)
    rng = np.random.default_rng()
    indices = rng.permutation(len(dataset))
    train_size = int(len(dataset) * 0.75)
    train_set = Subset(dataset, indices[:train_size])
    val_set = Subset(dataset, indices[train_size:])
    logger.info(f"Dataset in {dataset_root}:  train set size: " f"{len(train_set)}, val set size: {len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=16, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=4, shuffle=False)

    target_layer = model.get_submodule(layer_name)
    optimizer = AdamW(target_layer.parameters(), lr=lr, weight_decay=1e-6)

    logger.info(f"Start reconstructing layer [{layer_name}]")
    for epoch_index in range(num_epochs):
        for batch_index, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            loss = layer_forward_fn(target_layer, inputs, targets, device)
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                logger.info(
                    f"Epoch [{epoch_index + 1}/{num_epochs}] Batch [{batch_index + 1}/{len(train_loader)}]: "
                    f"lr:{optimizer.param_groups[0]['lr']:.6f}, training error: {loss:.5f}"
                )

        with torch.no_grad():
            total_val_loss = 0.0
            for batch_index, (inputs, targets) in enumerate(val_loader):
                loss = layer_forward_fn(target_layer, inputs, targets, device)
                total_val_loss += loss

            avg_val_loss = total_val_loss / len(val_loader)
            logger.info(f"Epoch [{epoch_index + 1}/{num_epochs}]: validation error: {avg_val_loss:.5f}")

    return model
