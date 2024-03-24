import logging

import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from ..utils import Device
from .builder import EVALUATORS


@EVALUATORS.register_module()
class Accuracy:

    @torch.no_grad()
    def evaluate(
        self,
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
