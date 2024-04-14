import logging

import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from ..utils import Device
from .builder import EVALUATORS


@EVALUATORS.register_module()
class Perplexity(nn.Module):

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
        ppls = []
        for data in alive_it(data_loader, total=len(data_loader), enrich_print=False):
            data = BatchEncoding(data).to(device)
            output = model(**data)
            ppls.append(torch.exp(torch.tensor([output.loss]).mean()).item())

        ppl = torch.tensor(ppls).mean()
        logger.info(f"Method: {method_name}, sparsity: {sparsity:.2f}, Perplexity: {ppl:.4f}")
        return ppl
