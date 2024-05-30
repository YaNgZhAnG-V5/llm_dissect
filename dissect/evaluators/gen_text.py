import logging

import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader

from ..utils import Device
from .builder import EVALUATORS


@EVALUATORS.register_module()
class GenTextEvaluator(nn.Module):

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        sparsity: float,
        data_loader: DataLoader,
        device: Device,
        logger: logging.Logger,
        method_name: str,
        verbose: bool = True,
    ) -> float:

        for data in alive_it(data_loader, total=len(data_loader), enrich_print=False, disable=not verbose):
            input_prompt = data["input_prompt"]
            inputs = self.tokenizer(input_prompt, return_tensors="pt").to(device)
            pred = model.generate(**inputs)
            decoded_pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
            logger.info(f"Sparsity: {sparsity}\n[Input prompt]: {input_prompt[0]}\n[Generated text]: {decoded_pred}")

        return 0
