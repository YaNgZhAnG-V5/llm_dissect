import logging
from typing import Tuple

import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from ..utils import Device
from .builder import EVALUATORS


class ModelTimer:
    def __init__(self, num_repetitions: int, warm_up_steps: int, device: Device):
        self.num_repetitions = num_repetitions
        self.warm_up_steps = warm_up_steps
        self.device = device
        self.timings = list()

    def warm_up(self, model, dummy_input):
        for _ in range(self.warm_up_steps):
            _ = model(**dummy_input)
        return True

    def __enter__(self):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.starter.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ender.record()
        # WAIT FOR GPU SYNC
        # torch.cuda.synchronize(self.device)
        self.ender.synchronize()
        curr_time = self.starter.elapsed_time(self.ender)
        self.timings.append(curr_time)

    def calc_time(self):
        timings_tensor = torch.tensor(self.timings)
        mean_syn = timings_tensor.mean().item()
        if timings_tensor.numel() <= 1:
            std_syn = 0.0
        else:
            std_syn = timings_tensor.std().item()

        # clear timing list
        self.timings.clear()
        return mean_syn, std_syn


@EVALUATORS.register_module()
class InferenceTime(nn.Module):
    def __init__(self, num_repetitions: int = 10, warmup_steps: int = 10):
        super().__init__()
        self.num_repetitions = num_repetitions
        self.warmup_steps = warmup_steps

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        sparsity: float,
        data_loader: DataLoader,
        device: Device,
        logger: logging.Logger,
        method_name: str,
    ) -> Tuple[float, float]:
        assert self.num_repetitions <= len(
            data_loader
        ), "Number of repetitions should be less than the number of batches."
        timer = ModelTimer(num_repetitions=self.num_repetitions, warm_up_steps=self.warmup_steps, device=device)
        warmed_up = False
        for data in alive_it(data_loader, total=self.num_repetitions, enrich_print=False):
            data = BatchEncoding(data).to(device)
            # GPU warm up
            if timer.warm_up_steps > 0 and not warmed_up:
                warmed_up = timer.warm_up(model, data)
            with timer:
                _ = model(**data)
            break
        mean_time, std_time = timer.calc_time()
        logger.info(f"Average model inference time in {len(data_loader)} runs: {mean_time:.4f}+/-{std_time:.4f}")
        return mean_time, std_time
