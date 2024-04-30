import logging

import torch
import torch.nn as nn
from alive_progress import alive_it
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BatchEncoding

from ..utils import Device
from .builder import EVALUATORS


@EVALUATORS.register_module()
class Output:
    def __init__(self, distance_metric: str = "norm"):
        self.outputs = None
        self.distance_metric = distance_metric

    @torch.no_grad()
    def collect_output_data(self, data_loader: DataLoader, model: nn.Module, device: Device, logger) -> None:
        # collect output data for comparison
        logger.info("Collecting output data for comparison...")
        for data in alive_it(data_loader, total=len(data_loader), enrich_print=False):
            data = BatchEncoding(data).to(device)
            output = model(**data)
            output_logits = output.logits.detach().cpu()
            if self.outputs is None:
                self.outputs = output_logits
            else:
                self.outputs = torch.cat([self.outputs, output_logits], dim=0)

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
        # check how the output differs from the original model output
        assert (
            self.outputs is not None
        ), "Output data is not collected yet, run collect_output_data on the original model first."
        current_outputs = None
        for data in alive_it(data_loader, total=len(data_loader), enrich_print=False, disable=not verbose):
            data = BatchEncoding(data).to(device)
            output = model(**data)
            output_logits = output.logits.detach().cpu()
            if current_outputs is None:
                current_outputs = output_logits
            else:
                current_outputs = torch.cat([current_outputs, output_logits], dim=0)
        match self.distance_metric:
            case "norm":
                distance = self.compare_outputs_norm(current_outputs)
            case "angular_distance":
                distance = self.compare_outputs_angular_distance(current_outputs)
            case "kl_divergence":
                distance = self.compare_outputs_kl_divergence(current_outputs)
            case "js_divergence":
                distance = self.compare_outputs_js_divergence(current_outputs)
            case _:
                raise NotImplementedError(f"Unsupported distance metric: {self.distance_metric}")

        if verbose:
            logger.info(
                f"Method: {method_name}, sparsity: {sparsity:.2f}, Output difference norm: {distance.item():.4f}"
            )
        return distance

    def compare_outputs_norm(self, output_logits: torch.Tensor):
        # compare the output logits
        output_diff = self.outputs - output_logits
        output_diff_norm = torch.linalg.matrix_norm(output_diff)
        return output_diff_norm.mean()

    def compare_outputs_angular_distance(self, output_logits: torch.Tensor):
        cosine_similarity = F.cosine_similarity(self.outputs, output_logits, dim=-1)
        cosine_similarity = torch.clamp(cosine_similarity, 0.0, 1.0)
        angular_distance = torch.acos(cosine_similarity)
        assert angular_distance.shape == output_logits.shape[:-1]
        return angular_distance.mean()

    def compare_outputs_kl_divergence(self, output_logits: torch.Tensor):
        kl_divergence = F.kl_div(
            F.log_softmax(self.outputs, dim=-1),
            F.softmax(output_logits, dim=-1),
            reduction="mean",
        )
        assert kl_divergence.ndim == 0 and isinstance(kl_divergence.item(), float)
        return kl_divergence

    def compare_outputs_js_divergence(self, output_logits: torch.Tensor):
        mean_prob = 0.5 * (F.softmax(self.outputs, dim=-1) + F.softmax(output_logits, dim=-1))
        js_divergence = 0.5 * (
            F.kl_div(
                F.log_softmax(self.outputs, dim=-1),
                mean_prob,
                reduction="mean",
            )
            + F.kl_div(
                F.log_softmax(output_logits, dim=-1),
                mean_prob,
                reduction="mean",
            )
        )
        return js_divergence
