import logging
import os.path as osp
from collections import defaultdict
from typing import List, Tuple

import mmengine
import torch
import torch.nn.functional as F
from alive_progress import alive_it
from torch import nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizer

from ..utils import Device
from .builder import PRUNERS


@PRUNERS.register_module()
class BIBasedPruner:

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        sparsities: List[float],
        hidden_size: int,
        layer_name_pattern: str,
        pruning_layer_suffixes: Tuple[str],
        bi_algorithm: str,
        logger: logging.Logger,
        device: Device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.device = device

        self.layer_name_pattern = layer_name_pattern
        self.pruning_layer_suffixes = pruning_layer_suffixes
        self.hidden_size = hidden_size
        self.sparsities = sparsities
        if bi_algorithm == "shortgpt":
            self.bi_algorithm = ShortGPTAlgorithm()
        else:
            raise NotImplementedError(f"Unsupported BI-based pruning algorithm: {bi_algorithm}")

    def reset(self):
        self.bi_algorithm.reset()

    @torch.no_grad()
    def analyze_model(self, data_loader: DataLoader, sparsity: float) -> List[str]:
        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
            batch = BatchEncoding(batch).to(self.device)
            hidden_states = self.model(**batch, output_hidden_states=True, return_dict=True).hidden_states
            hidden_states_with_layer_names = []
            # The first element in hidden_states is the input_embeds, therefore we start from the second hidden_state,
            # which corresponds to the output of the decoder layer with index 0.
            for i in range(1, len(hidden_states)):
                layer_name = self.layer_name_pattern.format(i - 1)
                hidden_states_with_layer_names.append((layer_name, hidden_states[i - 1], hidden_states[i]))
            self.bi_algorithm.process_batch(hidden_states_with_layer_names)

        pruned_layers = self.bi_algorithm.get_pruned_layers(sparsity=sparsity)
        return pruned_layers

    def prune(self, pruned_layers: List[str], work_dir: str, sparsity: float) -> None:
        save_dir = osp.join(work_dir, "pruning_masks")
        mmengine.mkdir_or_exist(save_dir)

        mask_dict = dict()
        for pruned_layer in pruned_layers:
            for suffix in self.pruning_layer_suffixes:

                mask = torch.zeros((self.hidden_size,), dtype=torch.bool, device="cpu")
                mask_dict.update({f"{pruned_layer}{suffix}": mask})

        save_path = osp.join(save_dir, f"sparsity_{str(sparsity).replace('.', '_')}_pruning_masks.pth")
        torch.save(mask_dict, save_path)
        self.logger.info(f"Pruning masks are saved to {save_path}.")


class ShortGPTAlgorithm:

    def __init__(self) -> None:
        self.bi_score_dict = defaultdict(list)

    def reset(self) -> None:
        self.bi_score_dict.clear()

    @staticmethod
    def cosine_block_influence(x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        # x_1: (batch_size, seq_len, hidden_size); x_2: (batch_size, seq_len, hidden_size)
        # cos_sim: (batch_size, seq_len) -> scalar tensor
        cos_sim = torch.clamp(F.cosine_similarity(x_1, x_2, dim=-1), 0.0, 1.0).mean()
        return 1 - cos_sim

    def process_batch(self, hidden_states_with_layer_names: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> None:
        for layer_name, layer_input, layer_output in hidden_states_with_layer_names:
            bi_score = self.cosine_block_influence(layer_output, layer_input)
            self.bi_score_dict[layer_name].append(bi_score)

    def get_pruned_layers(self, sparsity: float) -> List[str]:
        avg_bi_score_dict = dict()
        for layer_name, bi_scores in self.bi_score_dict.items():
            mean_bi_score = torch.tensor(bi_scores).mean().item()
            avg_bi_score_dict.update({layer_name: mean_bi_score})
        # prune the layers with the smallest BI scores
        sorted_pruned_layers = sorted(list(avg_bi_score_dict.items()), key=lambda x: x[1])
        pruned_layers = [
            layer_name for layer_name, _ in sorted_pruned_layers[: int(sparsity * len(sorted_pruned_layers))]
        ]
        return pruned_layers

    # @staticmethod
    # def angular_block_influence(x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
    #     # x_1: (batch_size, seq_len, hidden_size); x_2: (batch_size, seq_len, hidden_size)
    #     cos_sim = torch.clamp(F.cosine_similarity(x_1, x_2, dim=-1), 0.0, 1.0)
    #     # ang_dist: (batch_size, seq_len) -> scalar tensor
    #     ang_dist = torch.acos(cos_sim).mean()
    #     return ang_dist
