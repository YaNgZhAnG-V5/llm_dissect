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

        self.pruning_layer_suffixes = pruning_layer_suffixes
        self.hidden_size = hidden_size
        self.sparsities = sparsities
        if bi_algorithm == "shortgpt":
            logger.info("Using ShortGPT algorithm for BI-based pruning.")
            self.bi_algorithm = ShortGPTAlgorithm(layer_name_pattern=layer_name_pattern)
        elif bi_algorithm == "ineffectiveness":
            logger.info("Using Ineffectiveness algorithm for BI-based pruning.")
            self.bi_algorithm = IneffectivenessAlgorithm(layer_name_pattern=layer_name_pattern)
        else:
            raise NotImplementedError(f"Unsupported BI-based pruning algorithm: {bi_algorithm}")

    def reset(self):
        self.bi_algorithm.reset()

    @torch.no_grad()
    def analyze_model(self, data_loader: DataLoader, sparsity: float) -> List[str]:
        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
            batch = BatchEncoding(batch).to(self.device)
            hidden_states = self.model(**batch, output_hidden_states=True, return_dict=True).hidden_states
            self.bi_algorithm.process_batch(hidden_states, sparsity=sparsity)

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


class BaseBIAlgorithm:

    def __init__(self, layer_name_pattern: str) -> None:
        self.layer_name_pattern = layer_name_pattern
        self.bi_score_dict = defaultdict(list)

    def reset(self) -> None:
        self.bi_score_dict.clear()

    def process_batch(self, hidden_states: List[torch.Tensor], sparsity: float) -> None:
        raise NotImplementedError

    def get_pruned_layers(self, sparsity: float) -> List[str]:
        raise NotImplementedError


class ShortGPTAlgorithm(BaseBIAlgorithm):

    def __init__(self, layer_name_pattern: str) -> None:
        super().__init__(layer_name_pattern=layer_name_pattern)

    @staticmethod
    def cosine_block_influence(x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        # x_1: (batch_size, seq_len, hidden_size); x_2: (batch_size, seq_len, hidden_size)
        # cos_sim: (batch_size, seq_len) -> scalar tensor
        cos_sim = torch.clamp(F.cosine_similarity(x_1, x_2, dim=-1), 0.0, 1.0).mean()
        return 1 - cos_sim

    def process_batch(self, hidden_states: List[torch.Tensor], sparsity: float) -> None:
        # sparsity is not used. Just for compatibility.
        hidden_states_with_layer_names = []
        # The first element in hidden_states is the input_embeds, therefore we start from the second hidden_state,
        # which corresponds to the output of the decoder layer with index 0.
        for i in range(1, len(hidden_states)):
            layer_name = self.layer_name_pattern.format(i - 1)
            hidden_states_with_layer_names.append((layer_name, hidden_states[i - 1], hidden_states[i]))

        for layer_name, layer_input, layer_output in hidden_states_with_layer_names:
            bi_score = self.cosine_block_influence(layer_output, layer_input)
            self.bi_score_dict[layer_name].append(bi_score)

    def get_pruned_layers(self, sparsity: float) -> List[str]:
        avg_bi_score_dict = dict()
        for layer_name, bi_scores in self.bi_score_dict.items():
            mean_bi_score = torch.tensor(bi_scores).mean().item()
            avg_bi_score_dict.update({layer_name: mean_bi_score})
        # prune the layers with the smallest BI scores
        sorted_bi_scores = sorted(list(avg_bi_score_dict.items()), key=lambda x: x[1])
        pruned_layers = [layer_name for layer_name, _ in sorted_bi_scores[: int(sparsity * len(sorted_bi_scores))]]
        return pruned_layers


class IneffectivenessAlgorithm(BaseBIAlgorithm):

    def __init__(self, layer_name_pattern: str) -> None:
        super().__init__(layer_name_pattern=layer_name_pattern)

    @staticmethod
    def angular_block_influence(x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        # x_1: (batch_size, seq_len, hidden_size); x_2: (batch_size, seq_len, hidden_size)
        cos_sim = torch.clamp(F.cosine_similarity(x_1, x_2, dim=-1), 0.0, 1.0)
        # ang_dist: (batch_size, seq_len) -> scalar tensor
        ang_dist = torch.acos(cos_sim).mean()
        return ang_dist

    def process_batch(self, hidden_states: List[torch.Tensor], sparsity: float) -> None:
        # hidden_stats includes input_embdeds. So the number of layers is len(hidden_states) - 1
        num_pruned_layers = int(sparsity * (len(hidden_states) - 1))
        if num_pruned_layers < 1:
            raise ValueError(
                "int(sparsity * (len(hidden_states) - 1)) < 1. Please increase the sparsity or check hidden_states."
            )
        # the fist element in hidden_states is the input_embeds, therefore we start from the second hidden_state,
        for i in range(1, len(hidden_states) + 1 - num_pruned_layers):
            start_layer_index = i - 1
            end_layer_index = i + num_pruned_layers - 1

            start_hidden_states = hidden_states[i - 1]
            end_hidden_states = hidden_states[i + num_pruned_layers - 1]
            bi_score = self.angular_block_influence(start_hidden_states, end_hidden_states)
            # key is a tuple that represents interval: [start, end). The end layer is not included.
            self.bi_score_dict[(start_layer_index, end_layer_index)].append(bi_score)

    def get_pruned_layers(self, sparsity: float) -> List[str]:
        # sparsity is not used. Just for compatibility.
        avg_bi_score_dict = dict()
        for (start_layer_index, end_layer_index), bi_scores in self.bi_score_dict.items():
            mean_bi_score = torch.tensor(bi_scores).mean().item()
            avg_bi_score_dict.update({(start_layer_index, end_layer_index): mean_bi_score})
        # get the layer index interval associated with the smallest BI score
        sorted_layer_intervals = sorted(list(avg_bi_score_dict.items()), key=lambda x: x[1])
        prune_start_layer_index, prune_end_layer_index = sorted_layer_intervals[0][0]
        pruned_layers = [
            self.layer_name_pattern.format(i) for i in range(prune_start_layer_index, prune_end_layer_index)
        ]

        return pruned_layers
