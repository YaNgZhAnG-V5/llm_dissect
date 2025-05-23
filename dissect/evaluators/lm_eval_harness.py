import logging
from copy import deepcopy
from typing import Any, Dict, Optional

import lm_eval
import torch
from lm_eval.tasks import TaskManager
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..models import build_lm_eval_wrapper
from ..utils import Device
from .builder import EVALUATORS


@EVALUATORS.register_module()
class LMEvalHarness:

    _avg_task_keys = ("mmlu", "lambada")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lm_eval_cfg: Dict[str, Any],
        lm_wrapper_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.lm_eval_cfg = deepcopy(lm_eval_cfg)
        self.tokenizer = tokenizer
        self.lm_wrapper_cfg = deepcopy(lm_wrapper_cfg)

        self.lm_task_manager = TaskManager()
        self.lm_task_manager.initialize_tasks()

    @torch.inference_mode()
    def evaluate(
        self,
        model: PreTrainedModel,
        sparsity: float,
        data_loader: DataLoader,
        device: Device,
        logger: logging.Logger,
        method_name: str,
        verbose: bool = False,
    ) -> Dict[str, float]:
        lm_eval_wrapper = build_lm_eval_wrapper(model, self.tokenizer, lm_wrapper_cfg=self.lm_wrapper_cfg)

        if data_loader is not None:
            logger.warning(
                "LMEvalHarness.evaluate: data_loader is not needed for this method. "
                "As the lm_eval.tasks.TaskManager will automatically load the data."
            )

        lm_eval_results = lm_eval.simple_evaluate(
            model=lm_eval_wrapper, device=device, task_manager=self.lm_task_manager, **self.lm_eval_cfg
        )

        perf_dict = dict()
        for task_name, task_result in lm_eval_results["results"].items():
            log_str = f"Method: {method_name}, Sparsity: {sparsity}, Task: {task_name}, "
            # whether to use a random key in the task_result dict. This flag is set to True only when none of
            # ("acc_norm,none", "acc,none", "perplexity,none") exists in task_result.
            use_default_task_key = True
            if "acc_norm,none" in task_result:
                # If normalized acc exists in eval results then use it, otherwise resort to un-normalized acc.
                metric_key, log_str_key = "acc_norm,none", "acc_norm"
                log_str += f"Acc: {task_result[metric_key]:.4f}, "
                skip_acc_none = True
                self.update_perf_dict(perf_dict, task_name, log_str_key, task_result[metric_key])
                use_default_task_key = False
            else:
                skip_acc_none = False

            score_keys_and_log_str_keys = [
                ("acc,none", "acc"),
                ("perplexity,none", "ppl"),
                ("bleu,none", "bleu"),
                ("ter,none", "ter"),
                ("exact_match,flexible-extract", "em-flexible"),
                ("contains,none", "contains"),
                ("f1,none", "f1"),
            ]
            for score_key, log_str_key in score_keys_and_log_str_keys:
                if skip_acc_none and score_key == "acc,none":
                    # Skip "acc,none" if "acc_norm,none" exists.
                    continue
                if score_key in task_result:
                    log_str += f"{log_str_key}: {task_result[score_key]:.4f}, "
                    self.update_perf_dict(perf_dict, task_name, log_str_key, task_result[score_key])
                    use_default_task_key = False

            if use_default_task_key:
                # If none of the above keys exists, use the first key in the task_result dict.
                metric_key = list(task_result.keys())[0]
                log_str_key = metric_key
                log_str += f"{metric_key}: {task_result[metric_key]}"
                self.update_perf_dict(perf_dict, task_name, log_str_key, task_result[metric_key])

            logger.info(log_str)

        del lm_eval_wrapper
        return perf_dict

    def update_perf_dict(
        self, perf_dict: Dict[str, float], task_name: str, log_str_key: str, perf_value: float
    ) -> None:
        for avg_key in self._avg_task_keys:
            # Only keeping the average performance for avg_key.The accuracies of sub-tasks will be ignored.
            # E.g. "mmlu" will be kept, but sub-tasks e.g. "mmlu_electrical_engineering_acc" are ignored.
            if (avg_key in task_name) and (avg_key != task_name):
                return
        # Change '_' to '\n' for better visualization in the printed table.
        perf_dict.update({f"{task_name}_{log_str_key}".replace("_", "\n"): perf_value})
