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
        for k, v in lm_eval_results["results"].items():
            # If normalized acc exists in eval results then use it, otherwise resort to un-normalized acc.
            if "acc_norm,none" in v:
                acc_key = "acc_norm,none"
            else:
                if "acc,none" in v:
                    acc_key = "acc,none"
                else:
                    raise KeyError(
                        'LMEvalHarness: Neither "acc_norm,none" or "acc,none" exists in the evaluation results.'
                    )

            logger.info(f"Method: {method_name}, Sparsity: {sparsity}, Task: {k}, Acc: {v[acc_key]:.4f}")
            # Change '_' to '\n' for better visualization in the printed table.
            perf_dict.update({k.replace("_", "\n"): v[acc_key]})

        del lm_eval_wrapper
        # If MMLU in the tasks, only keep the average MMLU accuracy.
        if "mmlu" in perf_dict:
            logger.info("Only keeping the average MMLU accuracy. The accuracies of sub-tasks will be ignored.")
            # e.g. 'mmlu_electrical_engineering'
            ignore_keys = [k for k in perf_dict.keys() if "mmlu" in k and k != "mmlu"]
            for k in ignore_keys:
                perf_dict.pop(k)

        return perf_dict
