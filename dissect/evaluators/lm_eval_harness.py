import logging
from typing import Dict, Any, Optional
from copy import deepcopy

import lm_eval
from lm_eval.tasks import TaskManager
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from .builder import EVALUATORS
from ..models import build_lm_eval_wrapper
from ..utils import Device


@EVALUATORS.register_module()
class LMEvalHarness:

    def __init__(self, tokenizer: PreTrainedTokenizer, lm_eval_cfg: Dict[str, Any], lm_wrapper_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.lm_eval_cfg = deepcopy(lm_eval_cfg)
        self.tokenizer = tokenizer
        self.lm_wrapper_cfg = deepcopy(lm_wrapper_cfg)

        self.lm_task_manager = TaskManager()
        self.lm_task_manager.initialize_tasks()

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
            logger.warning(f'LMEvalHarness.evaluate: data_loader is not needed for this method. '
                           f'As the lm_eval.tasks.TaskManager will automatically load the data.')

        lm_eval_results = lm_eval.simple_evaluate(
            model=lm_eval_wrapper,
            device=device,
            task_manager=self.lm_task_manager,
            **self.lm_eval_cfg
        )

        perf_dict = dict()
        for k, v in lm_eval_results['results'].items():
            logger.info(f"Method: {method_name}, Sparsity: {sparsity}, Task: {k}, Acc: {v['acc,none']:.4f}")
            perf_dict.update({k: v['acc,none']})

        del lm_eval_wrapper
        return perf_dict
