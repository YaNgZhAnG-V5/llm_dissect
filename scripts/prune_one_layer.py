import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat
from typing import List

import mmengine
import torch
import yaml
from alive_progress import alive_it
from tabulate import tabulate
from torch.utils.data import DataLoader

from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER
from dissect.utils import suppress_output, suppress_tqdm


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument(
        "--config", default="./configs/prune_one_layer_diff_tasks/llama3_8b_lm_eval.yaml", help="Path to config file."
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument("--load-path", "-l", type=str, help="Path to load the result if load is used for pruning.")
    parser.add_argument(
        "--workdir", "-w", type=str, default="workdirs/prune_one_layer_boolq", help="Path to save the result."
    )
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


def greedy_pruning(
    model,
    model_cfg,
    data_loader: DataLoader,
    layer_dim: int,
    target_layers: List[str],
    pruned_layers: List,
    evaluator,
    testing_manager,
    device,
    logger,
):
    """prune the layer that cause the minimal performance drop."""
    ret_dict = {}

    # include original model performance
    target_layers = [None] + target_layers
    for layer in alive_it(target_layers, total=len(target_layers)):
        if layer is None:
            pass
        else:
            mask_state_dict = {layer: torch.zeros(layer_dim, dtype=torch.bool)}
            for pruned_layer in pruned_layers:
                mask_state_dict[pruned_layer] = torch.zeros(layer_dim, dtype=torch.bool)
            testing_manager.mask_state_dict = mask_state_dict
            testing_manager.prepare_environment(
                model=model,
                model_cfg=model_cfg,
            )
        with suppress_output() and suppress_tqdm():
            performance = evaluator.evaluate(
                model=model,
                sparsity=0.0,
                data_loader=data_loader,
                device=device,
                logger=logger,
                method_name="Greedy Pruning",
                verbose=False,
            )
        if isinstance(performance, torch.Tensor):
            performance = performance.item()
        elif isinstance(performance, dict):
            performance = list(performance.values())[0]
        else:
            raise NotImplementedError(f"Unsupported performance type: {type(performance)}")
        model = testing_manager.clean_environment_hook(model=model, model_cfg=model_cfg, device=device)
        if layer is None:
            layer = "original"
        logger.info(f"layer: {layer}, performance: {performance}")
        ret_dict[layer] = performance
    return ret_dict


class OrderedLayerNameHook:
    """get orders of all the target layers via hook."""

    def __init__(self, target_layers: dict):
        self.target_layers = target_layers
        self.ordered_target_layers = []

    def __call__(self, module, input, output):
        self.ordered_target_layers.append(
            list(self.target_layers.keys())[list(self.target_layers.values()).index(module)]
        )


def get_target_layers(model: torch.nn.Module, target_modules: List[str]):
    target_layers = {}
    hooks = []
    hook_callback = OrderedLayerNameHook(target_layers)
    # get target layers
    for target_module in target_modules:
        for name, layer in model.named_modules():
            if target_module in name.split(".")[-1] and name not in target_layers:
                target_layers[name] = layer
                hooks.append(layer.register_forward_hook(hook_callback))
    with torch.no_grad():
        _ = model(model.dummy_inputs["input_ids"].to(model.device))
    # remove hooks
    while hooks:
        hooks.pop().remove()
    target_layers = hook_callback.ordered_target_layers
    return target_layers


def main():
    args = parse_args()
    if "mixtral" in args.config:
        target_modules = ["self_attn", "block_sparse_moe"]
    else:
        target_modules = ["o_proj", "down_proj"]
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    exist_warning = True if os.path.exists(args.workdir) else False
    mmengine.mkdir_or_exist(args.workdir)
    device = torch.device(f"cuda:{args.gpu_id}")
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(args.workdir, f"{time_stamp}.log"),
    )
    if exist_warning:
        logger.warning(f"workdir {args.workdir} already exists, consider save it in another place.")
    logger.info(f"Model: \n{pformat(cfg.model)}")
    logger.info(f"Prune dataset: \n{pformat(cfg.pruning_dataset)}")
    logger.info(f"Test dataset: \n{pformat(cfg.test_dataset)}")
    logger.info(f"Testing manager: \n{pformat(cfg.test_cfg.testing_manager)}")
    logger.info(f"Evaluator:\n{pformat(cfg.test_cfg.evaluator)}")
    logger.info(f"parsed arguments: \n{pformat(vars(args))}")
    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", [])
    if len(cuda_visible_devices) == 0:
        model.to(device).eval()
    else:
        model.eval()
    target_layers = get_target_layers(model, target_modules)
    tabulate_target_layers = tabulate([layer.split(".") for layer in target_layers])
    logger.info(f"target layers are {tabulate_target_layers}")

    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)
    if cfg.test_cfg.evaluator["type"] == "LMEvalHarness":
        default_args = {"tokenizer": tokenizer}
    else:
        default_args = None
    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"], default_args=default_args)

    pruned_layers = []
    result_dict = greedy_pruning(
        model=model,
        model_cfg=cfg.model,
        data_loader=None,
        layer_dim=args.layer_dim,
        target_layers=target_layers,
        pruned_layers=pruned_layers,
        evaluator=evaluator,
        testing_manager=testing_manager,
        device=device,
        logger=logger,
    )

    # log the pruning result
    yaml.dump(result_dict, open(osp.join(args.workdir, "prune_one_layer.yaml"), "w"))


if __name__ == "__main__":
    main()
