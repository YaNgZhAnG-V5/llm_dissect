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

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER
from dissect.utils import get_target_layers, suppress_output, suppress_tqdm


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/llama3_8b.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument(
        "--exclude", "-x", type=float, default=0.0, help="rate of layers at the model front to exclude."
    )
    parser.add_argument("--prune-level", "-lv", type=str, default="am", help="At what level to prune.")
    parser.add_argument("--prune", "-p", type=str, default="loss", help="What option for prune.")
    parser.add_argument("--largest", action="store_true", help="True to prune the largest layer.")
    parser.add_argument("--eval", "-e", type=bool, default=True, help="True to evaluate on the run.")
    parser.add_argument("--load-path", "-l", type=str, help="Path to load the result if load is used for pruning.")
    parser.add_argument("--workdir", "-w", type=str, default="workdirs/layer_prune", help="Path to save the result.")
    parser.add_argument(
        "--verbose", action="store_true", help="True to print the performance of each layer before prune."
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
    smallest,
    verbose,
):
    """prune the layer that cause the minimal performance drop."""
    overall_performance = []
    for layer in target_layers:
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
        overall_performance.append(performance)
        model = testing_manager.clean_environment_hook(model=model, model_cfg=model_cfg, device=device)
        if verbose:
            logger.info(f"layer: {layer}, performance: {performance}")
    # select the layer cause the minimal eval metric
    if smallest:
        min_idx = overall_performance.index(min(overall_performance))
        return min_idx
    # select the layer cause the maximal eval metric
    else:
        max_idx = overall_performance.index(max(overall_performance))
        return max_idx


def one_shot_pruning(
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
    smallest,
    overall_performance,
    verbose,
):
    """
    prune the layer that cause the minimal performance drop.
    prune in one-shot
    a modified version of greedy_pruning
    """
    # calculate the performance of the model only once
    if overall_performance is None:
        overall_performance = []
        for layer in target_layers:
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
            overall_performance.append(performance)
            model = testing_manager.clean_environment_hook(model=model, model_cfg=model_cfg, device=device)
            if verbose:
                logger.info(f"layer: {layer}, performance: {performance}")
    # select the layer cause the minimal eval metric
    if smallest:
        min_idx = overall_performance.index(min(overall_performance))
        return min_idx, overall_performance
    # select the layer cause the maximal eval metric
    else:
        max_idx = overall_performance.index(max(overall_performance))
        return max_idx, overall_performance


def greedy_pruning_from_load(load_path: str, target_layers, pruned_layers, device):
    load_result = torch.load(load_path, map_location=device)
    result_score = []
    for target_layer in target_layers:
        result_score.append(load_result[target_layer].mean().item())
    min_idx = result_score.index(min(result_score))
    return min_idx


def random_prune(target_layers):
    random_idx = torch.randint(0, len(target_layers), (1,)).item()
    return random_idx


def main():
    args = parse_args()
    if args.prune_level == "decoder":
        target_modules = [f"{i}" for i in range(100)]
    elif args.prune_level == "am":
        target_modules = ["o_proj", "down_proj"]
    elif args.prune_level == "a":
        target_modules = ["o_proj"]
    elif args.prune_level == "m":
        target_modules = ["down_proj"]
    else:
        raise ValueError(f"Unsupported prune level {args.prune_level}")
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
    prune_dataset = build_dataset(cfg.pruning_dataset, tokenizer=tokenizer)
    test_dataset = build_dataset(cfg.test_dataset, tokenizer=tokenizer)
    prune_data_loader = DataLoader(prune_dataset, **cfg.data_loader)
    test_data_loader = DataLoader(test_dataset, **cfg.data_loader)
    target_layers, exclude_layers, total_target_layer_number = get_target_layers(model, target_modules, args.exclude)
    tabulate_target_layers = tabulate([layer.split(".") for layer in target_layers])
    logger.info(f"target layers are {tabulate_target_layers}")

    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)
    if cfg.test_cfg.evaluator["type"] == "LMEvalHarness":
        default_args = {"tokenizer": tokenizer}
    else:
        default_args = None
    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"], default_args=default_args)

    # run preparation if the evaluator is Output
    if cfg.test_cfg["evaluator"]["type"] == "Output":
        evaluator.collect_output_data(data_loader=prune_data_loader, model=model, device=device, logger=logger)
    pruned_layers = []
    result_dict = {}

    # create a overall performance list if one-shot pruning
    if args.prune == "loss_one_shot":
        overall_performance = None

    for _ in alive_it(range(total_target_layer_number), total=total_target_layer_number):
        if args.prune == "loss":
            index = greedy_pruning(
                model=model,
                model_cfg=cfg.model,
                data_loader=prune_data_loader,
                layer_dim=args.layer_dim,
                target_layers=target_layers,
                pruned_layers=pruned_layers,
                evaluator=evaluator,
                testing_manager=testing_manager,
                device=device,
                logger=logger,
                smallest=not args.largest,
                verbose=args.verbose,
            )
        elif args.prune == "loss_one_shot":
            index, overall_performance = one_shot_pruning(
                model=model,
                model_cfg=cfg.model,
                data_loader=prune_data_loader,
                layer_dim=args.layer_dim,
                target_layers=target_layers,
                pruned_layers=pruned_layers,
                evaluator=evaluator,
                testing_manager=testing_manager,
                device=device,
                logger=logger,
                smallest=not args.largest,
                verbose=args.verbose,
                overall_performance=overall_performance,
            )
        elif args.prune == "load":
            index = greedy_pruning_from_load(
                load_path=args.load_path,
                target_layers=target_layers,
                pruned_layers=pruned_layers,
                device=device,
            )
        elif args.prune == "random":
            # create random state dict and use it for evaluation
            index = random_prune(target_layers)
        else:
            raise NotImplementedError(f"Unsupported pruning method: {args.prune}")
        pruned_layers.append(target_layers[index])

        # pop selected layers
        target_layers.pop(index)
        if args.prune == "loss_one_shot":
            overall_performance.pop(index)
            # if all possible layers are pruned, set overall_performance to None
            # and run it for the newly added exclude layers
            if len(overall_performance) == 0:
                overall_performance = None

        # include excluded layers if pruning ratio just reach exclude rate
        if len(pruned_layers) == (int(total_target_layer_number * args.exclude) + 1):
            target_layers += exclude_layers

        # evaluate pruned model
        if args.eval:
            mask_state_dict = {}
            for pruned_layer in pruned_layers:
                mask_state_dict[pruned_layer] = torch.zeros(args.layer_dim, dtype=torch.bool).to(device)
            testing_manager.mask_state_dict = mask_state_dict
            testing_manager.prepare_environment(
                model=model,
                model_cfg=cfg.model,
            )
            performance = evaluator.evaluate(
                model=model,
                sparsity=0.0,
                data_loader=test_data_loader,
                device=device,
                logger=logger,
                method_name="Origin Model",
                verbose=False,
            )
            testing_manager.clean_environment_hook(model=model, model_cfg=cfg.model, device=device)
            print(f"pruned layer: {pruned_layers[-1]}, performance: {performance}")
            result_dict[pruned_layers[-1]] = performance

    # log the pruning result
    logger.info("Layer ranking:")
    for key, value in result_dict.items():
        logger.info(f"Layer: {key}, Perplexity: {value:.8f}")
    yaml.dump(result_dict, open(osp.join(args.workdir, "layer_ranking.yaml"), "w"))

    # save pruned layers
    pruning_rates = [i / 100 for i in range(5, 100, 5)]
    for pruning_rate in pruning_rates:
        mmengine.mkdir_or_exist(osp.join(args.workdir, "pruning_masks"))
        num_layers = int(total_target_layer_number * pruning_rate)
        mask_state_dict = {
            pruned_layer: torch.zeros(args.layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers[:num_layers]
        }
        assert len(mask_state_dict) == num_layers
        string_ratio = "_".join(str(pruning_rate).split("."))
        file_name = f"sparsity_{string_ratio}_pruning_masks.pth"
        save_path = osp.join(args.workdir, "pruning_masks", file_name)
        torch.save(mask_state_dict, save_path)


if __name__ == "__main__":
    main()
