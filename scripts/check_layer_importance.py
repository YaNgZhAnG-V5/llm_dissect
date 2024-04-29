import os
import os.path as osp
from argparse import ArgumentParser
from typing import List

import mmengine
import torch
from alive_progress import alive_it
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer
from dissect.pruners import TESTING_MANAGER


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_vicuna.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument("--prune", "-p", type=str, default="loss", help="What option for prune.")
    parser.add_argument("--eval", "-e", type=bool, default=True, help="True to evaluate on the run.")
    parser.add_argument("--load-path", "-l", type=str, help="Path to load the result if load is used for pruning.")
    parser.add_argument("--workdir", "-w", type=str, default="workdirs/layer_prune", help="Path to save the result.")
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
        performance = evaluator.evaluate(
            model=model,
            sparsity=0.0,
            data_loader=data_loader,
            device=device,
            logger=logger,
            method_name="Greedy Pruning",
            verbose=False,
        )
        overall_performance.append(performance.item())
        model = testing_manager.clean_environment_hook(model=model, model_cfg=model_cfg, device=device)
    # select least influencial layer
    min_idx = overall_performance.index(min(overall_performance))
    return min_idx


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
    target_modules = ["o_proj"]
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    device = torch.device(f"cuda:{args.gpu_id}")
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
    target_layers = []
    for target_module in target_modules:
        target_layers += [name for name, _ in model.named_modules() if target_module in name]
    print(f"target layers are {target_layers}")

    # create random state dict and use it for evaluation
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
    )
    testing_manager = TESTING_MANAGER.build(cfg.test_cfg.testing_manager)
    evaluator = EVALUATORS.build(cfg.test_cfg["evaluator"])
    pruned_layers = []
    for _ in alive_it(range(len(target_layers)), total=len(target_layers)):
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
            )
        elif args.prune == "load":
            index = greedy_pruning_from_load(
                load_path=args.load_path,
                target_layers=target_layers,
                pruned_layers=pruned_layers,
                device=device,
            )
        elif args.prune == "random":
            index = random_prune(target_layers)
        else:
            raise NotImplementedError(f"Unsupported pruning method: {args.prune}")
        pruned_layers.append(target_layers[index])

        # pop selected layers
        target_layers.pop(index)

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
            print(f"pruned layer: {pruned_layers[-1]}, performance: {performance.item()}")

    # save pruned layers
    pruning_rates = [i / 100 for i in range(5, 90, 5)]
    for pruning_rate in pruning_rates:
        mmengine.mkdir_or_exist(args.workdir)
        num_layers = int(len(pruned_layers) * pruning_rate)
        mask_state_dict = {
            pruned_layer: torch.zeros(args.layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers[:num_layers]
        }
        string_ratio = "_".join(str(pruning_rate).split("."))
        file_name = f"sparsity_{string_ratio}_pruning_masks.pth"
        save_path = osp.join(args.workdir, file_name)
        torch.save(mask_state_dict, save_path)


if __name__ == "__main__":
    main()
