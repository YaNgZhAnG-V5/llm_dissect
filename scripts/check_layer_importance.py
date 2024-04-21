from argparse import ArgumentParser

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
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU ID.")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


def greedy_pruning(model, data_loader, target_layers, pruned_layers, evaluator, testing_manager, device, logger):
    """prune the layer that cause the minimal performance drop."""
    overall_performance = []
    for layer in target_layers:
        mask_state_dict = {layer: torch.zeros(4096, dtype=torch.bool).to(device)}
        for pruned_layer in pruned_layers:
            mask_state_dict[pruned_layer] = torch.zeros(4096, dtype=torch.bool).to(device)
        testing_manager.mask_state_dict = mask_state_dict
        testing_manager.prepare_environment(
            model=model,
            in_place=False,
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
        testing_manager.clean_environment_hook()
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
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    device = torch.device(f"cuda:{args.gpu_id}")
    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    model.to(device).eval()
    prune_dataset = build_dataset(cfg.pruning_dataset, tokenizer=tokenizer)
    test_dataset = build_dataset(cfg.test_dataset, tokenizer=tokenizer)
    prune_data_loader = DataLoader(prune_dataset, **cfg.data_loader)
    test_data_loader = DataLoader(test_dataset, **cfg.data_loader)
    target_module = "v_proj"
    target_layers = [name for name, _ in model.named_modules() if target_module in name]
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
        # index = greedy_pruning(
        #     model, prune_data_loader, target_layers, pruned_layers, evaluator, testing_manager, device, logger
        # )
        index = greedy_pruning_from_load(
            load_path="./workdirs/prune_vicuna_4096_5000/backward_grads.pth",
            target_layers=target_layers,
            pruned_layers=pruned_layers,
            device=device,
        )
        # index = random_prune(target_layers)
        pruned_layers.append(target_layers[index])

        # pop selected layers
        target_layers.pop(index)

        # evaluate pruned model
        mask_state_dict = {}
        for pruned_layer in pruned_layers:
            mask_state_dict[pruned_layer] = torch.zeros(4096, dtype=torch.bool).to(device)
        testing_manager.mask_state_dict = mask_state_dict
        testing_manager.prepare_environment(
            model=model,
            in_place=False,
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
        testing_manager.clean_environment_hook()
        print(f"pruned layer: {pruned_layers[-1]}, performance: {performance.item()}")


if __name__ == "__main__":
    main()
