import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat
from typing import List

import mmengine
import torch
import torch.nn.functional as F
from alive_progress import alive_it
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from dissect.datasets import build_dataset
from dissect.models import build_model_and_tokenizer
from dissect.utils import Device, suppress_output, suppress_tqdm


def collect_output_data(data_loader: DataLoader, model: nn.Module, device: Device, logger) -> None:
    # collect output data for comparison
    logger.info("Collecting output data for comparison...")
    outputs = None
    for data in alive_it(data_loader, total=len(data_loader), enrich_print=False):
        data = BatchEncoding(data).to(device)
        output = model(**data)
        output_logits = output.logits.detach().cpu()
        if outputs is None:
            outputs = output_logits
        else:
            outputs = torch.cat([outputs, output_logits], dim=0)
    return outputs


def compare_outputs_norm(outputs: torch.Tensor, output_logits: torch.Tensor):
    # compare the output logits
    output_diff = outputs - output_logits
    output_diff_norm = torch.linalg.matrix_norm(output_diff)
    return output_diff_norm.mean()


def compare_outputs_angular_distance(outputs: torch.Tensor, output_logits: torch.Tensor):
    cosine_similarity = F.cosine_similarity(outputs, output_logits, dim=-1)
    cosine_similarity = torch.clamp(cosine_similarity, 0.0, 1.0)
    angular_distance = torch.acos(cosine_similarity)
    assert angular_distance.shape == output_logits.shape[:-1]
    return angular_distance.mean()


def compare_outputs_kl_divergence(outputs: torch.Tensor, output_logits: torch.Tensor):
    kl_divergence = F.kl_div(
        F.log_softmax(output_logits, dim=-1),
        F.softmax(outputs, dim=-1),
        reduction="mean",
    )
    assert kl_divergence.ndim == 0 and isinstance(kl_divergence.item(), float)
    return kl_divergence


def compare_outputs_js_divergence(outputs: torch.Tensor, output_logits: torch.Tensor):
    mean_prob = 0.5 * (F.softmax(outputs, dim=-1) + F.softmax(output_logits, dim=-1))
    js_divergence = 0.5 * (
        F.kl_div(
            F.log_softmax(outputs, dim=-1),
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


def get_target_layers(model: torch.nn.Module, target_modules: List[str], exclude_rate: float):
    target_layers = []

    # get target layers
    for target_module in target_modules:
        target_layers += [name for name, _ in model.named_modules() if target_module in name.split(".")[-1]]
    target_layers = sorted(list(set(target_layers)))
    total_target_layer_number = len(target_layers)

    # get exclude_layers, int always return the floor value, we want to use ceil here
    # since target layers contain both attn and mlp, we divide the exclude rate by 2
    num_exclude_layers = int(len(target_layers) * exclude_rate / 2) + 1
    exclude_layers = [f".{i}." for i in range(num_exclude_layers)]
    layer_to_remove = []
    for layer in target_layers:
        for exclude_layer in exclude_layers:
            if exclude_layer in layer:
                layer_to_remove.append(layer)
    for layer in layer_to_remove:
        target_layers.remove(layer)
    exclude_layers = layer_to_remove
    return target_layers, exclude_layers, total_target_layer_number


def lambda_hook(lamb):
    def call(module, input, output):
        sigmoid_lamb = torch.sigmoid(lamb)
        if isinstance(output, torch.Tensor):
            return output * sigmoid_lamb
        else:
            # The first element is usually the hidden_states
            return (output[0] * sigmoid_lamb,) + output[1:]

    return call


def optimize_smooth_mask(
    model: torch.nn.Module, outputs, data_loader, distance_metric: str, device, logger, verbose=False
):
    """learn a smooth mask for each layer s.t. the model performance is optimal."""
    # initialize lambda and optimizer, register hook
    lambs = []
    target_modules = ["attn", "mlp"]
    num_iterations = 50
    beta = 10
    lr = 1e-1
    for name, module in model.named_modules():
        for target_module in target_modules:
            if target_module in name.split(".")[-1]:
                lamb = torch.randn(1, requires_grad=True, device=device)
                module.register_forward_hook(lambda_hook(lamb))
                lambs.append(lamb)
    optimizer = torch.optim.Adam(lambs, lr=lr)

    # optimize the mask
    # Training loop
    for it in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        mean_distance = []
        with suppress_output() and suppress_tqdm():
            assert (
                outputs is not None
            ), "Output data is not collected yet, run collect_output_data on the original model first."
            for data, original_output in alive_it(
                zip(data_loader, outputs), total=len(data_loader), enrich_print=False, disable=not verbose
            ):
                data = BatchEncoding(data).to(device)
                original_output = original_output.to(device)
                output = model(**data)
                output_logits = output.logits
                match distance_metric:
                    case "norm":
                        distance = compare_outputs_norm(original_output, output_logits)
                    case "angular_distance":
                        distance = compare_outputs_angular_distance(original_output, output_logits)
                    case "kl_divergence":
                        distance = compare_outputs_kl_divergence(original_output, output_logits)
                    case "js_divergence":
                        distance = compare_outputs_js_divergence(original_output, output_logits)
                    case _:
                        raise NotImplementedError(f"Unsupported distance metric: {distance_metric}")

                # get the l1 norm of lambda and form the loss
                lamb_tensor = torch.cat(lambs)
                lamb_tensor = torch.sigmoid(lamb_tensor)
                loss = distance + beta * torch.norm(lamb_tensor, p=1)
                loss.backward()  # Compute gradients

                # update lambda
                optimizer.step()  # Update weights

                mean_distance.append(distance.item())
                if verbose:
                    logger.info(f"distance at this batch: {distance}")
        mean_distance = sum(mean_distance) / len(mean_distance)
        logger.info(f"iteration {it}, mean distance: {mean_distance}")
    return lambs


def main():
    target_modules = ["o_proj", "down_proj"]
    args = parse_args()
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

    # run preparation to collect original output
    outputs = collect_output_data(data_loader=prune_data_loader, model=model, device=device, logger=logger)

    # perform the pruning by optimizing a smooth mask
    mask = optimize_smooth_mask(
        model=model,
        outputs=outputs,
        data_loader=prune_data_loader,
        distance_metric="js_divergence",
        device=device,
        logger=logger,
    )
    print([torch.sigmoid(i).item() for i in mask])

    # # save pruned layers
    # pruning_rates = [i / 100 for i in range(5, 100, 5)]
    # for pruning_rate in pruning_rates:
    #     mmengine.mkdir_or_exist(osp.join(args.workdir, "pruning_masks"))
    #     num_layers = int(total_target_layer_number * pruning_rate)
    #     mask_state_dict = {
    #         pruned_layer: torch.zeros(args.layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers[:num_layers]
    #     }
    #     assert len(mask_state_dict) == num_layers
    #     string_ratio = "_".join(str(pruning_rate).split("."))
    #     file_name = f"sparsity_{string_ratio}_pruning_masks.pth"
    #     save_path = osp.join(args.workdir, "pruning_masks", file_name)
    #     torch.save(mask_state_dict, save_path)


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_llama.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument(
        "--exclude", "-x", type=float, default=0.0, help="rate of layers at the model front to exclude."
    )
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


if __name__ == "__main__":
    main()
