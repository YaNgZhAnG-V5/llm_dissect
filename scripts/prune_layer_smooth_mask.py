import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat

import mmengine
import torch
import torch.nn.functional as F
from alive_progress import alive_it
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


def binary_cross_entropy_regularization(lamb: torch.Tensor):
    reg = -lamb * torch.log(lamb) - (1 - lamb) * torch.log(1 - lamb)
    reg = reg.mean()
    return reg


def bernoulli_regularization(lamb: torch.Tensor):
    return (lamb * (1 - lamb)).mean()


def l1_regularization(lamb: torch.Tensor):
    return torch.norm(lamb, p=1)


def l2_regularization(lamb: torch.Tensor):
    return torch.norm(lamb, p=2)


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
    model: torch.nn.Module,
    target_modules,
    outputs,
    data_loader,
    distance_metric: str,
    num_iterations: int,
    alpha: float,
    beta: float,
    lr: float,
    device,
    logger,
    lamb_init: str = "random",
    verbose=False,
):
    """learn a smooth mask for each layer s.t. the model performance is optimal."""
    # initialize lambda and optimizer, register hook
    lambs = []
    for name, module in model.named_modules():
        for target_module in target_modules:
            if target_module in name.split(".")[-1]:
                if lamb_init == "random":
                    lamb = torch.rand((1,), requires_grad=True, device=device)
                elif lamb_init == "ones":
                    lamb = torch.full((1,), 2.0, requires_grad=True, device=device)
                elif lamb_init == "zeros":
                    lamb = torch.full((1,), 0.0, requires_grad=True, device=device)
                else:
                    raise NotImplementedError(f"Unsupported lambda initialization: {lamb_init}")
                module.register_forward_hook(lambda_hook(lamb))
                lambs.append(lamb)
    optimizer = torch.optim.Adam(lambs, lr=lr)

    # optimize the mask
    # Training loop
    for it in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        mean_distance = []
        mean_loss = []
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
                spasity_regularization = l1_regularization(lamb_tensor)
                polar_regularization = binary_cross_entropy_regularization(lamb_tensor)
                loss = distance + alpha * spasity_regularization + beta * polar_regularization
                loss.backward()  # Compute gradients

                # update lambda
                optimizer.step()  # Update weights

                mean_distance.append(distance.item())
                mean_loss.append(loss.item())
                if verbose:
                    logger.info(f"distance at this batch: {distance}")
        mean_distance = sum(mean_distance) / len(mean_distance)
        mean_loss = sum(mean_loss) / len(mean_loss)
        logger.info(f"iteration {it}, mean distance: {mean_distance}, mean loss: {mean_loss}")
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
    prune_data_loader = DataLoader(prune_dataset, **cfg.data_loader)

    # run preparation to collect original output
    outputs = collect_output_data(data_loader=prune_data_loader, model=model, device=device, logger=logger)

    # perform the pruning by optimizing a smooth mask
    mask = optimize_smooth_mask(
        model=model,
        target_modules=target_modules,
        outputs=outputs,
        data_loader=prune_data_loader,
        distance_metric="js_divergence",
        num_iterations=args.num_iterations,
        alpha=args.alpha,
        beta=args.beta,
        lr=args.lr,
        device=device,
        logger=logger,
        lamb_init=args.lamb_init,
    )

    # print and save pruned layers
    smooth_mask = [torch.sigmoid(i).item() for i in mask]
    logger.info(smooth_mask)
    binary_mask = [1 if i > 0.5 else 0 for i in smooth_mask]
    logger.info(binary_mask)

    # save pruned mask
    mmengine.mkdir_or_exist(osp.join(args.workdir, "pruning_masks"))
    pruned_layers = [index for index, value in enumerate(binary_mask) if value == 1]
    mask_state_dict = {pruned_layer: torch.zeros(args.layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers}
    file_name = "pruning_masks.pth"
    save_path = osp.join(args.workdir, "pruning_masks", file_name)
    torch.save(mask_state_dict, save_path)


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_llama.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU ID.")
    parser.add_argument("--layer-dim", "-d", type=int, default=4096, help="layer dimension in the model.")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate for the mask.")
    parser.add_argument("--alpha", "-a", type=float, default=5e-7, help="alpha for the sparsity regularization.")
    parser.add_argument("--beta", "-b", type=float, default=1e-8, help="beta for the polar regularization.")
    parser.add_argument(
        "--lamb-init", type=str, default="random", help="initialization for lambda. Option: random, ones, zeros"
    )
    parser.add_argument("--num-iterations", "-it", type=int, default=200, help="number of iterations for the mask.")
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
