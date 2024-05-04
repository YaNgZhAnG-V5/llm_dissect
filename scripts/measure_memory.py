from argparse import ArgumentParser

import mmengine
import torch
from alive_progress import alive_it

from dissect.models import build_model_and_tokenizer


def get_module_memory_consumption(module, device):
    module.to(device)
    memory_in_mb = torch.cuda.memory_allocated() / 1024 / 1024
    module.to(torch.device("cpu"))
    assert torch.cuda.memory_allocated(device=device) == 0.0
    return memory_in_mb


def get_model_memory_consumption_summary(model, device, verbose=False):
    assert torch.cuda.memory_allocated(device=device) == 0.0, "GPU is not empty, cannot measure memory consumption."
    memory_consumption = dict()
    for name, module in alive_it(
        model.named_modules(),
        total=len(list(model.named_modules())),
    ):
        if name == "":
            name = "overall"
        module.to(torch.device("cuda:0"))
        memory_in_mb = get_module_memory_consumption(module, device)
        if verbose:
            print(f"GPU Memory Requirement for {name}: {memory_in_mb} MiB")
        memory_consumption.update({name: memory_in_mb})
    return memory_consumption


def get_model_param_summary(model, device, verbose=False):
    params_dict = dict()
    overall_params = 0
    for name, params in alive_it(
        model.named_parameters(),
        total=len(list(model.named_parameters())),
    ):
        num_params = params.numel()
        overall_params += num_params
        if verbose:
            print(f"GPU Memory Requirement for {name}: {params} MiB")
        params_dict.update({name: num_params})
    params_dict.update({"overall": overall_params})
    return params_dict


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/prune_llama.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    device = torch.device(f"cuda:{args.gpu_id}")
    model, _ = build_model_and_tokenizer(cfg.model, device=device)
    # model.to(torch.device("cpu")).eval()
    modules_of_interest = ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm", "overall"]

    # get model parameters summary
    print("######## Parameter Summary: ########")
    params_dict = get_model_param_summary(model, device)
    summary_dict = dict()
    for module_name in modules_of_interest:
        summary_dict.update({module_name: 0.0})
    for name, val in params_dict.items():
        for module_name in modules_of_interest:
            if module_name in name:
                print(f"# of parameters for {name}: {val / 1000}k")
                summary_dict[module_name] += val
    print("Summary:")
    for module_name, val in summary_dict.items():
        print(f"Overall parameters for all {module_name}: {val / 1000}k")

    # get memory consumption summary for interested modules
    print("######## Memory Consumption Summary: ########")
    memory_consumption = get_model_memory_consumption_summary(model, device)
    summary_dict = dict()
    for module_name in modules_of_interest:
        summary_dict.update({module_name: 0.0})
    for name, val in memory_consumption.items():
        for module_name in modules_of_interest:
            if name.split(".")[-1] == module_name:
                print(f"GPU Memory Requirement for {name}: {val} MiB")
                summary_dict[module_name] += val
    print("Summary:")
    for module_name, val in summary_dict.items():
        print(f"Overall GPU Memory Requirement for all {module_name}: {val} MiB")


if __name__ == "__main__":
    main()
