import os
import os.path as osp
from argparse import ArgumentParser

import mmengine
import torch
from alive_progress import alive_it
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from dissect.datasets import build_dataset
from dissect.models import build_model_and_tokenizer


def parse_args():
    parser = ArgumentParser("Test input output similarity of pruned models")
    parser.add_argument("--config", default="./configs/prune_llama.yaml", help="Path to config file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        default="workdirs/layer_prune_llama_7b_all_100_samples",
        help="Path to save the result.",
    )
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


class InputOutputSimilarity:

    def __init__(self, add_skip_connection: bool, avg_over_token: bool) -> None:
        self.add_skip_connection = add_skip_connection
        self.avg_over_token = avg_over_token
        self.input = None
        self.output = None
        self.similarities = []
        self.distances = []

    def output_hook(self, module, args, kwargs, output):
        # here is tricky, MLP takes tensor as input while self attention takes a dict
        if len(args) == 0:
            self.input = kwargs["hidden_states"]
        elif len(args) == 1:
            self.input = args[0]
        else:
            raise ValueError(f"Unexpected input: {args}")
        assert isinstance(self.input, torch.Tensor)
        if isinstance(output, tuple):
            self.output = output[0]
        else:
            self.output = output
        assert isinstance(self.output, torch.Tensor)

        # manually add skip connection
        if self.add_skip_connection:
            self.output = self.input + self.output
        self.calculate_similarity()
        self.calculate_distance()

    def calculate_similarity(self):
        assert self.input.shape == self.output.shape
        if self.avg_over_token is False:
            self.input = self.input.view(self.input.shape[0], -1)
            self.output = self.output.view(self.output.shape[0], -1)
        input_norm = torch.norm(self.input, dim=-1, keepdim=False)
        output_norm = torch.norm(self.output, dim=-1, keepdim=False)
        similarity = self.input * self.output
        similarity = similarity.sum(dim=-1) / (input_norm * output_norm)
        self.similarities.append(similarity.mean().item())

    def calculate_distance(self):
        assert self.input.shape == self.output.shape
        # TODO: bug, already calculated in calculate_similarity
        if self.avg_over_token is False:
            self.input = self.input.view(self.input.shape[0], -1)
            self.output = self.output.view(self.output.shape[0], -1)
        distance = torch.norm(self.input - self.output, dim=-1, keepdim=False)
        distance /= torch.norm(self.input, dim=-1, keepdim=False)
        self.distances.append(distance.mean().item())

    def average_similarity(self):
        return sum(self.similarities) / len(self.similarities)

    def average_distance(self):
        return max(self.distances)


def main():
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
    prune_data_loader = DataLoader(prune_dataset, **cfg.data_loader)
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
    )
    use_decoder = False
    avg_over_token = True
    save_mask = False
    if not use_decoder:
        target_modules = ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]
        # target_modules = ["self_attn", "mlp"]
    else:
        target_modules = [str(i) for i in range(32)]
    target_layer_hooks = {}
    for target_module in target_modules:
        if target_module == "self_attn" or target_module == "mlp":
            add_skip_connection = True
        else:
            add_skip_connection = False
        for name, module in model.named_modules():
            if target_module in name.split(".")[-1]:
                target_layer_hooks[name] = InputOutputSimilarity(
                    add_skip_connection=add_skip_connection, avg_over_token=avg_over_token
                )
                logger.info(f"Insert similarity extraction hook on target layer: {name}")
                module.register_forward_hook(target_layer_hooks[name].output_hook, with_kwargs=True)

    for data in alive_it(prune_data_loader, total=len(prune_data_loader), enrich_print=False):
        data = BatchEncoding(data).to(device)
        _ = model(**data)

    # save ranked similarity
    sorted_similarity = {
        k: v.average_similarity()
        for k, v in sorted(target_layer_hooks.items(), key=lambda item: -item[1].average_similarity())
    }
    print("Sorted similarity:")
    for k, v in sorted_similarity.items():
        print(f"Cos similarity {k}: {v}")
        print(f"Distance {k}: {target_layer_hooks[k].average_distance()}")
    # TODO: complete the implementation on saving layer similarity statistics
    # mmengine.mkdir_or_exist("layer_similarity_stats")
    # with open("layer_similarity_stats/overall.yml", "w") as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

    # save masks
    mask_state_dict = {}
    layer_order = list(sorted_similarity.keys())
    ratio = [i / 10 for i in range(1, 10)]
    if not save_mask:
        return None
    print("Save masks")
    mmengine.mkdir_or_exist("layer_similarity")
    for idx, r in enumerate(ratio):
        length = int(len(layer_order) * r)
        pruned_layers = layer_order[:length]
        for pruned_layer in pruned_layers:
            if "self_attn" in pruned_layer:
                mask_state_dict[pruned_layer + ".o_proj"] = torch.zeros(4096, dtype=torch.bool)
            elif "mlp" in pruned_layer:
                mask_state_dict[pruned_layer + ".down_proj"] = torch.zeros(4096, dtype=torch.bool)
        print(mask_state_dict.keys())
        assert len(mask_state_dict) == int(len(layer_order) * r)
        torch.save(mask_state_dict, osp.join("layer_similarity", f"sparsity_0_{idx + 1}_pruning_masks.pth"))


if __name__ == "__main__":
    main()
