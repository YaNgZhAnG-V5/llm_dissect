import os
import os.path as osp
import re
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict

import mmengine
import torch
from alive_progress import alive_it
from mmengine.runner import set_random_seed
from torch import nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from dissect.datasets import build_dataset
from dissect.dissectors import ActivationExtractor
from dissect.models import build_model_and_tokenizer


def parse_args():
    parser = ArgumentParser("Save input and output of target layers.")

    parser.add_argument("config", help="Path to config")
    parser.add_argument(
        "--work-dir",
        "-w",
        default="workdirs/vicuna_c4_layer_in_out/",
        help="Working directory to save the output files.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    return parser.parse_args()


def match_layer_name_pattern(layer_name: str, pattern: str) -> bool:
    if re.match(pattern, layer_name):
        return True
    else:
        return False


def main():
    args = parse_args()
    set_random_seed(42)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
    in_out_save_dir = osp.join(work_dir, "layer_input_output")
    mmengine.mkdir_or_exist(in_out_save_dir)

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'),
    )
    logger.info("Using config:\n" + "=" * 60 + f"\n{cfg.pretty_text}\n" + "=" * 60)
    device = torch.device(f"cuda:{args.gpu_id}")

    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)
    model.eval()

    dataset = build_dataset(cfg.dataset, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, **cfg.data_loader)
    assert (
        data_loader.batch_size == 1
    ), "data_loader.batch_size needs to be 1 for extracting and saving layer input and output"

    # exclude some layers
    layer_dict: Dict[str, nn.Module] = dict()
    # TODO: move to config
    layer_name_pattern = r"^model\.layers\.(\d+)\.(self_attn|mlp)$"
    for layer_name, layer_module in model.named_modules():
        if match_layer_name_pattern(layer_name, layer_name_pattern):
            layer_dict.update({layer_name: layer_module})
            mmengine.mkdir_or_exist(osp.join(in_out_save_dir, layer_name))
    # directly pass the filtered layer dict to the extractor
    # TODO move input_key_mapping to config
    extractor = ActivationExtractor(model=model, layers=layer_dict, norm=False, input_key_strategy="vicuna")

    # Folder structure: the outer level comprises various model layers,
    # while the inner level houses the files corresponding to each sample.
    with torch.no_grad():
        for batch_index, batch in alive_it(enumerate(data_loader), total=len(data_loader), enrich_print=False):
            batch = BatchEncoding(batch).to(device)
            input_ids = batch.pop("input_ids")

            layer_out_dict, layer_in_dict = extractor.extract_activations(input_ids, forward_kwargs=batch)
            assert len(layer_in_dict) == len(layer_out_dict)

            for layer_name, layer_in_tensor in layer_in_dict.items():
                layer_out_tensor = layer_out_dict[layer_name]
                result_dict = {"input": layer_in_tensor, "output": layer_out_tensor}

                torch.save(result_dict, osp.join(in_out_save_dir, layer_name, f"{batch_index}.pth"))

    logger.info(f"Layer input output of each sample is saved to {in_out_save_dir}")


if __name__ == "__main__":
    main()
