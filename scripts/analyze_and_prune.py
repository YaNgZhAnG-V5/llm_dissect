import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict

import mmengine
import torch
import transformers
from datasets import load_dataset
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dissect.pruners import PRUNERS


def parse_args():
    parser = ArgumentParser("Analyze and prune model.")
    parser.add_argument("--config", default="./configs/prune_bert.yaml", help="Path to yaml config file.")
    parser.add_argument(
        "--prev-result-dir",
        "-p",
        help="Directory of previous analysis result. If it is given, the analysis step will be skipped, "
        "and only pruning step is performed.",
    )
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/prune_bert/", help="Working directory to save the output files."
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


def main():
    args = parse_args()
    set_random_seed(42)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'),
    )
    logger.info("Using config:\n" + "=" * 60 + f"\n{cfg.pretty_text}\n" + "=" * 60)
    device = torch.device(f"cuda:{args.gpu_id}")

    # TODO: generalize to more models and datasets
    imdb = load_dataset("imdb")
    dataset = imdb["train"].shuffle().select(list(range(3000)))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples: Dict[str, Any]) -> transformers.BatchEncoding:
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    dataset.set_format("torch")
    dataset = dataset.map(preprocess_function, batched=True)
    remove_column_names = ["text"] if cfg.dataset.use_label else ["text", "label"]
    dataset = dataset.remove_columns(column_names=remove_column_names)
    data_loader = DataLoader(dataset, **cfg.data_loader)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.ckpt_path).to(device)
    model.eval()

    pruner = PRUNERS.build(cfg.pruner, default_args={"model": model})
    if args.prev_result_dir is not None:
        analysis_result = pruner.load_analysis_result(args.prev_result_dir, device=device)
    else:
        analysis_result = pruner.analyze_model(data_loader=data_loader, work_dir=work_dir, device=device, logger=logger)

    pruner.prune(analyze_result=analysis_result, work_dir=work_dir, logger=logger)


if __name__ == "__main__":
    main()
