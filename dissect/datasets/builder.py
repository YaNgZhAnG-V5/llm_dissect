from typing import Any, Dict

import datasets
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer


def build_dataset(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    dataset_name = cfg["dataset_name"]
    if dataset_name == "imbd":
        return build_imbd(cfg, tokenizer)
    elif dataset_name == "c4":
        return build_c4(cfg, tokenizer)


def build_c4(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    # dataset = datasets.load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    dataset = datasets.load_dataset("allenai/c4", data_files=cfg.get("data_files", None), split=cfg["split"])
    dataset = dataset.select(list(range(cfg["num_samples"]))) if cfg.get("num_samples", None) is not None else dataset
    dataset = dataset.remove_columns(["url", "timestamp"])

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        tokenized_data = tokenizer(
            examples["text"], truncation=True, padding="max_length", return_tensors="pt", max_length=cfg["max_length"]
        )
        if cfg["use_label"]:
            labels = tokenized_data.input_ids.clone()
            # let the loss ignore the padding token
            labels[labels == tokenizer.pad_token_id] = -100
            tokenized_data.update(label=labels)
        return tokenized_data

    dataset.set_format("torch")
    dataset = dataset.map(preprocess_function, batched=True).remove_columns(column_names=["text"])
    return dataset


def build_imbd(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    imdb = datasets.load_dataset("imdb", split=cfg["split"])
    dataset = imdb.shuffle().select(list(range(cfg["num_samples"])))

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="pt"
        )

    dataset.set_format("torch")
    dataset = dataset.map(preprocess_function, batched=True)
    remove_column_names = ["text"] if cfg["use_label"] else ["text", "label"]
    dataset = dataset.remove_columns(column_names=remove_column_names)
    return dataset
