from copy import deepcopy
from typing import Any, Callable, Dict, Union

import datasets
import torch.utils.data
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer


def build_dataset(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
    cfg = deepcopy(cfg)
    dataset_name = cfg["dataset_name"]
    if dataset_name == "imdb":
        return build_imdb(cfg, tokenizer)
    elif dataset_name == "c4":
        return build_c4_dataset(cfg, tokenizer)
    elif dataset_name == "wikitext":
        return build_wikitext_dataset(cfg, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_preprocess_function(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Callable:

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        tokenized_data = tokenizer(
            examples["text"], truncation=True, padding="max_length", return_tensors="pt", max_length=cfg["max_length"]
        )
        if cfg["use_label"]:
            labels = tokenized_data.input_ids.clone()
            # let the loss ignore the padding token
            labels[labels == tokenizer.pad_token_id] = -100
            tokenized_data.update(labels=labels)
        return tokenized_data

    return preprocess_function


def build_c4_dataset(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    dataset = datasets.load_dataset(cfg.get("dataset_name"), data_files=cfg.get("data_files", None), split=cfg["split"])
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    dataset = dataset.select(list(range(cfg["num_samples"]))) if cfg.get("num_samples", None) is not None else dataset
    if cfg.get("remove_columns", None) is not None:
        dataset = dataset.remove_columns(cfg["remove_columns"])
    dataset.set_format("torch")
    dataset = dataset.map(get_preprocess_function(cfg=cfg), batched=True).remove_columns(column_names=["text"])
    return dataset


def build_wikitext_dataset(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    dataset = datasets.load_dataset(cfg.get("dataset_name"), data_files=cfg.get("data_files", None), split=cfg["split"])
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    if cfg.get("remove_columns", None) is not None:
        dataset = dataset.remove_columns(cfg["remove_columns"])
    datastring = " ".join(dataset["text"])
    return_dict = tokenizer(datastring, return_tensors="pt")
    input_ids, attn_mask = return_dict["input_ids"], return_dict["attention_mask"]

    # truncate the dataset to the desired length
    input_ids = input_ids[:, : cfg["num_samples"] * cfg["max_length"]]
    input_ids = input_ids.view(-1, cfg["max_length"])
    attn_mask = attn_mask[:, : cfg["num_samples"] * cfg["max_length"]]
    attn_mask = attn_mask.view(-1, cfg["max_length"])
    dataset = {"input_ids": input_ids, "attention_mask": attn_mask}
    if cfg["use_label"]:
        labels = input_ids.clone()
        # let the loss ignore the padding token
        labels[labels == tokenizer.pad_token_id] = -100
        dataset.update(labels=labels)
    dataset = Dataset.from_dict(dataset)
    dataset.set_format("torch")
    return dataset


def build_imdb(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    imdb = datasets.load_dataset("imdb", split=cfg["split"])
    dataset = imdb.shuffle().select(list(range(cfg["num_samples"])))
    dataset = dataset.rename_column("label", "labels")

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="pt"
        )

    dataset.set_format("torch")
    dataset = dataset.map(preprocess_function, batched=True)
    remove_column_names = ["text"] if cfg["use_label"] else ["text", "labels"]
    dataset = dataset.remove_columns(column_names=remove_column_names)
    return dataset
