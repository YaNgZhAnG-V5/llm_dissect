from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import datasets
from datasets import Dataset
from mmengine import MMLogger
from torch.utils.data import Dataset as TorchDataset
from transformers import BatchEncoding, PreTrainedTokenizer


def build_dataset(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Union[datasets.Dataset, TorchDataset]:
    cfg = deepcopy(cfg)
    dataset_name = cfg["dataset_name"]
    if dataset_name == "imdb":
        return build_imdb(cfg, tokenizer)
    elif dataset_name == "c4":
        return build_c4_dataset(cfg, tokenizer)
    elif dataset_name == "wikitext":
        return build_wikitext_dataset(cfg, tokenizer)
    elif dataset_name == "mmlu":
        return build_mmlu(cfg, tokenizer)
    elif dataset_name == "mixed":
        assert "num_samples" in cfg, "num_samples should be provided for MixedDataset"
        base_dataset_cfgs = deepcopy(cfg["base_datasets"])
        cfg.pop("dataset_name")
        cfg.pop("base_datasets")
        # rest config entries are common config entries for each base_dataset
        for base_cfg in base_dataset_cfgs:
            base_cfg.update(cfg)
        return MixedDataset(base_dataset_cfgs, tokenizer)
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
    dataset = datasets.load_dataset("allenai/c4", data_files=cfg.get("data_files", None), split=cfg["split"])
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    dataset = dataset.select(list(range(cfg["num_samples"]))) if cfg.get("num_samples", None) is not None else dataset
    if cfg.get("remove_columns", None) is not None:
        dataset = dataset.remove_columns(cfg["remove_columns"])
    dataset.set_format("torch")
    dataset = dataset.map(get_preprocess_function(cfg=cfg, tokenizer=tokenizer), batched=True).remove_columns(
        column_names=["text"]
    )
    return dataset


def build_wikitext_dataset(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    dataset = datasets.load_dataset(cfg.get("dataset_name"), data_files=cfg.get("data_files", None), split=cfg["split"])
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    if cfg.get("remove_columns", None) is not None:
        dataset = dataset.remove_columns(cfg["remove_columns"])
    datastring = " ".join(dataset["text"])
    return_dict = tokenizer(datastring, return_tensors="pt")
    input_ids, attn_mask = return_dict["input_ids"], return_dict["attention_mask"]

    # input_ids from return_dict has shape (1, total_seq_len). Then, the input_ids is truncated to a desired length
    # such that it can be reshaped to (num_samples, max_length)
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


def build_mmlu(cfg: Dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    mmlu = datasets.load_dataset("cais/mmlu", name="all", split=cfg["split"])
    dataset = mmlu.shuffle().select(list(range(cfg["num_samples"])))

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        # examples are batched samples
        batched_questions = examples["question"]
        batched_choices = examples["choices"]
        batched_answers = examples["answer"]
        batched_prompts = [
            (
                f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
                f"Answer: {chr(ord('A') + answer.item() - 1)}"
            )
            for question, choices, answer in zip(batched_questions, batched_choices, batched_answers)
        ]
        batch_encoding = tokenizer(
            batched_prompts, truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="pt"
        )
        labels = batch_encoding.input_ids.clone()
        # let the loss ignore the padding token
        labels[labels == tokenizer.pad_token_id] = -100
        batch_encoding["labels"] = labels
        return batch_encoding

    dataset.set_format("torch")
    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.remove_columns(column_names=["question", "choices", "answer", "subject"])
    return dataset


class MixedDataset(TorchDataset):

    def __init__(self, base_dataset_cfgs: List[Dict], tokenizer: PreTrainedTokenizer) -> None:
        for dataset_cfg in base_dataset_cfgs:
            assert "mixed" not in dataset_cfg["dataset_name"], "MixedDataset should not contain MixedDataset"
        self.datasets = [build_dataset(cfg, tokenizer) for cfg in base_dataset_cfgs]
        logger = MMLogger.get_instance("dissect")
        logger.info("Built MixedDataset: " + ", ".join([cfg["dataset_name"] for cfg in base_dataset_cfgs]))

        # sanity check if all fields are the same for all datasets
        all_columns = [set(dataset.column_names) for dataset in self.datasets]
        assert all([columns == all_columns[0] for columns in all_columns]), "All datasets should have the same columns"

    def __len__(self) -> int:
        return min([len(dataset) for dataset in self.datasets]) * len(self.datasets)

    def __getitem__(self, idx: int) -> Any:
        dataset_idx = idx % len(self.datasets)
        dataset = self.datasets[dataset_idx]
        return dataset[idx // len(self.datasets)]
