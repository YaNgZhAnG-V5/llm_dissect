import random
from typing import Any, Dict

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_vacuna_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    return tokenizer


def get_vacuna_model():
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map="auto")
    return model


def get_vacuna():
    tokenizer = get_vacuna_tokenizer()
    model = get_vacuna_model()
    return model, tokenizer


def get_c4(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    torch.manual_seed(seed)
    traindata = (
        load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00000-of-01024.json.gz",
            split="train",
        )
        .shuffle()
        .select(list(range(nsamples)))
    )
    traindata = traindata.remove_columns(column_names=["url", "timestamp"])

    def preprocess_function(examples: Dict[str, Any]) -> transformers.BatchEncoding:
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=seqlen, return_tensors="pt"
        )

    traindata.set_format("torch")
    traindata = traindata.map(preprocess_function, batched=True)
    data_loader = DataLoader(
        traindata,
        batch_size=1,
        shuffle=False,
    )
    return data_loader


def main():
    train_loader = get_c4(100, 42, 256, get_vacuna_tokenizer())
    model = get_vacuna_model()

    data = next(iter(train_loader))
    data.pop("text")

    output = model(**data)
    print(output[0].shape)


if __name__ == "__main__":
    main()
