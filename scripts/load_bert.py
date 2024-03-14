import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dissect.dissectors import ForwardADExtractor


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def main():
    imdb = load_dataset("imdb")
    small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")

    small_train_dataset.set_format("torch", device="cuda")
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True, batch_size=16)
    tokenized_train = tokenized_train.remove_columns(column_names=["text", "label"])
    data_loader = torch.utils.data.DataLoader(tokenized_train, batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("./workdirs/bert-imdb-finetuned/checkpoint-188").to(
        "cuda:0"
    )

    data = next(iter(data_loader))
    print(data)

    extractor = ForwardADExtractor(model, insert_layer=model.bert.embeddings)
    input_tensor = data.pop("input_ids")
    output_forward_grads = extractor.forward_ad(input_tensor, forward_kwargs=data)

    print(list(output_forward_grads.keys()))


if __name__ == "__main__":
    main()
