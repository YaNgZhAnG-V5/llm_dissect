import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from datasets import load_dataset
import torch
import torch.autograd.forward_ad as fw_ad


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
    # small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")

    small_train_dataset.set_format("torch", device="cuda")
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True, batch_size=16)
    # tokenized_test = small_test_dataset.map(preprocess_function, batched=True, batch_size=16)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(tokenized_train)
    data = tokenized_train[0:8]
    model = AutoModelForSequenceClassification.from_pretrained("./workdirs/bert-imdb-finetuned/checkpoint-188").to(
        "cuda:0"
    )

    # print(model)
    def replace_forward_hook(module, input, output):
        tangent = torch.ones_like(output)
        output = fw_ad.make_dual(output, tangent)
        return output

    model.bert.embeddings.register_forward_hook(replace_forward_hook)

    def inspect_forward_hook(module, input, output):
        return output

    model.bert.register_forward_hook(inspect_forward_hook)

    with torch.no_grad():
        with fw_ad.dual_level():
            result = model(data["input_ids"], data["attention_mask"], data["token_type_ids"])
            print(result)


if __name__ == "__main__":
    main()
