import pytest
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast

from dissect.datasets import build_dataset
from dissect.models import build_model_and_tokenizer


@pytest.fixture(scope="session")
def model_and_tokenizer():
    model_cfg = dict(
        dtype="half",
        model_class="LlamaForCausalLM",
        model_name="meta-llama/Llama-2-7b-hf",
        tokenizer_class="AutoTokenizer",
        tokenizer_name="meta-llama/Llama-2-7b-hf",
    )
    model, tokenizer = build_model_and_tokenizer(model_cfg, "cpu")
    return model, tokenizer


def test_build_model_and_tokenizer(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(tokenizer, (LlamaTokenizerFast, LlamaTokenizer))


def test_build_dataset(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    dataset_cfg = dict(
        data_files=dict(validation="en/c4-validation.00000-of-00008.json.gz"),
        dataset_name="c4",
        max_length=256,
        num_samples=2,
        split="validation",
        use_label=True,
    )
    dataset = build_dataset(dataset_cfg, tokenizer=tokenizer)
    assert len(dataset) == 2
    sample = dataset[0]
    assert set(sample.keys()) == {"label", "input_ids", "attention_mask"}
