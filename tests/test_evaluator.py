import mmengine
import torch
from torch.utils.data import DataLoader

from dissect.datasets import build_dataset
from dissect.evaluators import EVALUATORS
from dissect.models import build_model_and_tokenizer


def test_output_evaluator():
    device = "cuda:1"
    cfg_model = {
        "model_class": "LlamaForCausalLM",
        "model_name": "meta-llama/Llama-2-7b-hf",
        "model_args": {
            "torch_dtype": "float32",
            "do_sample": False,
            "use_cache": True,
        },
        "tokenizer_class": "AutoTokenizer",
        "tokenizer_name": "meta-llama/Llama-2-7b-hf",
        "mem_efficient_sdp": False,
    }
    model, tokenizer = build_model_and_tokenizer(cfg_model, device=device)
    cfg_test_dataset = {
        "dataset_name": "wikitext",
        "num_samples": 5,
        "data_files": {"validation": "wikitext-2-raw-v1/train-00000-of-00001.parquet"},
        "split": "validation",
        "use_label": True,
        "max_length": 256,
    }
    cfg_data_loader = {"batch_size": 1, "shuffle": False}
    test_dataset = build_dataset(cfg_test_dataset, tokenizer=tokenizer)
    test_data_loader = DataLoader(test_dataset, **cfg_data_loader)
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
    )

    cfg = {"type": "Output"}
    evaluator = EVALUATORS.build(cfg)

    # test collect_output_data works
    evaluator.collect_output_data(data_loader=test_data_loader, model=model, device=device, logger=logger)
    assert evaluator.outputs is not None
    assert evaluator.outputs.shape[-1] == 32000
    assert evaluator.outputs.shape[-2] == 256

    # test compare_outputs works
    current_outputs = torch.zeros_like(evaluator.outputs)
    output_diff_norm = evaluator.compare_outputs(current_outputs)
    assert abs(output_diff_norm.item() - torch.linalg.matrix_norm(evaluator.outputs).mean().item()) < 1e-6

    # test evaluate works
    performance = evaluator.evaluate(
        model=model, sparsity=0.0, data_loader=test_data_loader, device=device, logger=logger, method_name="Output"
    )
    assert isinstance(performance.item(), float)


if __name__ == "__main__":
    test_output_evaluator()
