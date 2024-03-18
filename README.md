# LLM Dissection

## Installation
Install via `pip install -e .` .

## Run scripts
1. Analyze the model and save the pruning masks
```shell
python scripts/analyze_and_prune.py configs/prune_bert.yaml \
  -w workdirs/prune_pert/
```
Please refer to the script for the documentation of other arguments.

2. Test pruned model
```shell
python scripts/test_pruned_mlp.py configs/prune_bert.yaml -w workdirs/debug/
```

## Running Tests

To execute the tests, you can specify a `gpu_id` to determine the processing unit.
By default, the `gpu_id` is set to `0`, indicating the first GPU. Utilizing a negative `gpu_id` value will direct
the tests to run on the CPU instead.
Here's how you can run a test with an example command:
```shell
pytest tests/test_dissect.py --gpu-id 0
```
