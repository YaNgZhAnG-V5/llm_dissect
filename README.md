# LLM Dissection

## Installation
Install via `pip install -e .` .

## Run scripts
1. Prune model
```shell
python scripts/prune_mlp.py 0.2 0.4 0.6 0.8 \
  workdirs/mlp_mnist/ckpts/trained_model.pth \
  -w workdirs/prune_mlp_mnist/
```
2. Test pruned model
```shell
python scripts/test_pruned_mlp.py 0.2 0.4 0.6 0.8 \
  -p workdirs/prune_mlp_mnist/pruning_masks/ \
  -c workdirs/mlp_mnist/ckpts/trained_model.pth \
  -w workdirs/debug/ \
  --prior-path workdirs/reversed_prune_mlp_mnist/priors.pth
```

## Running Tests

To execute the tests, you can specify a `gpu_id` to determine the processing unit.
By default, the `gpu_id` is set to `0`, indicating the first GPU. Utilizing a negative `gpu_id` value will direct
the tests to run on the CPU instead.
Here's how you can run a test with an example command:
```shell
pytest tests/test_dissect.py --gpu-id 0
```
