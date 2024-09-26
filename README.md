# LLM Dissection

## Installation
Install via `pip install -e .` .

## Run scripts

### Perform layer pruning
```shell
CUDA_VISIBLE_DEVICES=gpu_id python scripts/prune_layer.py --config configs/llama3_8b.yaml -x 0.4 -w workdirs/remove_layers
```
Change the gpu_id to the actual GPU id you want to use, options are like: 0 or 0,1 (can be a list of ids seperated by space).
Change the config file to run on other models. Option -x indicates how many layers at front to ignore during pruning.
Please refer to the script for the documentation of other arguments.

You may also try the experimental script for gradient mask optimization pruning strategy:
```shell
CUDA_VISIBLE_DEVICES=gpu_id python scripts/prune_layer_l0_mask.py --config configs/llama3_8b_mask_optim.yaml --lr-mask 3e-3 --gamma 3e-7 --eta 3e-8 --warm-up 0 -it 5 --skip-layers 0 --use-lagrangian-proxy --weight-lm-loss 2e-6
```

### Evaluate pruned model
To execute the model evaluation, you can specify a `gpu_id` to determine the processing unit.

Here's how you can run a test with an example command:
```shell
# evaluate on zero-shot tasks
CUDA_VISIBLE_DEVICES=gpu_id python scripts/eval_pruned_models.py --config configs/llama3_8b_perp.yaml -p workdirs/remove_layers --mask-prefix length --cfg-options test_cfg.testing_manager.in_place=True

# evaluate on MMLU tasks
CUDA_VISIBLE_DEVICES=gpu_id python scripts/eval_pruned_models.py --config configs/llama3_8b_mmlu.yaml -p workdirs/remove_layers --mask-prefix length --cfg-options test_cfg.testing_manager.in_place=True

```
You can change the config file to evaluate other models

Option in_place performs actual layer removal while setting it to false will mask the input of the layer output instead of prune it (but the outcome should be the same).

### Run ablation study
This script is intergrates the prune and evaluation to run ablation study on one-shot pruning, use perplexity as pruning metric, and SLEB. Please check the oneshot_perplexity_sleb.py to understand the usage of all arguments
```shell
python scripts/rebuttal_ablation/oneshot_perplexity_sleb.py --device 0 1 --model llama3_70b --perplexity --exclude --prune --eval --eval-mmlu -w workdirs/ablation/perplexity_exclude/llama3_70b
```
