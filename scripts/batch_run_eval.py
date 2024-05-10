import argparse
import os
from itertools import product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", nargs="+", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", nargs="+", type=str, required=True)
    args = parser.parse_args()

    cuda_visible_devices = ",".join(args.gpu_id)
    print(f"GPU: {cuda_visible_devices}")
    model = args.model
    configs = [f"configs/{model}_per_attn.yaml", f"configs/{model}_per_attn_mmlu.yaml"]
    modes = args.mode
    print(f"Configs: {configs}")
    print(f"Modes: {modes}")
    for config, mode in product(configs, modes):
        os.system(
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
            f"python scripts/eval_pruned_models.py --config {config} "
            f"-p workdirs/{model}_{mode}_10_samples/ -w workdirs/{model}_{mode}_10_samples/ --gpu-id 0"
        )


if __name__ == "__main__":
    main()
