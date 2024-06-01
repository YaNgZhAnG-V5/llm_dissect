import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", nargs="+", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    if "8b" in args.model:
        dim = 4096
    elif "70b" in args.model:
        dim = 8192
    else:
        raise ValueError(f"Unsupported model {args.model}")
    gpu_id_str = ",".join(args.gpu_id)
    if args.task == "mmlu" or "perplexity" in args.task:
        os.system(
            f"CUDA_VISIBLE_DEVICES={gpu_id_str} python scripts/prune_one_layer.py"
            f" --config ./configs/prune_one_layer_diff_tasks/{args.model}_{args.task}.yaml"
            f" --gpu-id 0 -d {dim}"
            f" --workdir workdirs/prune_one_layer/{args.model}/{args.task}"
        )
    else:
        os.system(
            f"CUDA_VISIBLE_DEVICES={gpu_id_str} python scripts/prune_one_layer.py"
            f" --config ./configs/prune_one_layer_diff_tasks/{args.model}_lm_eval.yaml"
            f" --gpu-id 0 -d {dim}"
            f" --workdir workdirs/prune_one_layer/{args.model}/{args.task}"
            f" --cfg-options test_cfg.evaluator.lm_eval_cfg.tasks=['{args.task}']"
        )


if __name__ == "__main__":
    main()
