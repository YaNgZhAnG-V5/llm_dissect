import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", nargs="+", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--shapley", action="store_true")
    args = parser.parse_args()
    if "8b" in args.model:
        dim = 4096
    elif "70b" in args.model:
        dim = 8192
    elif "8x7b" in args.model:
        dim = 4096
    else:
        raise ValueError(f"Unsupported model {args.model}")
    gpu_id_str = ",".join(args.gpu_id)
    if args.shapley:
        script_name = "layer_shapley"
    else:
        script_name = "prune_one_layer"
    if (args.task == "mmlu") or (args.task == "mmlu_zeroshot") or ("perplexity" in args.task):
        os.system(
            f"CUDA_VISIBLE_DEVICES={gpu_id_str} python scripts/{script_name}.py"
            f" --config ./configs/prune_one_layer_diff_tasks/{args.model}_{args.task}.yaml"
            f" --gpu-id 0 -d {dim}"
            f" --workdir workdirs/{script_name}/{args.model}/{args.task}"
        )
    else:
        os.system(
            f"CUDA_VISIBLE_DEVICES={gpu_id_str} python scripts/{script_name}.py"
            f" --config ./configs/prune_one_layer_diff_tasks/{args.model}_lm_eval.yaml"
            f" --gpu-id 0 -d {dim}"
            f" --workdir workdirs/{script_name}/{args.model}/{args.task}"
            f" --cfg-options test_cfg.evaluator.lm_eval_cfg.tasks=['{args.task}']"
        )


if __name__ == "__main__":
    main()
