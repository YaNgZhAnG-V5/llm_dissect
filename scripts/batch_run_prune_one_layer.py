import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    if args.task == "mmlu":
        os.system(
            f"python scripts/prune_one_layer.py --config ./configs/prune_one_layer_diff_tasks/llama3_8b_mmlu.yaml"
            f" --gpu-id {args.gpu_id}"
            f" --workdir workdirs/prune_one_layer_{args.task}"
        )
    else:
        os.system(
            f"python scripts/prune_one_layer.py --config ./configs/prune_one_layer_diff_tasks/llama3_8b_lm_eval.yaml"
            f" --gpu-id {args.gpu_id}"
            f" --workdir workdirs/prune_one_layer_{args.task}"
            f" --cfg-options test_cfg.evaluator.lm_eval_cfg.tasks=['{args.task}']"
        )


if __name__ == "__main__":
    main()
