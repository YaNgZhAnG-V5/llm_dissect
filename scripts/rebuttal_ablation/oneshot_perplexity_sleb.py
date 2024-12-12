import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", nargs="+", type=int, help="GPU ID.")
    parser.add_argument("--model", type=str, default="llama3_8b", help="Model name.")
    parser.add_argument("--perplexity", action="store_true", help="Whether to prune based on perplexity.")
    parser.add_argument("--one-shot", action="store_true", help="Whether to prune based on one-shot.")
    parser.add_argument("--exclude", action="store_true", help="Whether to exclude the first 40%.")
    parser.add_argument("--sleb", action="store_true", help="Whether to perform SLEB.")
    parser.add_argument("--prune", action="store_true", help="Whether to prune.")
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate.")
    parser.add_argument("--eval-mmlu", action="store_true", help="Whether to evaluate mmlu.")
    parser.add_argument(
        "--workdir", "-w", type=str, default="workdirs/ablation/perplexity", help="Path to save the result."
    )
    args = parser.parse_args()
    device_str = ",".join([str(i) for i in args.device])
    perplexity_str = "_perp" if args.perplexity else ""
    exclude_str = " -x 0.4" if args.exclude else ""
    one_shot_str = " -p loss_one_shot" if args.one_shot else ""
    if args.model == "llama3_70b" or args.model == "llama2_70b":
        dimension = 8192
    else:
        dimension = 4096
    # perform pruning
    if args.prune:
        if args.sleb:
            # sleb use perplexity, perform one-shot pruning on decoders, without excluding the first 40%
            level = "moe_decoder" if args.model == "mixtral_8x7b" else "decoder"
            os.system(
                f"CUDA_VISIBLE_DEVICES={device_str} python scripts/prune_layer.py -d {dimension} "
                f"--config configs/{args.model}_perp.yaml -lv {level} -w {args.workdir}"
            )
        else:
            os.system(
                f"CUDA_VISIBLE_DEVICES={device_str} python scripts/prune_layer.py -d {dimension} "
                f"--config configs/{args.model}{perplexity_str}.yaml{exclude_str}{one_shot_str} -w {args.workdir}"
            )

    # perform zero-shot evaluation
    if args.eval:
        os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python scripts/eval_pruned_models.py "
            f"--config configs/{args.model}_perp.yaml -p {args.workdir} "
            "--cfg-options test_cfg.testing_manager.in_place=True test_cfg.eval_original=False"
        )

    # perform one-shot evaluation (mmlu)
    if args.eval_mmlu:
        os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python scripts/eval_pruned_models.py "
            f"--config configs/{args.model}_mmlu.yaml -p {args.workdir} "
            "--cfg-options test_cfg.testing_manager.in_place=True test_cfg.eval_original=False"
        )


if __name__ == "__main__":
    main()
