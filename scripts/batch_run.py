import argparse
import os
import os.path as osp
from itertools import product

import mmengine
from matplotlib import pyplot as plt


def main(p_path: str, work_dir: str, gpu_id: int = 0):
    scopes = ["global_thres", "global", "local"]
    strategies = ["forward_grads", "backward_grads", "activations", "weights"]
    for scope, strategy in product(scopes, strategies):
        workdir = osp.join(work_dir, scope, strategy)
        # run analyse
        os.system(
            f"python '/home/bizon/projects/llm_dissect/scripts/analyze_and_prune.py' -p {p_path} --gpu-id {gpu_id}"
            f" --cfg-options pruner.criterion.scope={scope} pruner.criterion.strategy={strategy}"
        )

        # run prune test
        os.system(
            f"python '/home/bizon/projects/llm_dissect/scripts/eval_pruned_models.py'"
            f" --work-dir {workdir} --gpu-id {gpu_id}"
            f" --cfg-options pruner.criterion.scope={scope} pruner.criterion.strategy={strategy}"
        )


def print_summary(work_dir: str):
    scopes = ["global_thres", "global", "local"]
    strategies = ["forward_grads", "backward_grads", "activations", "weights"]
    result_dict = {}
    conditions = None
    for scope, strategy in product(scopes, strategies):
        print("Scope:", scope, "Strategy:", strategy)
        method = f"{scope}+{strategy}"
        workdir = osp.join(work_dir, scope, strategy)
        results = mmengine.load(osp.join(workdir, "test_results.yaml"))
        scores = [f"{x['accuracy']:.3f}" for x in results]
        print("    ".join(scores))
        result_dict[method] = [float(x) for x in scores]
        if conditions is None:
            conditions = [f"{x['sparsity']:.2f}" for x in results]

    # Plotting the performance of each method
    for method, scores in result_dict.items():
        plt.plot(conditions, scores, label=method)

    # Adding title and labels
    plt.title("Pruning Strategy Performance Comparison")
    plt.xlabel("Neuron Sparsity")
    plt.ylabel("Accuracy")
    plt.xticks(conditions)  # This explicitly sets the ticks on the x-axis to match your conditions.
    plt.legend()  # This adds a legend using the labels defined in the plot calls.
    plt.savefig(osp.join(work_dir, "performance.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-path", "-p", type=str, help="Path to the saved stats for prune")
    parser.add_argument("--work-dir", "-w", type=str, help="path to save results")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    args = parser.parse_args()
    main(p_path=args.p_path, work_dir=args.work_dir, gpu_id=args.gpu_id)
    print_summary(work_dir=args.work_dir)
