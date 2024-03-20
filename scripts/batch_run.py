import os
import os.path as osp
from itertools import product

import mmengine


def main():
    scopes = ["global_thres", "global", "local"]
    strategies = ["forward_grads", "backward_grads", "activations", "weights"]
    for scope, strategy in product(scopes, strategies):
        workdir = osp.join("workdirs/batch_run", scope, strategy)
        # run analyse
        os.system(
            "python '/home/bizon/projects/llm_dissect/scripts/analyze_and_prune.py' -p workdirs/prune_bert"
            f" --cfg-options pruner.criterion.scope={scope} pruner.criterion.strategy={strategy}"
        )

        # run prune test
        os.system(
            "python '/home/bizon/projects/llm_dissect/scripts/test_pruned_models.py' --work-dir {workdir}"
            f" --cfg-options pruner.criterion.scope={scope} pruner.criterion.strategy={strategy}"
        )


def print_summary():
    scopes = ["global_thres", "global", "local"]
    strategies = ["forward_grads", "backward_grads", "activations", "weights"]
    for scope, strategy in product(scopes, strategies):
        print("Scope:", scope, "Strategy:", strategy)
        workdir = osp.join("workdirs/batch_run", scope, strategy)
        results = mmengine.load(osp.join(workdir, "test_results.yaml"))
        print("    ".join([f"{x['accuracy']:.3f}" for x in results]))


if __name__ == "__main__":
    main()
    print_summary()
