from typing import List

import numpy as np
import yaml

task_title = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "piqa": "PIQA",
    "winogrande": "Winogrande",
}


def main(
    model: str,
    tasks: List[str],
):
    cornerstone_performance_over_tasks, non_cornerstone_performance_over_tasks = [], []
    for task in tasks:
        # load data from yaml file
        path = f"./workdirs/estimated_layer_shapley/{model}/{task}/estimated_shapley.yaml"
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # process data to get relative difference
        for key in data:
            data[key] = round(data[key], 4)
        if model == "mixtral_8x7b":
            suffix_position = -2
        else:
            suffix_position = -3
        keys = sorted(list(data.keys()), key=lambda x: int(x.split(".")[suffix_position]))
        for i in range(len(keys) // 2):
            keys[2 * i], keys[2 * i + 1] = keys[2 * i + 1], keys[2 * i]

        # get average of performance drop for cornerstone and non-cornerstone layers
        if model == "llama3_8b":
            cornerstone_layers = ["layers.0.self_attn", "layers.0.mlp", "layers.1.mlp"]
        elif model == "llama3_70b":
            cornerstone_layers = ["layers.0.self_attn", "layers.0.mlp", "layers.3.mlp"]
        elif model == "mixtral_8x7b":
            cornerstone_layers = [
                "layers.0.self_attn",
                "layers.0.block_sparse_moe",
                "layers.1.block_sparse_moe",
            ]
        else:
            raise ValueError(f"Unsupported model {model}")
        cornerstone_performance, non_cornerstone_performance = [], []
        for key in keys:
            is_cornerstone = False
            for layer in cornerstone_layers:
                if layer in key:
                    cornerstone_performance.append(data[key])
                    is_cornerstone = True
                    break
            if not is_cornerstone:
                non_cornerstone_performance.append(data[key])
        assert len(cornerstone_performance) == len(cornerstone_layers)
        assert len(non_cornerstone_performance) == len(keys) - len(cornerstone_layers)
        for idx, i in enumerate(cornerstone_performance):
            if i < 0:
                cornerstone_performance[idx] = 0
        for idx, i in enumerate(non_cornerstone_performance):
            if i < 0:
                non_cornerstone_performance[idx] = 0
        sum_cornerstone_performance = np.sum(cornerstone_performance)
        sum_non_cornerstone_performance = np.sum(non_cornerstone_performance)
        sum_total = sum_cornerstone_performance + sum_non_cornerstone_performance
        cornerstone_performance_over_tasks.append(sum_cornerstone_performance / sum_total)
        non_cornerstone_performance_over_tasks.append(sum_non_cornerstone_performance / sum_total)
    print("##########", model)
    print(np.mean(cornerstone_performance_over_tasks))
    print(np.mean(non_cornerstone_performance_over_tasks))


if __name__ == "__main__":
    models = ["llama3_8b", "llama3_70b", "mixtral_8x7b"]
    tasks = ["boolq", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa"]
    for model in models:
        main(model, tasks)
