from typing import List

import numpy as np
import yaml


def get_mean_performance_for_model(model: str, tasks: List[str]):
    # load data from yaml file
    mean_cornerstone_performance_over_task, mean_non_cornerstone_performance_over_task = [], []
    for task in tasks:
        path = f"./workdirs/prune_one_layer/{model}/{task}/prune_one_layer.yaml"
        max_cutoff = 10
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # process data to get relative difference
        for key in data:
            if key == "original":
                continue
            data[key] = round(data[key], 4)
            data[key] = min(data[key], max_cutoff)
        original_performance = data.pop("original")
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
        cornerstone_performance = original_performance - np.array(cornerstone_performance)
        non_cornerstone_performance = original_performance - np.array(non_cornerstone_performance)
        cornerstone_performance = np.clip(cornerstone_performance, 0, 100)
        non_cornerstone_performance = np.clip(non_cornerstone_performance, 0, 100)
        mean_cornerstone_performance = cornerstone_performance.mean()
        mean_non_cornerstone_performance = non_cornerstone_performance.mean()
        mean_cornerstone_performance_over_task.append(mean_cornerstone_performance)
        mean_non_cornerstone_performance_over_task.append(mean_non_cornerstone_performance)
    print(np.mean(mean_cornerstone_performance_over_task))
    print(np.mean(mean_non_cornerstone_performance_over_task))


def get_mean_performance_for_task(models: List[str], task: str):
    # load data from yaml file
    mean_cornerstone_performance_over_task, mean_non_cornerstone_performance_over_task = [], []
    for model in models:
        path = f"./workdirs/prune_one_layer/{model}/{task}/prune_one_layer.yaml"
        max_cutoff = 10
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # process data to get relative difference
        for key in data:
            if key == "original":
                continue
            data[key] = round(data[key], 4)
            data[key] = min(data[key], max_cutoff)
        original_performance = data.pop("original")
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
        cornerstone_performance = original_performance - np.array(cornerstone_performance)
        non_cornerstone_performance = original_performance - np.array(non_cornerstone_performance)
        cornerstone_performance = np.clip(cornerstone_performance, 0, 100)
        non_cornerstone_performance = np.clip(non_cornerstone_performance, 0, 100)
        mean_cornerstone_performance = cornerstone_performance.mean()
        mean_non_cornerstone_performance = non_cornerstone_performance.mean()
        mean_cornerstone_performance_over_task.append(mean_cornerstone_performance)
        mean_non_cornerstone_performance_over_task.append(mean_non_cornerstone_performance)
    print(task)
    # print("cornerstone", np.mean(mean_cornerstone_performance_over_task))
    print("non cornerstone", np.mean(mean_non_cornerstone_performance_over_task))


if __name__ == "__main__":
    models = ["llama3_8b", "llama3_70b", "mixtral_8x7b"]
    tasks = ["boolq", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]

    for model in models:
        get_mean_performance_for_model(model, tasks)

    for model in models:
        print("#############", model)
        for task in tasks:
            get_mean_performance_for_task([model], task)
