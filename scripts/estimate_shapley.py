import argparse
from os import path as osp

import mmengine
import numpy as np
import yaml


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-path",
        "-l",
        type=str,
        default="workdirs/layer_shapley/llama3_8b/winogrande/prune_one_layer.yaml",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        type=str,
        default="workdirs/estimated_layer_shapley/llama3_8b/winogrande",
    )
    parser.add_argument("--level", type=int, default=3, help="level to use")
    args = parser.parse_args()
    return args


def load_data(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def estimate(data: dict, level: int):
    estimated_shapley = {}

    # get the original score
    original_score = data["original"]
    print(f"Original score: {original_score}")

    # remove higher level that we dont want to use
    remove_keys = []
    for key in data.keys():
        if key.count("%") >= level:
            remove_keys.append(key)
    for key in remove_keys:
        data.pop(key)

    # get all target layers
    target_layers = [name for name in data.keys() if "%" not in name]
    target_layers.remove("original")
    print(f"We have {len(target_layers)} target layers")

    # get the shapley value for each target layer
    for layer in target_layers:
        # get original vs remove target layer only
        estimated_shapley[layer] = [original_score - data[layer]]

        # get all candidate results (target at beginning or at the end)
        candidate_results = [name for name in data.keys() if name.startswith(layer + "%") or name.endswith("%" + layer)]
        for candidate_ret in candidate_results:
            if candidate_ret.startswith(layer):
                candidate_ret_exclude_target = candidate_ret.replace(layer + "%", "")
            else:
                candidate_ret_exclude_target = candidate_ret.replace("%" + layer, "")
            score_diff = data[candidate_ret_exclude_target] - data[candidate_ret]
            estimated_shapley[layer].append(score_diff)

        # get the average of all candidate results
        estimated_shapley[layer] = np.mean(estimated_shapley[layer]).item()
        print(f"estimated shapley for layer {layer}: {estimated_shapley[layer]}")
    return estimated_shapley


def save_shapley(estimated_shapley: dict, save_path: str):
    mmengine.mkdir_or_exist(save_path)
    with open(osp.join(save_path, "estimated_shapley.yaml"), "w") as f:
        yaml.dump(estimated_shapley, f)


def main():
    args = arg_parser()
    data = load_data(args.load_path)
    estimated_shapley = estimate(data, args.level)
    save_shapley(estimated_shapley, args.save_path)


if __name__ == "__main__":
    main()
