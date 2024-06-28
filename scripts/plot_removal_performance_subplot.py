from typing import List

import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

task_random_performance = {
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "boolq": 0.5,
    "openbookqa": 0.25,
    "piqa": 0.5,
    "winogrande": 0.5,
}

task_title = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "piqa": "PIQA",
    "winogrande": "Winogrande",
}


def main(model: str, tasks: List[str], plot_type: str = "bar"):
    fig, axs = plt.subplots(1, 6, figsize=(32, 5))

    for ax, task in zip(axs, tasks):
        # load data from yaml file
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

        colors = ["red" if i % 2 == 0 else "blue" for i in range(len(keys))]
        if plot_type == "bar":
            height = [data[key] for key in keys]
            height = [i - original_performance for i in height]
            bottom_value = [original_performance] * len(keys)
            ax.bar(range(len(keys)), height, bottom=bottom_value, color=colors, label="Layer Importance")
        elif plot_type == "scatter":
            ax.scatter(range(len(keys)), [data[key] for key in keys], c=colors, label="Layer Importance")
        else:
            raise ValueError(f"Unsupported plot type {plot_type}")
        ax.axhline(y=original_performance, color="k", linestyle="--", linewidth=1)
        text_position = int(len(keys) * 0.3)
        ax.text(
            text_position,
            original_performance + 0.04,
            "Original Performance",
            color="k",
            fontsize=12,
            verticalalignment="top",
        )
        ax.axhline(y=task_random_performance[task], color="k", linestyle="--", linewidth=1)
        ax.text(
            text_position,
            task_random_performance[task] + 0.04,
            "Random Performance",
            color="k",
            fontsize=12,
            verticalalignment="top",
        )
        ax.set_ylim(0.2, 0.95)
        ax.set_xlabel("Layer ID")
        ax.set_ylabel("Accuracy")

        # Custom legend handles
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=10, label="FFN"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=10, label="Attention"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        ax.set_title(f"{task_title[task]}")
    plt.tight_layout()
    plt.savefig(f"./workdirs/{model}_all.png")
    plt.close()


if __name__ == "__main__":
    model = "mixtral_8x7b"
    tasks = ["boolq", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    main(model, tasks)
