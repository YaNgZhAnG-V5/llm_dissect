import os
from typing import List

import matplotlib.pyplot as plt
import yaml
from alive_progress import alive_bar
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


def all_in_one(models: List[str], tasks: List[str], plot_type: str = "bar"):
    fig, axs = plt.subplots(len(models), len(tasks), figsize=(32, 15))
    with alive_bar(len(models) * len(tasks)) as bar:
        for idx_model, model in enumerate(models):
            for idx_task, task in enumerate(tasks):
                ax = axs[idx_model][idx_task]
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
                ax.set_xlabel("Layer ID", fontsize=16)
                ax.set_ylabel("Accuracy", fontsize=16)

                # Custom legend handles
                legend_elements = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=10, label="FFN"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=10, label="Attention"),
                ]
                ax.legend(handles=legend_elements, loc="lower right")
                if idx_model == 0:
                    ax.set_title(f"{task_title[task]}", fontsize=26, fontweight="bold")
                bar()
    fig.text(0.02, 0.84, "Llama3 8B", va="center", rotation="vertical", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.5, "Llama3 70B", va="center", rotation="vertical", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.17, "Mixtral 8x7B", va="center", rotation="vertical", fontsize=24, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig("./workdirs/all_models_all_tasks.pdf")
    plt.close()


def one_model(model: str, tasks: List[str], plot_type: str = "bar"):
    model_caption = {
        "llama3_8b": "Llama3 8B",
        "llama3_70b": "Llama3 70B",
        "mixtral_8x7b": "Mixtral 8x7B",
    }
    assert len(tasks) % 2 == 0, "Number of tasks should be even"
    columns = len(tasks) // 2
    fig, axs = plt.subplots(2, columns, figsize=(32, 12))
    with alive_bar(len(tasks)) as bar:
        for idx_task, task in enumerate(tasks):
            row, col = divmod(idx_task, columns)
            ax = axs[row][col]
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
            text_position = int(len(keys) * 0.5)
            ax.text(
                text_position,
                original_performance + 0.06,
                "Original Performance",
                color="k",
                fontsize=18,
                verticalalignment="top",
            )
            ax.axhline(y=task_random_performance[task], color="k", linestyle="--", linewidth=1)
            ax.text(
                text_position,
                task_random_performance[task] + 0.04,
                "Random Performance",
                color="k",
                fontsize=18,
                verticalalignment="top",
            )
            ax.set_ylim(0.2, 0.95)
            ax.set_xlabel("Layer ID", fontsize=16)
            ax.set_ylabel("Accuracy", fontsize=16)

            # Custom legend handles
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=14, label="FFN"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=14, label="Attention"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            bar()
            ax.set_title(f"{task_title[task]}", fontsize=24)
    plt.suptitle(f"{model_caption[model]}", fontsize=32, fontweight="bold")
    # plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.tight_layout()
    save_dir = "./workdirs/llm_layer_ablation_plot/one_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + f"/{model}_all_tasks.pdf")
    plt.close()


if __name__ == "__main__":
    models = ["llama3_8b", "llama3_70b", "mixtral_8x7b"]
    tasks = ["boolq", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]

    # # all in one plot
    # all_in_one(models, tasks)

    # one model per plot
    for model in models:
        one_model(model, tasks)
