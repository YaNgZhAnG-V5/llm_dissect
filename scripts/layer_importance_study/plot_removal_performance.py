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


def main(model: str, task: str):
    # load data from yaml file
    path = f"./workdirs/prune_one_layer_old/{model}/{task}/prune_one_layer.yaml"
    max_cutoff = 20
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
    plt.scatter(range(len(keys)), [data[key] for key in keys], c=colors, label="Layer Importance")
    plt.axhline(y=original_performance, color="k", linestyle="--", linewidth=1)
    text_position = int(len(keys) * 0.3)
    plt.text(
        text_position,
        original_performance + 0.04,
        "Original Performance",
        color="k",
        fontsize=12,
        verticalalignment="top",
    )
    if "perplexity" not in task:
        plt.axhline(y=task_random_performance[task], color="k", linestyle="--", linewidth=1)
        plt.text(
            text_position,
            task_random_performance[task] + 0.04,
            "Random Performance",
            color="k",
            fontsize=12,
            verticalalignment="top",
        )
        plt.ylim(0.2, 0.95)
    else:
        plt.ylim(original_performance - 2, max_cutoff)
    plt.xlabel("Layer ID")
    plt.ylabel("Accuracy")

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=10, label="FFN"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=10, label="Attention"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.title(f"Layer Importance for {task} on {model}")
    plt.savefig(f"./workdirs/{model}_{task}.png")
    plt.close()


if __name__ == "__main__":
    model = "llama3_70b"
    tasks = ["perplexity_wikitext"]  # ["arc_easy", "arc_challenge", "boolq", "openbookqa", "piqa", "winogrande"]
    for task in tasks:
        main(model, task)
