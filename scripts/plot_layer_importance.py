from typing import List

import matplotlib.pyplot as plt
import yaml
from alive_progress import alive_bar

task_title = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "piqa": "PIQA",
    "winogrande": "Winogrande",
}


def main(
    models: List[str],
    tasks: List[str],
    large_better: bool = True,
    bar_plot: bool = True,
    grey_later_layers: bool = False,
):
    fig, axs = plt.subplots(len(models), len(tasks), figsize=(32, 15))
    with alive_bar(len(models) * len(tasks)) as bar:
        for idx_model, model in enumerate(models):
            for idx_task, task in enumerate(tasks):
                ax = axs[idx_model][idx_task]
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

                colors = ["red" if i % 2 == 0 else "blue" for i in range(len(keys))]
                # plt.scatter(range(len(keys)), [data[key] for key in keys], c=colors)
                data = [data[key] for key in keys]
                if not bar_plot:
                    for idx, i in enumerate(data):
                        if i < 0:
                            data[idx] = 0

                import matplotlib.colors as mcolors

                norm = mcolors.Normalize(vmin=0, vmax=max(data))
                # Generate colors using a colormap
                import matplotlib.cm as cm

                cmap = cm.viridis
                colors = cmap(norm(data))
                if grey_later_layers:
                    for idx in range(len(colors)):
                        if idx > 7:
                            colors[idx] = (0.8, 0.8, 0.8, 1)
                # Number of top layers to label
                num_top_layers = 4

                # # Sort the data by size
                import numpy as np

                important_indices = np.argsort(data)[::-1][:num_top_layers]
                labels = [keys[i] for i in range(len(keys))]
                label_text = {"mlp": "FFN", "self_attn": "Attn", "block_sparse_moe": "MoE"}
                for idx, i in enumerate(labels):
                    labels[idx] = label_text[i.split(".")[3]] + " " + i.split(".")[2]

                # # Create new labels list with only top layers labeled
                new_labels = [labels[i] if i in important_indices else "" for i in range(len(labels))]

                # Custom autopct function to display percentages only for top-valued data
                used_index = []

                def autopct_func(pct, allvalues, used_index):
                    allvalues = [i / sum(allvalues) * 100 for i in allvalues]
                    for idx, i in enumerate(allvalues):
                        if abs(i - pct) < 0.01:
                            index = idx
                            break
                    if index in important_indices and index not in used_index:
                        used_index.append(index)
                        return f"{pct:.1f}%"
                    else:
                        return ""

                if bar_plot:
                    ax.bar(range(len(data)), data)
                else:
                    wedges, texts, autotexts = ax.pie(
                        data,
                        colors=colors,
                        labels=new_labels,
                        autopct=lambda pct: autopct_func(pct, data, used_index),
                        startangle=90,
                    )
                    # Customizing the font properties
                    for text in texts:
                        text.set_fontsize(14)  # Change font size for labels
                        # text.set_fontweight("bold")  # Make labels bold

                    for autotext in autotexts:
                        autotext.set_fontsize(12)  # Change font size for percentage labels
                        # autotext.set_fontweight("bold")  # Make percentage labels bold
                if idx_model == 0:
                    ax.set_title(task_title[task], fontsize=26, fontweight="bold")
                bar()
    fig.text(0.02, 0.84, "Llama3 8B", va="center", rotation="vertical", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.5, "Llama3 70B", va="center", rotation="vertical", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.17, "Mixtral 8x7B", va="center", rotation="vertical", fontsize=24, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig("./workdirs/shapley_all_models_all_tasks.pdf")
    plt.close()


if __name__ == "__main__":
    models = ["llama3_8b", "llama3_70b", "mixtral_8x7b"]
    tasks = ["boolq", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa"]
    main(models, tasks, large_better=False)
