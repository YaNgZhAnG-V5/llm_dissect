from typing import List

import matplotlib.pyplot as plt
import yaml

task_title = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "piqa": "PIQA",
    "winogrande": "Winogrande",
}


def main(tasks: List[str], large_better: bool = True):
    fig, axs = plt.subplots(1, 6, figsize=(32, 5))

    for ax, task in zip(axs, tasks):
        # load data from yaml file
        path = f"./workdirs/estimated_layer_shapley/llama3_8b/{task}/estimated_shapley.yaml"
        max_cutoff = 10
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # process data to get relative difference
        for key in data:
            # if key == "original":
            #     continue
            # if large_better:
            #     data[key] = max(data["original"] - data[key], 0)
            # else:
            #     data[key] = max(data[key] - data["original"], 0)
            data[key] = round(data[key], 4)

        keys = sorted(list(data.keys()), key=lambda x: int(x.split(".")[-3]))
        for i in range(len(keys) // 2):
            keys[2 * i], keys[2 * i + 1] = keys[2 * i + 1], keys[2 * i]

        colors = ["red" if i % 2 == 0 else "blue" for i in range(len(keys))]
        # plt.scatter(range(len(keys)), [data[key] for key in keys], c=colors)
        data = [data[key] for key in keys]
        for idx, i in enumerate(data):
            if i < 0:
                data[idx] = 0

        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=min(data), vmax=max(data))
        # Generate colors using a colormap
        import matplotlib.cm as cm

        cmap = cm.viridis
        colors = cmap(norm(data))
        # # Number of top layers to label
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

        ax.set_title(task_title[task], fontsize=16)
        # plt.xticks(range(len(keys)), keys, rotation=90)

        # if task == "perplexity":
        #     plt.ylim(0, max_cutoff)
        # else:
        #     plt.ylim(-0.5, 0.5)
        # plt.title(f"Layer Importance for {task}")
    plt.tight_layout()
    plt.savefig("./workdirs/shapley_llama3_8b_all.png")
    plt.close()


if __name__ == "__main__":
    tasks = ["boolq", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa"]
    main(tasks, large_better=False)
