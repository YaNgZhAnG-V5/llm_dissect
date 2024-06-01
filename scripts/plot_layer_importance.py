import matplotlib.pyplot as plt
import yaml


def main():
    # load data from yaml file
    path = "./workdirs/prune_one_layer_boolq/prune_one_layer.yaml"
    large_better = True
    max_cutoff = 10
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # process data to get relative difference
    for key in data:
        if key == "original":
            continue
        if large_better:
            data[key] = max(data["original"] - data[key], 0)
        else:
            data[key] = max(data[key] - data["original"], 0)
        data[key] = round(data[key], 4)
        data[key] = min(data[key], max_cutoff)
    data.pop("original")

    keys = sorted(list(data.keys()), key=lambda x: int(x.split(".")[-3]))
    for i in range(len(keys) // 2):
        keys[2 * i], keys[2 * i + 1] = keys[2 * i + 1], keys[2 * i]

    colors = ["red" if i % 2 == 0 else "blue" for i in range(len(keys))]
    plt.scatter(range(len(keys)), [data[key] for key in keys], c=colors)
    # plt.xticks(range(len(keys)), keys, rotation=90)
    plt.savefig("./workdirs/debug_boolq.png")


if __name__ == "__main__":
    main()
