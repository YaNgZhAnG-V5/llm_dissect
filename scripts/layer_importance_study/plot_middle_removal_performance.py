import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_log_perp(text):
    # Regular expression to match the table rows
    row_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        + r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        + r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )

    # Find the header row to get the keys
    header_match = re.search(
        r"\|\s*(desired)\s*\|\s*(sparsity within)\s*\|\s*(sparsity)\s*\|"
        + r"\s*(mean)\s*\|\s*(main)\s*\|\s*(lambada)\s*\|\s*(lambada)\s*\|"
        + r"\s*(boolq)\s*\|\s*(arc)\s*\|\s*(arc)\s*\|\s*(winogrande)\s*\|"
        + r"\s*(hellaswag)\s*\|\s*(piqa)\s*\|\s*(openbookqa)\s*\|",
        text,
    )

    if not header_match:
        raise ValueError("Header row not found in the text")

    keys = [key.strip() for key in header_match.groups()]

    # Find all data rows
    matches = row_pattern.findall(text)

    # Convert matches to list of dictionaries
    results = []
    for match in matches:
        row_dict = {key: float(value) for key, value in zip(keys, match)}
        results.append(row_dict)

    # Sort the results by the 'desired' key (pruning ratio)
    results.sort(key=lambda x: x["desired"])

    return results


def parse_log_mmlu(text):
    # Regular expression to match the table rows
    row_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)" + r"\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )

    # Find the header row to get the keys
    header_match = re.search(
        r"\|\s*(desired)\s*\|\s*(sparsity within)\s*\|"
        + r"\s*(sparsity)\s*\|\s*(mean)\s*\|\s*(main)\s*\|\s*(mmlu)\s*\|",
        text,
    )

    if not header_match:
        raise ValueError("Header row not found in the text")

    keys = [key.strip() for key in header_match.groups()]

    # Find all data rows
    matches = row_pattern.findall(text)

    # Convert matches to list of dictionaries
    results = []
    for match in matches:
        row_dict = {key: float(value) for key, value in zip(keys, match)}
        results.append(row_dict)

    # Sort the results by the 'desired' key (pruning ratio)
    results.sort(key=lambda x: x["desired"])

    return results


def get_summarized_data(path: str):
    # load data from yaml file
    files = [file for file in os.listdir(path) if file.endswith(".yaml")]
    target_file = None
    for file in files:
        if "perp" in file:
            target_file = "_".join(file.split("_")[:2]) + ".log"
    assert target_file is not None, "No target file found"

    with open(os.path.join(path, target_file), "r") as file:
        log_text = file.read()

    # Calling the function with the example table text
    parsed_data_perp = parse_log_perp(log_text)

    target_file = None
    for file in files:
        if "mmlu" in file:
            target_file = "_".join(file.split("_")[:2]) + ".log"
    assert target_file is not None, "No target file found"

    with open(os.path.join(path, target_file), "r") as file:
        log_text = file.read()

    # Calling the function with the example table text
    parsed_data_mmlu = parse_log_mmlu(log_text)

    # post process
    lengths = []
    for i, _ in enumerate(parsed_data_perp):
        lengths.append(int(parsed_data_perp[i]["desired"]))

    # remove lambada in parsed_data_perp
    for i, _ in enumerate(parsed_data_perp):
        parsed_data_perp[i].pop("lambada")

    # remove other metrics
    for parsed_data in [parsed_data_perp, parsed_data_mmlu]:
        for i, _ in enumerate(parsed_data):
            parsed_data[i].pop("main")
            parsed_data[i].pop("sparsity within")
            parsed_data[i].pop("sparsity")
            parsed_data[i].pop("mean")
            parsed_data[i].pop("desired")

    # get summarized
    summarized_data = []
    for i, _ in enumerate(parsed_data_perp):
        data = 0.0
        for key in parsed_data_perp[i]:
            data += parsed_data_perp[i][key]
        data += parsed_data_mmlu[i]["mmlu"]
        data /= len(parsed_data_perp[i].keys()) + 1
        summarized_data.append(data)

    return lengths, summarized_data


def plot_heatmap(data):
    # Get all unique values for start and end
    starts = sorted(data.keys())
    ends = sorted(set(end for start_dict in data.values() for end in start_dict.keys()), reverse=True)

    # Create a 2D numpy array to hold the values
    heatmap_data = np.zeros((len(ends), len(starts)))

    # Fill the array with values from the dictionary
    for i, end in enumerate(ends):
        for j, start in enumerate(starts):
            if end in data[start]:
                heatmap_data[i, j] = data[start][end]
            else:
                heatmap_data[i, j] = np.nan  # Use NaN for missing values

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(starts)))
    ax.set_yticks(np.arange(len(ends)))
    ax.set_xticklabels(starts)
    ax.set_yticklabels(ends)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    # Add labels and title
    ax.set_xlabel("Start")
    ax.set_ylabel("End")
    ax.set_title("2D Heatmap of Values")

    # Loop over data dimensions and create text annotations
    for i in range(len(ends)):
        for j in range(len(starts)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig("workdirs/remove_middle_layers/heatmap.png")


def main():
    results = {}
    starts = [6, 8, 10, 12, 14, 16, 18, 20, 25, 28, 29, 30, 31]
    for start in starts:
        path = f"workdirs/remove_middle_layers/remove_middle_attn_{start}"
        length, data = get_summarized_data(path)

        # create a dictionary and put the dictionary to results
        dict_to_add = {}
        for length, data in zip(length, data):
            dict_to_add[start - length] = data
        results[start] = dict_to_add

    # plot a 2d heatmap
    plot_heatmap(results)


if __name__ == "__main__":
    main()
