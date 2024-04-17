import argparse
from os import path as osp

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic stats.")
    parser.add_argument(
        "--ref-path",
        type=str,
        help="Path to the reference file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory.",
    )
    return parser.parse_args()


def gen_synthetic_stats(ref_path: str, output_path: str):
    ref_data = torch.load(ref_path, map_location=torch.device("cpu"))

    assert isinstance(ref_data, dict), "The reference file should be a dictionary."
    zero_stats = {}
    random_stats = {}
    for key in ref_data.keys():
        zero_stats[key] = torch.zeros_like(ref_data[key])
        random_stats[key] = torch.rand_like(ref_data[key])

    print(f"Saving synthetic stats to {output_path}")
    torch.save(zero_stats, osp.join(output_path, "zero_stats.pth"))
    torch.save(random_stats, osp.join(output_path, "random_stats.pth"))


def main():
    args = parse_args()
    gen_synthetic_stats(args.ref_path, args.output_path)


if __name__ == "__main__":
    main()
