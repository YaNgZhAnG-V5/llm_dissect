import argparse
import os
from collections import namedtuple

import torch

ModelSpec = namedtuple("ModelSpec", ["name", "num_layers", "mlp_params", "attn_params", "total_params"])
# parameters all all in unit k
llama_7b_spec = ModelSpec("llama_7b", 32, 135270.4, 67112.96, 6738415.616)
llama_13b_spec = ModelSpec("llama_13b", 40, 212341.76, 104862.72, 13015864.32)
llama_70b_spec = ModelSpec("llama_70b", 80, 704651.264, 151003.136, 68976648.192)
llama3_8b_spec = ModelSpec("llama3_8b", 32, 176164.864, 41947.136, 8030261.248)
llama3_70b_spec = ModelSpec("llama3_70b", 80, 704651.264, 151003.136, 70553706.496)
specs = [llama_7b_spec, llama_13b_spec, llama_70b_spec, llama3_8b_spec, llama3_70b_spec]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="path to load masks.",
    )
    parser.add_argument("--model", "-m", type=str, default="llama_7b", help="with model to eval.")
    args = parser.parse_args()
    for spec in specs:
        if spec.name == args.model:
            model_spec = spec
            break
    for file in sorted(os.listdir(args.path), key=lambda x: float(".".join(["0", x.split("_")[2]]))):
        data = torch.load(os.path.join(args.path, file), map_location="cpu")
        actual_rate = 0.0
        for key in data.keys():
            if "attn" in key:
                actual_rate += model_spec.attn_params
            elif "mlp" in key:
                actual_rate += model_spec.mlp_params
        actual_rate_decoder = actual_rate / (model_spec.num_layers * (model_spec.attn_params + model_spec.mlp_params))
        actual_rate /= model_spec.total_params
        print(file)
        print(f"Sparsity rate within decoder layers: {actual_rate_decoder:4f}")
        print(f"Sparsity rate: {actual_rate:4f}")


if __name__ == "__main__":
    main()
