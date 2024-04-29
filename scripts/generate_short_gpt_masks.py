import mmengine
import torch

layer_dim = 4096

layer_ids = [
    27,
    26,
    25,
    28,
    24,
    29,
    23,
    21,
    22,
    30,
    19,
    20,
    18,
    17,
    16,
    11,
    12,
    13,
    14,
    15,
    10,
    9,
    8,
    7,
    6,
    5,
    3,
    2,
    4,
    1,
    31,
    0,
]
assert len(layer_ids) == 32
for i in range(32):
    assert i in layer_ids
print(f"Layer ids: {layer_ids}")
pruning_ratio = [int(i * 32 / 10) for i in range(1, 10)]
print(f"Actual pruning ratio: {[i / 32 for i in pruning_ratio]}")
mmengine.mkdir_or_exist("short_gpt_masks")
for idx, ratio in enumerate(pruning_ratio):
    pruning_layer_ids = layer_ids[:ratio]
    print(f"Pruning {len(pruning_layer_ids)} layer")
    pruned_layers = [f"model.layers.{i}.self_attn.o_proj" for i in pruning_layer_ids]
    pruned_layers += [f"model.layers.{i}.mlp.down_proj" for i in pruning_layer_ids]
    print(pruned_layers)
    assert len(pruned_layers) == 2 * ratio
    mask_state_dict = {pruned_layer: torch.zeros(layer_dim, dtype=torch.bool) for pruned_layer in pruned_layers}
    # print(mask_state_dict)
    torch.save(mask_state_dict, f"./short_gpt_masks/sparsity_0_{idx + 1}_pruning_masks.pth")
