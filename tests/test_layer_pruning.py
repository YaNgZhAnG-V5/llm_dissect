from itertools import product

from dissect.pruners import middle_attn_layer_pruning


def test_middle_attn_layer_pruning():
    # create a target layer list
    target_layers = [f"{module}_{i}" for module, i in product(["mlp", "self_attn"], range(32))]

    # start = -1, various length
    for length in [4, 10, 20]:
        layers = middle_attn_layer_pruning(target_layers, -1, length)
        assert layers == [f"self_attn_{i}" for i in range(32 - length, 32)]
        assert len(layers) == length

    # start = -1, length = -1
    layers = middle_attn_layer_pruning(target_layers, -1, -1)
    assert layers == [f"self_attn_{i}" for i in range(0, 32)]
    assert len(layers) == 32

    # start from some layer, various length
    starts = [22, 26, 30]
    lengths = [5, 10, 20]
    for start, length in product(starts, lengths):
        layers = middle_attn_layer_pruning(target_layers, start, length)
        assert layers == [f"self_attn_{i}" for i in range(start - length + 1, start + 1)]
        assert len(layers) == length

    # start from last ends at beginning
    layers = middle_attn_layer_pruning(target_layers, 31, 32)
    assert layers == [f"self_attn_{i}" for i in range(0, 32)]
    assert len(layers) == 32

    # check for None cases
    layers = middle_attn_layer_pruning(target_layers, 2, 5)
    assert layers is None
    layers = middle_attn_layer_pruning(target_layers, 2, 33)
    assert layers is None


if __name__ == "__main__":
    layers = test_middle_attn_layer_pruning()
