from typing import List


def middle_attn_layer_pruning(
    target_layers: List[str],
    start: int = -1,
    length: int = -1,
):
    """prune attention blocks in the middle."""
    # remove all non attention layers
    not_target_layers = []
    for layer in target_layers:
        if "self_attn" not in layer:
            not_target_layers.append(layer)
    for layer in not_target_layers:
        target_layers.remove(layer)

    # set length to maximal length if not set
    if length == -1:
        length = len(target_layers)

    # stop if length is longer than the number of target layers
    if length > len(target_layers):
        return None

    # find starting layer
    if start == -1:
        start = len(target_layers)
    else:
        assert start > 0, "start must be positive"
        start += 1
    end = start - length
    if end < 0:
        return None

    # prune from later to earlier, hence reversed order
    pruned_layers = target_layers[end:start]
    return pruned_layers


# TODO: move functions from layer_prune.py to here
