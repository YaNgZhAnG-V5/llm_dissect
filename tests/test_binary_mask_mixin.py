import torch

from dissect.pruners.binary_mask_mixin import BinaryMaskMixin


def test_per_head_prune():
    stats = torch.randn(4096)
    prune_ratio = 0.2
    num_heads = 32

    binary_mask = BinaryMaskMixin.per_head_prune(stats, prune_ratio, num_heads)

    assert binary_mask.shape == stats.shape
    reshaped_mask = binary_mask.reshape(num_heads, -1)
    for mask_per_head in reshaped_mask:
        # mask value = 1 means not prune
        assert mask_per_head.sum() == 102
