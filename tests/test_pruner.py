import torch

from dissect.pruners.forward_pruner import ForwardPruner


def test_get_target_layers():
    test_fn = ForwardPruner._get_target_layers
    stats = {
        "attn.1.q": torch.zeros(20),
        "attn.1.k": torch.zeros(20),
        "attn.1.v": torch.zeros(10),
        "attn.2.q": torch.zeros(20),
        "attn.2.k": torch.zeros(20),
        "attn.2.v": torch.zeros(10),
        "layer_norm": torch.zeros(10),
        "lm_head": torch.zeros(10),
        "mlp.1.up": torch.zeros(20),
        "mlp.1.down": torch.zeros(20),
        "mlp.2.up": torch.zeros(20),
        "mlp.2.down": torch.zeros(20),
    }
    target_name = "attn"
    group = ["attn", "mlp"]
    exclude_layers = ["layer_norm", "lm_head", "v"]
    result = test_fn(stats, target_name, group, exclude_layers)
    print(result)
    excepted = ["attn.1.q", "attn.1.k", "attn.2.q", "attn.2.k"]
    for exp in excepted:
        assert exp in result
        result.pop(exp)
    assert len(result) == 0

    target_name = "mlp"
    result = test_fn(stats, target_name, group, exclude_layers)
    excepted = ["mlp.1.up", "mlp.1.down", "mlp.2.up", "mlp.2.down"]
    for exp in excepted:
        assert exp in result
        result.pop(exp)
    assert len(result) == 0
