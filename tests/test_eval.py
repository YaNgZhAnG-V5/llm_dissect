import torch
from torch import nn

from dissect.pruners import TESTING_MANAGER


# Cannot call it TestModel because pytest will detect TestModel as a test suit.
class DummyModel(nn.Module):
    # a simple 3 layer mlp model for testing.
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def test_calc_pruned_parameters():
    model = DummyModel()
    ori_param_count_dict = {"fc1.weight": 200, "fc2.weight": 400, "fc3.weight": 200}
    testing_manager = TESTING_MANAGER.build(
        {"type": "ForwardPrunerTestingManager", "prune_input": [], "in_place": False}
    )

    # test case 1
    mask_state_dict = {
        "fc1": torch.zeros(20),
        "fc2": torch.zeros(20),
        "fc3": torch.zeros(10),
    }
    log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, mask_state_dict, ori_param_count_dict
    )
    assert len(log_tabulate) == 3
    assert sparsity_target_layers == 1.0
    assert sparsity_whole_model == 1.0

    # test case 2
    mask_state_dict = {
        "fc1": torch.ones(20),
        "fc2": torch.ones(20),
        "fc3": torch.ones(10),
    }
    log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, mask_state_dict, ori_param_count_dict
    )
    assert len(log_tabulate) == 3
    assert sparsity_target_layers == 0.0
    assert sparsity_whole_model == 0.0

    # test case 3
    mask_state_dict["fc1"][0] = 0
    mask_state_dict["fc1"][4] = 0
    mask_state_dict["fc1"][17] = 0
    log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, mask_state_dict, ori_param_count_dict
    )
    assert sparsity_target_layers == 30 / 800.0

    # test case 4
    mask_state_dict["fc2"][0] = 0
    mask_state_dict["fc2"][1] = 0
    mask_state_dict["fc2"][2] = 0
    mask_state_dict["fc2"][3] = 0
    mask_state_dict["fc2"][4] = 0
    log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, mask_state_dict, ori_param_count_dict
    )
    assert sparsity_target_layers == 130 / 800.0

    # test case 5
    mask_state_dict.pop("fc3")
    copied_mask = mask_state_dict.copy()
    log_tabulate, sparsity_target_layers, sparsity_whole_model = testing_manager.calc_pruned_parameters(
        model, mask_state_dict, ori_param_count_dict
    )
    assert sparsity_whole_model == 130 / 800.0
    assert sparsity_target_layers == 130 / 600.0
    for k in mask_state_dict.keys():
        assert torch.all(mask_state_dict[k] == copied_mask[k])


def test_merge_mask():
    testing_manager = TESTING_MANAGER.build(
        {"type": "ForwardPrunerTestingManager", "prune_input": [], "in_place": False}
    )

    # test case 1
    mask_state_dict = {
        "model.attn.q_proj": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "model.attn.k_proj": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        "fc3": torch.ones(10),
    }
    mask_state_dict = testing_manager.merge_mask(mask_state_dict)
    assert mask_state_dict["model.attn.q_proj"].sum() == 0
    assert mask_state_dict["model.attn.k_proj"].sum() == 0
    assert mask_state_dict["fc3"].sum() == 10

    # test case 2
    mask_state_dict = {
        "model.attn.q_proj": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "model.attn.k_proj": torch.tensor([1, 0, 1, 0, 1, 1, 1, 1, 1, 1]),
        "fc3": torch.ones(10),
    }
    mask_state_dict = testing_manager.merge_mask(mask_state_dict)
    assert mask_state_dict["model.attn.q_proj"].sum() == 3
    assert mask_state_dict["model.attn.k_proj"].sum() == 3
    assert mask_state_dict["fc3"].sum() == 10
