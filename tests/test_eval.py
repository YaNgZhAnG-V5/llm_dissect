import torch
from torch import nn

from dissect.utils.misc import calc_pruned_parameters


class TestModel(nn.Module):
    # a simple 3 layer mlp model for testing
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def test_calc_pruned_parameters():
    model = TestModel()

    # test case 1
    mask_state_dict = {
        "fc1": torch.zeros(20),
        "fc2": torch.zeros(20),
        "fc3": torch.zeros(10),
    }
    log_tabulate, sparsity_target_layers, sparsity_whole_model = calc_pruned_parameters(model, mask_state_dict)
    assert len(log_tabulate) == 3
    assert sparsity_target_layers == 1.0
    assert sparsity_whole_model == 1.0

    # test case 2
    mask_state_dict = {
        "fc1": torch.ones(20),
        "fc2": torch.ones(20),
        "fc3": torch.ones(10),
    }
    log_tabulate, sparsity_target_layers, sparsity_whole_model = calc_pruned_parameters(model, mask_state_dict)
    assert len(log_tabulate) == 3
    assert sparsity_target_layers == 0.0
    assert sparsity_whole_model == 0.0

    # test case 3
    mask_state_dict["fc1"][0] = 0
    mask_state_dict["fc1"][4] = 0
    mask_state_dict["fc1"][17] = 0
    log_tabulate, sparsity_target_layers, sparsity_whole_model = calc_pruned_parameters(model, mask_state_dict)
    assert sparsity_target_layers == 30 / 800.0

    # test case 4
    mask_state_dict["fc2"][0] = 0
    mask_state_dict["fc2"][1] = 0
    mask_state_dict["fc2"][2] = 0
    mask_state_dict["fc2"][3] = 0
    mask_state_dict["fc2"][4] = 0
    log_tabulate, sparsity_target_layers, sparsity_whole_model = calc_pruned_parameters(model, mask_state_dict)
    assert sparsity_target_layers == 130 / 800.0

    # test case 5
    mask_state_dict.pop("fc3")
    log_tabulate, sparsity_target_layers, sparsity_whole_model = calc_pruned_parameters(model, mask_state_dict)
    assert sparsity_whole_model == 130 / 800.0
    assert sparsity_target_layers == 130 / 600.0
