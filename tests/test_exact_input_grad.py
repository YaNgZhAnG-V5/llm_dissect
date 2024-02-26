import pytest
import torch
from torch import nn
from torchvision.models import vgg16

from dissect.prototypes.exact_input_grad import input_grad
from dissect.prototypes.forward_grad import DualHookRegister


class MLP(nn.Module):
    def __init__(self, num_units_hidden=2, num_classes=1):
        super().__init__()
        self.linear_1 = nn.Linear(1, num_units_hidden)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(num_units_hidden, num_units_hidden)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(num_units_hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        out = self.relu_1(self.linear_1(x))
        logits = self.linear_3(self.relu_2(self.linear_2(out)))
        return logits


def test_input_grad():
    device = torch.device("cuda:1")

    # get model with designed parameters
    model = MLP()
    model.linear_1.weight = torch.nn.Parameter(torch.tensor([[1.0], [1.0]]))
    model.linear_1.bias = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    model.linear_2.weight = torch.nn.Parameter(torch.tensor([[1.0, 4.0], [3.0, 2.0]]))
    model.linear_2.bias = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    model.linear_3.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
    model.linear_3.bias = torch.nn.Parameter(torch.tensor([0.0]))
    model.to(device)

    # get input gradient
    layers = [model.linear_1, model.linear_2, model.linear_3]
    hook_list = []
    hook_objs = []
    for layer in layers:
        register_dual_hook = DualHookRegister()
        hook_objs.append(register_dual_hook)
        hook_list.append(layer.register_forward_hook(register_dual_hook()))
    input_grads = input_grad(model, torch.tensor([[1.0]]).to(device), hook_objs)
    assert input_grads[0].shape == (2,)
    assert input_grads[1].shape == (2,)
    assert input_grads[2].shape == (1,)
    assert torch.allclose(input_grads[0], torch.tensor([1.0, 1.0]).to(device))
    assert torch.allclose(input_grads[1], torch.tensor([5.0, 5.0]).to(device))
    assert torch.allclose(input_grads[2], torch.tensor([15.0]).to(device))


def test_input_grad_cnn():
    model = vgg16()
    device = torch.device("cuda:1")
    model.to(device)
    layers = [model.features[0]]
    hook_list = []
    hook_objs = []
    for layer in layers:
        register_dual_hook = DualHookRegister()
        hook_objs.append(register_dual_hook)
        hook_list.append(layer.register_forward_hook(register_dual_hook()))
    input_grads = input_grad(model, torch.randn(1, 3, 224, 224).to(device), hook_objs)
    for grad in input_grads:
        print(grad.shape)
