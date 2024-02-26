import pytest
import torch
from torch import nn

from dissect.dissectors import ForwardADExtractor, BackwardADExtractor, WeightExtractor, ActivationExtractor, Dissector


class MLP(nn.Module):
    def __init__(self, num_units_hidden=1024, num_classes=10):
        super().__init__()
        self.linear_1 = nn.Linear(784, num_units_hidden)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(num_units_hidden, num_units_hidden)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(num_units_hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        out = self.relu_1(self.linear_1(x))
        logits = self.linear_3(self.relu_2(self.linear_2(out)))
        return logits


def prepare_model_and_data(cuda_id=2):
    # for model training on MNIST, initialize model and data loader
    train_kwargs = {"batch_size": 256}
    test_kwargs = {"batch_size": 1000}
    use_cuda = True
    device = torch.device(f"cuda:{cuda_id}" if use_cuda else "cpu")
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = MLP()
    model.to(device)
    input_tensor = torch.rand(256, 1, 28, 28).to(device)
    return model, input_tensor


def test_dissect_forward():
    model, input_tensor = prepare_model_and_data()

    # get input gradient
    forward_ad_extractor = ForwardADExtractor(model)
    input_grads = forward_ad_extractor.forward_ad(input_tensor)
    assert len(input_grads) == len(forward_ad_extractor.hook_list) == len(forward_ad_extractor.hook_objs)
    for _, input_grad in input_grads.items():
        assert input_grad.shape[0] == input_tensor.shape[0]


def test_dissect_backward():
    model, input_tensor = prepare_model_and_data()
    backward_ad_extractor = BackwardADExtractor(model)
    output_backward_grads = backward_ad_extractor.backward_ad(input_tensor)
    assert len(output_backward_grads) == len(backward_ad_extractor.hook_list) == len(backward_ad_extractor.hook_objs)


def test_dissect_weight():
    model, _ = prepare_model_and_data()
    weight_extractor = WeightExtractor(model)

    # get weights
    weights_and_biases = weight_extractor.extract_weights()
    weights = weights_and_biases["weights"]
    biases = weights_and_biases["biases"]
    assert len(weights) == len(biases)
    for name, layer in weight_extractor.layers.items():
        assert torch.allclose(layer.weight.data.detach().cpu(), weights[name])
        if biases[name] is not None:
            assert torch.allclose(layer.bias.data.detach().cpu(), biases[name])
        else:
            assert not hasattr(layer, "bias")


def test_dissect_activation():
    model, input_tensor = prepare_model_and_data()

    # get activations
    activation_extractor = ActivationExtractor(model)
    activations = activation_extractor.extract_activations(input_tensor)
    assert len(activations) == len(activation_extractor.layers)


def test_dissector():
    model, input_tensor = prepare_model_and_data()
    dissector = Dissector(model)

    # check all submodules have identical layers
    for name, layer in dissector.layers.items():
        assert layer == dissector.forward_ad_extractor.layers[name]
        assert layer == dissector.backward_ad_extractor.layers[name]
        assert layer == dissector.activation_extractor.layers[name]
        assert layer == dissector.weight_extractor.layers[name]

    # check dissect results
    input_tangent = torch.rand_like(input_tensor)
    output_tangent = torch.rand(256, 10).to(input_tensor.device)
    dissect_ret = dissector.dissect(input_tensor, input_tangent, output_tangent)
    assert "weights" in dissect_ret.keys()
    assert "activations" in dissect_ret.keys()
    assert "forward_grads" in dissect_ret.keys()
    assert "backward_grads" in dissect_ret.keys()
