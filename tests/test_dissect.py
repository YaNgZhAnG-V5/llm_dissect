import torch
from torch import nn

from dissect.dissectors import ActivationExtractor, BackwardADExtractor, Dissector, ForwardADExtractor, WeightExtractor


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


def prepare_model_and_data(gpu_id=0):
    # for model training on MNIST, initialize model and data loader
    train_kwargs = {'batch_size': 256}
    test_kwargs = {'batch_size': 1000}
    # if gpu_id < 0, then use CPU
    use_cuda = gpu_id >= 0
    device = torch.device(f'cuda:{gpu_id}' if use_cuda else 'cpu')
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = MLP()
    model.to(device)
    input_tensor = torch.rand(256, 1, 28, 28).to(device)
    return model, input_tensor


def test_dissect_forward(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)

    # get input gradient
    forward_ad_extractor = ForwardADExtractor(model)
    input_grads = forward_ad_extractor.forward_ad(input_tensor)
    assert len(input_grads) == len(forward_ad_extractor.hook_handles) == len(forward_ad_extractor.hook_registers)
    for _, input_grad in input_grads.items():
        assert input_grad.shape[0] == input_tensor.shape[0]


def test_dissect_backward(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)
    backward_ad_extractor = BackwardADExtractor(model)
    output_backward_grads = backward_ad_extractor.backward_ad(input_tensor)
    assert len(output_backward_grads) == len(backward_ad_extractor.hook_handles) == len(backward_ad_extractor.hook_registers)


def test_dissect_weight(gpu_id):
    model, _ = prepare_model_and_data(gpu_id)
    weight_extractor = WeightExtractor(model)

    # get weights
    weights, biases = weight_extractor.extract_weights_biases()
    assert len(weights) == len(biases)
    for name, layer in weight_extractor.layers.items():
        assert torch.allclose(layer.weight.data.detach().cpu(), weights[name])
        if biases[name] is not None:
            assert torch.allclose(layer.bias.data.detach().cpu(), biases[name])
        else:
            assert not hasattr(layer, 'bias')


def test_dissect_activation(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)

    # get activations
    activation_extractor = ActivationExtractor(model)
    activations = activation_extractor.extract_activations(input_tensor)
    assert len(activations) == len(activation_extractor.layers)


def test_dissector(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)
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
    dissect_ret = dissector.dissect(input_tensor, input_tangent, output_tangent=output_tangent)
    assert 'weights' in dissect_ret.keys()
    assert 'biases' in dissect_ret.keys()
    assert 'activations' in dissect_ret.keys()
    assert 'forward_grads' in dissect_ret.keys()
    assert 'backward_grads' in dissect_ret.keys()
