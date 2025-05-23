import torch
import torch.autograd.forward_ad as fw_ad
from torch import nn

from dissect.dissectors import ActivationExtractor, BackwardADExtractor, Dissector, ForwardADExtractor, WeightExtractor


# helper function for the test ################################################
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
    train_kwargs = {"batch_size": 256}
    test_kwargs = {"batch_size": 1000}
    # if gpu_id < 0, then use CPU
    use_cuda = gpu_id >= 0
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = MLP()
    model.to(device)

    # TODO: currently only support single batch, so only test single batch
    input_tensor = torch.rand(1, 1, 28, 28).to(device)
    return model, input_tensor


class GetOutputGradHook:
    def __init__(self):
        self.grad_output = None

    def __call__(self, module, grad_input, grad_output):
        self.grad_output = grad_output[0]


def get_true_output_grads(model, input_tensor):
    # get the true gradient of the model
    hook_objs_dict, hook_handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = GetOutputGradHook()
            handle = module.register_full_backward_hook(hook)
            hook_objs_dict[name] = hook
            hook_handles.append(handle)
    model.zero_grad()
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    grad_dict = {}
    for name, hook in hook_objs_dict.items():
        grad_dict[name] = hook.grad_output
    for handle in hook_handles:
        handle.remove()
    return grad_dict


class GetOutputHook:
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        self.output = output


def get_true_input_grads(model, input_tensor):
    # get the true input gradient of the model via forward grads
    hook_objs_dict, hook_handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = GetOutputHook()
            handle = module.register_forward_hook(hook)
            hook_objs_dict[name] = hook
            hook_handles.append(handle)
    tangent_dict = {}
    with torch.no_grad():
        with fw_ad.dual_level():
            input_tensor = fw_ad.make_dual(input_tensor, torch.ones_like(input_tensor))
            output = model(input_tensor)
            for name, hook in hook_objs_dict.items():
                tangent = fw_ad.unpack_dual(hook.output).tangent
                tangent_dict[name] = tangent
    for handle in hook_handles:
        handle.remove()
    return tangent_dict


def get_true_output(model, input_tensor):
    hook_objs_dict, hook_handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = GetOutputHook()
            handle = module.register_forward_hook(hook)
            hook_objs_dict[name] = hook
            hook_handles.append(handle)
    output_dict = {}
    with torch.no_grad():
        with fw_ad.dual_level():
            input_tensor = fw_ad.make_dual(input_tensor, torch.ones_like(input_tensor))
            output = model(input_tensor)
            for name, hook in hook_objs_dict.items():
                output_dict[name] = hook.output
    for handle in hook_handles:
        handle.remove()
    return output_dict


# helper function for the test ################################################


def test_dissect_forward(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)

    # get input gradient
    forward_ad_extractor = ForwardADExtractor(model)
    input_grads = forward_ad_extractor.forward_ad(input_tensor)
    assert len(input_grads) == len(forward_ad_extractor.hook_handles) == len(forward_ad_extractor.hook_registers)
    for module_name, input_grad in input_grads.items():
        # input gradient should only have one dimension, and the dim consist with the neuron number at that layer
        assert input_grad.ndim == 1
        assert input_grad.shape[0] == model.get_submodule(module_name).out_features

    # run correctness check
    forward_ad_extractor.clear_hooks()
    true_input_grads = get_true_input_grads(model, input_tensor)
    for name, input_grad in input_grads.items():
        assert torch.allclose(input_grad, true_input_grads[name].abs().mean(0))


def test_dissect_backward(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)
    backward_ad_extractor = BackwardADExtractor(model)
    output_backward_grads = backward_ad_extractor.backward_ad(input_tensor)
    assert (
        len(output_backward_grads)
        == len(backward_ad_extractor.hook_handles)
        == len(backward_ad_extractor.hook_registers)
    )

    # run correctness check
    backward_ad_extractor.clear_hooks()
    true_output_grads = get_true_output_grads(model, input_tensor)
    for name, output_grad in output_backward_grads.items():
        assert torch.allclose(output_grad, true_output_grads[name].abs().mean(0))


def test_dissect_weight(gpu_id):
    model, _ = prepare_model_and_data(gpu_id)
    weight_extractor = WeightExtractor(model)

    # get weights
    weights, biases = weight_extractor.extract_weights_biases()
    assert len(weights) == len(biases)
    for name, layer in weight_extractor.layers.items():
        assert torch.allclose(layer.weight.data.detach(), weights[name])
        if biases[name] is not None:
            assert torch.allclose(layer.bias.data.detach(), biases[name])
        else:
            assert not hasattr(layer, "bias")


def test_dissect_activation(gpu_id):
    model, input_tensor = prepare_model_and_data(gpu_id)

    # get activations
    activation_extractor = ActivationExtractor(model)
    activations, input_norm = activation_extractor.extract_activations(input_tensor)
    assert len(activations) == len(activation_extractor.layers)
    assert len(input_norm) == len(activation_extractor.layers)
    activation_extractor.clear_hooks()
    true_activation = get_true_output(model, input_tensor)
    for name, activation in activations.items():
        assert torch.allclose(activation, true_activation[name].abs().mean(0))


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
    output_tangent = torch.rand(input_tensor.shape[0], 10).to(input_tensor.device)
    dissect_ret = dissector.dissect(input_tensor, input_tangent, output_tangent=output_tangent)
    assert "weights" in dissect_ret.keys()
    assert "biases" in dissect_ret.keys()
    assert "activations" in dissect_ret.keys()
    assert "forward_grads" in dissect_ret.keys()
    assert "backward_grads" in dissect_ret.keys()
