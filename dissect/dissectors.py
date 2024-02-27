import torch
import torch.autograd.forward_ad as fwAD
from typing import Optional, Union, Callable
from tqdm import tqdm


class OutputHookRegister:
    def __init__(self, module_name: str = None):
        self.output = None
        self.module_name = module_name

    def output2dual(self, module, input, output):
        self.output = output
        return output

    def __call__(self):
        return self.output2dual


class BackwardHookRegister:
    def __init__(self, module_name: str = None):
        self.output = None
        self.module_name = module_name

    def output2dual(self, module, grad_input, grad_output):
        self.output = grad_output

        # we do not change grad_input in our case
        return None

    def __call__(self):
        return self.output2dual


def get_layers(model, return_dict: bool = False):
    """get layers with trainable parameters"""
    params_name = list(dict(model.named_parameters()).keys())
    params_name = {".".join(name.split(".")[:-1]) for name in params_name}
    layers = [] if not return_dict else {}
    for name in params_name:
        name_list = name.split(".")
        layer = model
        for submodule_name in name_list:
            if submodule_name in [str(i) for i in range(100)]:
                layer = layer[int(submodule_name)]
            elif hasattr(layer, submodule_name):
                layer = getattr(layer, submodule_name)
        if return_dict:
            layers[name] = layer
        else:
            layers.append(layer)
    return layers


class BasedExtractor:
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        self.model = model

        # add hook on all layers
        if not layers:
            layers = get_layers(self.model, return_dict=True)
        assert layers, "layers not initialized correctly!"
        self.layers = layers


class ForwardADExtractor(BasedExtractor):
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)
        self.hook_list, self.hook_objs = [], []
        for name, layer in self.layers.items():
            output_hook_register = OutputHookRegister(module_name=name)
            self.hook_objs.append(output_hook_register)
            self.hook_list.append(layer.register_forward_hook(output_hook_register()))

    def forward_ad(self, input_tensor: torch.Tensor, tangent: Optional[torch.Tensor] = None):
        output_forward_grads = {}
        with torch.no_grad():
            with fwAD.dual_level():
                # make input a dual tensor
                if tangent is None:
                    tangent = torch.ones_like(input_tensor)
                input_tensor = fwAD.make_dual(input_tensor, tangent)

                # perform forward and collect forward gradient
                _ = self.model(input_tensor)
                for hook in tqdm(self.hook_objs, desc="Collecting forward gradients"):
                    output_jvp = fwAD.unpack_dual(hook.output).tangent.detach().cpu()
                    output_forward_grads[hook.module_name] = output_jvp
        return output_forward_grads

    def clear_hooks(self):
        while self.hook_list:
            hook = self.hook_list.pop()
            hook.remove()
        while self.hook_objs:
            hook_obj = self.hook_objs.pop()
            del hook_obj


class BackwardADExtractor(BasedExtractor):
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)

        # add backward hooks
        self.hook_list, self.hook_objs = [], []
        for name, layer in self.layers.items():
            output_hook_register = BackwardHookRegister(module_name=name)
            self.hook_objs.append(output_hook_register)
            self.hook_list.append(layer.register_full_backward_hook(output_hook_register()))

    def backward_ad(
        self,
        input_tensor: torch.Tensor,
        tangent: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        criterion: Optional[Callable] = None,
    ):
        # forward pass
        model_output = self.model(input_tensor)
        if tangent is None:
            tangent = torch.ones_like(model_output)

        # backward pass
        if target is None and criterion is None:
            model_output.backward(tangent)
        else:
            assert target is not None, "target must be provided if criterion is provided"
            assert criterion is not None, "criterion must be provided if target is provided"
            loss = criterion(model_output, target)
            loss.backward()
        output_backward_grads = {}
        for hook in tqdm(self.hook_objs, desc="Collecting forward gradients"):
            # get output gradients (always wrapped in a tuple)
            output_backward_grad = hook.output[0].detach().cpu()
            output_backward_grads[hook.module_name] = output_backward_grad
        return output_backward_grads


class WeightExtractor(BasedExtractor):
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)

    def extract_weights(self):
        weights = {}
        bias = {}
        for name, layer in tqdm(self.layers.items(), desc="Collecting weights"):
            weights[name] = layer.weight.detach().cpu()
            if hasattr(layer, "bias"):
                bias[name] = layer.bias.detach().cpu()
            else:
                bias[name] = None
        return {"weights": weights, "biases": bias}


class ActivationExtractor(BasedExtractor):
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)
        self.hook_list, self.hook_objs = [], []
        for name, layer in tqdm(self.layers.items(), desc="Collecting activations"):
            dual_hook_register = OutputHookRegister(module_name=name)
            self.hook_objs.append(dual_hook_register)
            self.hook_list.append(layer.register_forward_hook(dual_hook_register()))

    def extract_activations(self, input_tensor: torch.Tensor):
        activations = {}
        with torch.no_grad():
            _ = self.model(input_tensor)
            for hook in self.hook_objs:
                activations[hook.module_name] = hook.output.detach().cpu()
        return activations

    def clear_hooks(self):
        while self.hook_list:
            hook = self.hook_list.pop()
            hook.remove()
        while self.hook_objs:
            hook_obj = self.hook_objs.pop()
            del hook_obj


class Dissector(BasedExtractor):
    def __init__(self, model):
        super().__init__(model)
        self.forward_ad_extractor = ForwardADExtractor(model, self.layers)
        self.backward_ad_extractor = BackwardADExtractor(model, self.layers)
        self.activation_extractor = ActivationExtractor(model, self.layers)
        self.weight_extractor = WeightExtractor(model, self.layers)

    def dissect(
        self,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        criterion: Optional[Callable] = None,
        input_tangent: Optional[torch.Tensor] = None,
        output_tangent: Optional[torch.Tensor] = None,
    ):
        weights = self.weight_extractor.extract_weights()
        activations = self.activation_extractor.extract_activations(input_tensor)
        output_forward_grads = self.forward_ad_extractor.forward_ad(input_tensor, input_tangent)
        backward_grads = self.backward_ad_extractor.backward_ad(input_tensor, output_tangent, target, criterion)
        return {
            "forward_grads": output_forward_grads,
            "activations": activations,
            "weights": weights,
            "backward_grads": backward_grads,
        }
