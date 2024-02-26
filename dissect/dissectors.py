import torch
import torch.autograd.forward_ad as fwAD
from typing import Optional, Union


class OutputHookRegister:
    def __init__(self, module_name: str = None):
        self.output = None
        self.module_name = module_name

    def output2dual(self, module, input, output):
        self.output = output
        return output

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
        with fwAD.dual_level():
            # make input a dual tensor
            if not tangent:
                tangent = torch.ones_like(input_tensor)
            input_tensor = fwAD.make_dual(input_tensor, tangent)

            # perform forward and collect forward gradient
            _ = self.model(input_tensor)
            for hook in self.hook_objs:
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

        # TODO add backward hooks

    def backward_ad(self, input_tensor: torch.Tensor, tangent: Optional[torch.Tensor] = None):
        # TODO implement backward ad
        pass


class WeightExtractor(BasedExtractor):
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)
        self.weights = {}
        self.bias = {}
        for name, layer in self.layers.items():
            self.weights[name] = layer.weight.detach().cpu()
            if hasattr(layer, "bias"):
                self.bias[name] = layer.bias.detach().cpu()
            else:
                self.bias[name] = None

    def extract_weights(self):
        return {"weights": self.weights, "biases": self.bias}


class ActivationExtractor(BasedExtractor):
    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)
        self.hook_list, self.hook_objs = [], []
        self.activations = {}
        for name, layer in self.layers.items():
            dual_hook_register = OutputHookRegister(module_name=name)
            self.hook_objs.append(dual_hook_register)
            self.hook_list.append(layer.register_forward_hook(dual_hook_register()))

    def extract_activations(self, input_tensor: torch.Tensor):
        with torch.no_grad():
            _ = self.model(input_tensor)
            for hook in self.hook_objs:
                self.activations[hook.module_name] = hook.output.detach().cpu()
        return self.activations

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
        input_tangent: Optional[torch.Tensor] = None,
        output_tangent: Optional[torch.Tensor] = None,
    ):
        weights = self.weight_extractor.get_weights()
        activations = self.activation_extractor.get_activations(input_tensor)
        output_forward_grads = self.forward_ad_extractor.forward_ad(input_tensor, input_tangent)
        backward_grads = self.backward_ad_extractor.backward_ad(input_tensor, output_tangent)
        return {
            "forward_grads": output_forward_grads,
            "activations": activations,
            "weights": weights,
            "backward_grads": backward_grads,
        }
