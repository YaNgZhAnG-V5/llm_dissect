from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import mmengine
import torch
import torch.autograd.forward_ad as fw_ad
import torch.nn as nn

from .utils import get_input_key_mapping


def get_layers(model: nn.Module, return_dict: bool = True) -> Union[List[nn.Module], Dict[str, nn.Module]]:
    """Get all trainable layers in format {layer_name: layer} or [layer]"""
    # rsplit "." to get rid of the "weight" or "bias" suffix. E.g., 'features.0.conv.weight' -> 'features.0.conv'
    names = [k.rsplit(".", maxsplit=1)[0] for k, _ in model.named_parameters()]
    name_layer_tuples = [(n, model.get_submodule(n)) for n in names]
    if return_dict:
        return dict(name_layer_tuples)
    else:
        return [x[1] for x in name_layer_tuples]


class ReplaceHookRegister:

    def __init__(self) -> None:
        self.enabled = False

    def replace_forward_hook(self, module, input, output):
        # TODO this should be changed to account for cases where a different tensor is needed
        if self.enabled:
            tangent = torch.ones_like(output)
            output = fw_ad.make_dual(output, tangent)
        return output

    def __call__(self):
        return self.replace_forward_hook


class EnableReplaceHook:
    def __init__(self, hook_register) -> None:
        self.hook_register = hook_register

    def __enter__(self):
        if self.hook_register is not None:
            self.hook_register.enabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_register is not None:
            self.hook_register.enabled = False


class InputOutputHookRegister:
    """Store (1) input or input norm; and (2) output activations."""

    def __init__(self, module_name: str, norm: bool = True, input_key: Optional[List[str]] = None) -> None:
        self.input = None
        self.output = None
        self.module_name = module_name
        self._norm = norm
        # if input_key is None, then the input tensor is passed to the args of the hook. Otherwise, the input tensor
        # is passed to kwargs of the hook. In this case, we use the input_key to retrieve the actual input from the
        # kwargs.
        self._input_key = input_key

    def _save_input_output(self, module, input, output):
        """The actual hook method"""
        self.input = input

        if self._norm:
            self.input = self.input.reshape(-1, self.input.shape[-1]).norm(p=2, dim=0)
        if isinstance(output, tuple):
            self.output = output[0]
        elif isinstance(output, torch.Tensor):
            self.output = output
        else:
            raise TypeError(f"Unsupported output type: {output.__class__.__name__}")
        return output

    def save_input_output_without_kwargs(self, module, input, output):
        # TODO input is saved in the same format as wanda
        actual_input = input[0]
        return self._save_input_output(module, actual_input, output)

    def save_input_output_with_kwargs(self, module, input, kwargs, output):
        # nn.Module.register_forward_hook requires kwargs to be in the hook function's param list,
        # when with_kwargs = True
        actual_input = {k: kwargs[k] for k in self._input_key}
        return self._save_input_output(module, actual_input, output)

    def __call__(self):
        if self._input_key is None:
            return self.save_input_output_without_kwargs
        else:
            return self.save_input_output_with_kwargs


class BackwardHookRegister:

    def __init__(self, module_name: str) -> None:
        self.output = None
        self.module_name = module_name

    def output2dual(self, module, grad_input, grad_output):
        self.output = grad_output

        # we do not change grad_input in our case
        return None

    def __call__(self):
        return self.output2dual


class BasedExtractor:

    def __init__(self, model: nn.Module, layers: Optional[Union[List, Dict]] = None):
        self.model = model

        # add hook on all layers
        if not layers:
            layers = get_layers(self.model, return_dict=True)
        assert layers, "layers not initialized correctly!"
        self.layers = layers

    def clear_hooks(self) -> None:
        pass


class ForwardADExtractor(BasedExtractor):

    def __init__(
        self, model: nn.Module, layers: Optional[Union[List, Dict]] = None, dual_insert_layer: Optional[str] = None
    ):
        super().__init__(model, layers)
        self.hook_handles, self.hook_registers = [], []

        # create forward hooks to extract forward gradients
        for name, layer in self.layers.items():
            output_hook_register = InputOutputHookRegister(module_name=name)
            self.hook_registers.append(output_hook_register)
            self.hook_handles.append(layer.register_forward_hook(output_hook_register()))

        # define insert layer to initialize dual tensor for forward ad
        if dual_insert_layer is not None:
            self.replace_register = ReplaceHookRegister()
            # self.layers store all the leaf layers, however dual_insert_layer might be a whole (non-leaf) layer
            # so use get_submodule instead of self.layers[dual_insert_layer]
            self.dual_insert_layer = self.model.get_submodule(dual_insert_layer)
            self.replace_hook_handle = self.dual_insert_layer.register_forward_hook(self.replace_register())
        else:
            self.dual_insert_layer = None
            self.replace_register = None

    @property
    def not_from_input(self) -> bool:
        return self.dual_insert_layer is not None

    def forward_ad(
        self,
        input_tensor: torch.Tensor,
        tangent: Optional[torch.Tensor] = None,
        forward_kwargs: Optional[Mapping] = None,
    ) -> Dict[str, torch.Tensor]:
        output_forward_grads = {}
        with EnableReplaceHook(self.replace_register):
            with torch.no_grad():
                with fw_ad.dual_level():
                    # make input a dual tensor
                    if not self.not_from_input:
                        if tangent is None:
                            tangent = torch.ones_like(input_tensor)
                        input_tensor = fw_ad.make_dual(input_tensor, tangent)

                    # perform forward and collect forward gradient
                    _ = (
                        self.model(input_tensor)
                        if forward_kwargs is None
                        else self.model(input_tensor, **forward_kwargs)
                    )
                    for hook_register in self.hook_registers:
                        output_jvp = fw_ad.unpack_dual(hook_register.output).tangent

                        # account for the case where input is not the dual tensor
                        if output_jvp is None:
                            continue
                        output_jvp = output_jvp.detach().cpu()
                        output_forward_grads[hook_register.module_name] = output_jvp
        return output_forward_grads

    def clear_hooks(self) -> None:
        while self.hook_handles:
            hook = self.hook_handles.pop()
            hook.remove()
            self.replace_hook_handle.remove()
        while self.hook_registers:
            hook_register = self.hook_registers.pop()
            del hook_register
            del self.replace_register
            self.replace_register = None


class BackwardADExtractor(BasedExtractor):

    def __init__(self, model, layers: Optional[Union[list, dict]] = None):
        super().__init__(model, layers)

        # add backward hooks
        self.hook_handles, self.hook_registers = [], []
        for name, layer in self.layers.items():
            output_hook_register = BackwardHookRegister(module_name=name)
            self.hook_registers.append(output_hook_register)
            self.hook_handles.append(layer.register_full_backward_hook(output_hook_register()))

    def backward_ad(
        self,
        input_tensor: torch.Tensor,
        tangent: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        criterion: Optional[Callable] = None,
        forward_kwargs: Optional[Dict] = None,
        use_loss: bool = False,
    ):
        # forward pass
        model_output = (
            self.model(input_tensor) if forward_kwargs is None else self.model(input_tensor, **forward_kwargs)
        )

        if not use_loss:
            # set valid tangent if not given
            if tangent is None:
                if not isinstance(model_output, torch.Tensor):
                    model_output = model_output.logits
                tangent = torch.ones_like(model_output)
            # backward pass using logits
            model_output.backward(tangent)
        else:
            # backward pass using loss from the output
            if not isinstance(model_output, torch.Tensor):
                loss = model_output.loss
                loss.backward()
            else:
                # backward pass using loss from the criterion
                assert target is not None, "target must be provided if criterion is provided"
                assert criterion is not None, "criterion must be provided if target is provided"
                loss = criterion(model_output, target)
                loss.backward()
        output_backward_grads = {}
        for hook_register in self.hook_registers:
            # get output gradients (always wrapped in a tuple)
            output_backward_grad = hook_register.output[0].detach().cpu()
            output_backward_grads[hook_register.module_name] = output_backward_grad
        return output_backward_grads

    def clear_hooks(self) -> None:
        while self.hook_handles:
            hook = self.hook_handles.pop()
            hook.remove()
        while self.hook_registers:
            hook_register = self.hook_registers.pop()
            del hook_register


class WeightExtractor(BasedExtractor):

    def __init__(self, model: nn.Module, layers: Optional[Union[List, Dict]] = None) -> None:
        super().__init__(model, layers)

    def extract_weights_biases(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        weights = {}
        biases = {}
        for name, layer in self.layers.items():
            weights[name] = layer.weight.detach().cpu()
            if getattr(layer, "bias", None) is not None:
                biases[name] = layer.bias.detach().cpu()
            else:
                biases[name] = None
        return weights, biases


class ActivationExtractor(BasedExtractor):

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[Union[List, Dict]] = None,
        norm: bool = True,
        input_key_strategy: Optional[str] = None,
    ) -> None:
        super().__init__(model, layers)
        self.hook_list, self.hook_objs = [], []
        logger = mmengine.MMLogger.get_instance("dissect")
        if input_key_strategy is not None:
            logger.info(f"Using input_key_strategy: {input_key_strategy}")
            self.input_key_mapping: Callable[[str], Optional[List[str]]] = get_input_key_mapping(input_key_strategy)

        for name, layer in self.layers.items():
            if input_key_strategy is None:
                input_key = None
            else:
                input_key: Optional[List[str]] = self.input_key_mapping(name)
            logger.debug(f"layer_name: {name}: input_key: {input_key}")

            if input_key is None:
                # Case 1: layer's forward method takes args instead of kwargs, we don't need input_key to retrieve the
                # input tensor in the hook. When registering hooks, with_kwargs is set to False
                hook_register = InputOutputHookRegister(module_name=name, norm=norm, input_key=input_key)
                self.hook_objs.append(hook_register)
                # If with_kwargs = True, the hook will ignore all kwargs passed to the layer's forward method.
                self.hook_list.append(layer.register_forward_hook(hook_register(), with_kwargs=False))
            else:
                # Case 2: layer's forward method takes kwargs instead of kwargs, we need input_key to retrieve the input
                # tensor in the hook. When registering hooks, with_kwargs is set to True
                hook_register = InputOutputHookRegister(module_name=name, norm=norm, input_key=input_key)
                self.hook_objs.append(hook_register)
                # If with_kwargs = True, the hook will ignore all kwargs passed to the layer's forward method.
                self.hook_list.append(layer.register_forward_hook(hook_register(), with_kwargs=True))

    def extract_activations(
        self, input_tensor: torch.Tensor, forward_kwargs: Optional[Mapping] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        inputs = {}
        activations = {}
        with torch.no_grad():
            _ = self.model(input_tensor) if forward_kwargs is None else self.model(input_tensor, **forward_kwargs)
            for hook in self.hook_objs:
                activations[hook.module_name] = hook.output.detach().cpu()
                if isinstance(hook.input, torch.Tensor):
                    inputs[hook.module_name] = hook.input.detach().cpu()
                elif isinstance(hook.input, dict):
                    # if the layer is e.g. self_attn, then hook.input is a dict
                    inputs[hook.module_name] = {k: v.detach().cpu() for k, v in hook.input.items()}
                else:
                    raise TypeError(f"Invalid type for hook.input: {type(hook.input)}")
        return activations, inputs

    def clear_hooks(self) -> None:
        while self.hook_list:
            hook = self.hook_list.pop()
            hook.remove()
        while self.hook_objs:
            hook_obj = self.hook_objs.pop()
            del hook_obj


class Dissector(BasedExtractor):

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[Union[str, Dict]] = None,
        dual_insert_layer: Optional[str] = None,
        norm: bool = True,
    ) -> None:
        super().__init__(model, layers=layers)
        self.forward_ad_extractor = ForwardADExtractor(model, layers=self.layers, dual_insert_layer=dual_insert_layer)
        self.backward_ad_extractor = BackwardADExtractor(model, layers=self.layers)
        self.activation_extractor = ActivationExtractor(model, layers=self.layers, norm=norm)
        self.weight_extractor = WeightExtractor(model, layers=self.layers)

    def dissect(
        self,
        input_tensor: torch.Tensor,
        input_tangent: Optional[torch.Tensor] = None,
        output_tangent: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        criterion: Optional[Callable] = None,
        forward_kwargs: Optional[Mapping] = None,
        use_loss: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        weights, biases = self.weight_extractor.extract_weights_biases()
        activations, inputs = self.activation_extractor.extract_activations(
            input_tensor=input_tensor, forward_kwargs=forward_kwargs
        )
        output_forward_grads = self.forward_ad_extractor.forward_ad(
            input_tensor=input_tensor, tangent=input_tangent, forward_kwargs=forward_kwargs
        )
        backward_grads = self.backward_ad_extractor.backward_ad(
            input_tensor=input_tensor,
            tangent=output_tangent,
            target=target,
            criterion=criterion,
            forward_kwargs=forward_kwargs,
            use_loss=use_loss,
        )
        return {
            "forward_grads": output_forward_grads,
            "activations": activations,
            "inputs": inputs,
            "weights": weights,
            "biases": biases,
            "backward_grads": backward_grads,
        }
