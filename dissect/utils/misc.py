from typing import Iterable
from typing import Dict, Tuple, List
from torch import nn


def name_contains_keys(name: str, keys: Iterable[str]) -> bool:
    """If `name` contains any sub-string key in `keys`, return True. Otherwise, return False."""
    return any(key in name for key in keys)


def calc_pruned_parameters(model: nn.Module, mask_state_dict: Dict) -> Tuple[List[Dict[str, float]], float, float]:
    """
    calculate the number of pruned parameters in each layer and the total number of pruned parameters
    assume neuron pruning
    param model: the model to be pruned
    param mask_state_dict: the mask state dict that is used to mask the model
    return: a list of dictionaries, each dictionary contains the layer name, neuron sparsity, and parameter sparsity
    return: the total sparsity ratio in the target layers
    return: the total sparsity ratio in the entire model
    """
    # get total number of parameters, ignore bias
    total_params_model = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    pruned_parameters = 0

    # calculate total_params_target_layers only once since it wont change
    total_params_target_layers = sum(model.get_submodule(k).weight.data.numel() for k in mask_state_dict.keys())

    log_tabulate = []
    for k in sorted(mask_state_dict.keys()):
        v = mask_state_dict[k]
        # if the mask stores weight gradients, then it does not have to be one-dim
        assert v.ndim == 1, "mask should be one-dimensional, calculation is only for neuron pruning."

        # TODO: we ignore bias here, not sure if we need to include later
        total_params_layer = model.get_submodule(k).weight.data.numel()
        pruned_parameters_layer = (
            total_params_layer - v.float().sum().item() * model.get_submodule(k).weight.data.shape[1]
        )
        pruned_parameters += pruned_parameters_layer

        log_tabulate.append(
            {
                "layer": k,
                "neuron_sparsity": 1 - v.float().mean().item(),
                "param_sparsity": pruned_parameters_layer / total_params_layer,
            }
        )

    # get global pruning ratio
    sparsity_target_layers = pruned_parameters / total_params_target_layers
    sparsity_whole_model = pruned_parameters / total_params_model
    return log_tabulate, sparsity_target_layers, sparsity_whole_model
