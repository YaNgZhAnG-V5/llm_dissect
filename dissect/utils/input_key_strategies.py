from typing import Callable, List, Optional


def get_input_key_mapping(strategy: str) -> Callable[[str], Optional[str]]:
    """Get the function that tells which key to use to retrieve the input tensor from the parameter dict from
    the hook. This function is used in the InputOutputHook."""
    if strategy == "vicuna":
        return vicuna_input_key_mapping
    else:
        raise NotImplementedError(f"Input mapping is not implemented for {strategy}")


def vicuna_input_key_mapping(layer_name: str) -> Optional[List[str]]:
    if "self_attn" in layer_name:
        return ["hidden_states", "attention_mask", "position_ids"]
    elif "mlp" in layer_name:
        return None
    else:
        raise ValueError(f"Unmatched layer_name: {layer_name}")
