from typing import Callable, List, Optional


def get_input_key_mapping(strategy: str) -> Callable[[str], Optional[str]]:
    """Get the function that tells which key to use to retrieve the input tensor from the parameter dict from
    the hook. This function is used in the InputOutputHook."""
    if strategy == "vicuna" or "llama2":
        return vicuna_input_key_mapping
    else:
        raise NotImplementedError(f"Input mapping is not implemented for {strategy}")


def vicuna_input_key_mapping(layer_name: str) -> Optional[List[str]]:
    if "layers" in layer_name:
        if "self_attn" in layer_name:
            return ["hidden_states", "attention_mask", "position_ids"]
        elif "mlp" in layer_name:
            return None
        else:
            # E.g. "model.layers.1" instead of "model.layers.1.self_attn" or "model.layers.1.mlp",
            # the decoder_layer is called as follows:
            # """
            # layer_outputs = decoder_layer(
            #                     hidden_states,
            #                     attention_mask=causal_mask,
            #                     position_ids=position_ids,
            #                     past_key_value=past_key_values,
            #                     output_attentions=output_attentions,
            #                     use_cache=use_cache,
            #                     cache_position=cache_position,
            #                 )
            # """
            # the hidden_states is passed as positional argument.

            return None
    else:
        raise ValueError(f"Unmatched layer_name: {layer_name}")
