from typing import Dict, Tuple

import mmengine
import torch.backends.cuda
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils import Device


def build_model_and_tokenizer(cfg: Dict, device: Device) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger = mmengine.MMLogger.get_instance("dissect")
    model_class = getattr(transformers, cfg["model_class"])
    model = model_class.from_pretrained(cfg["model_name"])
    dtype = cfg.get("dtype", "float")
    if dtype == "half":
        logger.info("Converting model to half precision.")
        model = model.half().to(device)
    elif dtype == "float":
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model dtype: {dtype}")

    # It requires enable_mem_efficient_sdp=False for Vicuna model to work.
    if not cfg.get("mem_efficient_sdp", True):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        logger.info("cfg.model.mem_efficient_sdp=False, so called torch.backends.cuda.enable_mem_efficient_sdp(False)")

    tokenizer_class = getattr(transformers, cfg["tokenizer_class"])
    tokenizer = tokenizer_class.from_pretrained(cfg["tokenizer_name"])

    if "llama" in cfg["model_name"].lower():
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad token to eos token for LLama.")

    return model, tokenizer
