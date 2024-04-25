from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import mmengine
import torch.backends.cuda
import transformers
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils import Device, get_cuda_visible_devices


def build_model_and_tokenizer(cfg: Dict, device: Device) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger = mmengine.MMLogger.get_instance("dissect")
    cfg = deepcopy(cfg)

    dtype = cfg.get("model_args", dict()).pop("dtype")
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        logger.info("Model will be loaded with torch.float16 precision.")
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        logger.info("Model will be loaded with torch.bfloat16 precision")
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    cfg.setdefault("model_args", dict())
    cfg["model_args"].update({"torch_dtype": torch_dtype})

    model_class = getattr(transformers, cfg["model_class"])
    # Multi-gpu inference
    cuda_visible_devices = get_cuda_visible_devices()
    if len(cuda_visible_devices) > 1:
        model = model_class.from_pretrained(cfg["model_name"], device_map="auto", **cfg["model_args"])
    else:
        model = model_class.from_pretrained(cfg["model_name"], **cfg["model_args"]).to(device)

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


def build_lm_eval_wrapper(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, lm_wrapper_cfg: Optional[Dict[str, Any]] = None
) -> HFLM:
    lm_wrapper_cfg = dict() if lm_wrapper_cfg is None else lm_wrapper_cfg
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, **lm_wrapper_cfg)
    return hflm
