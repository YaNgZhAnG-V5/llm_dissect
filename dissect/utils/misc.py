import os
import sys
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional, Union

import torch
from mmengine.device import is_cuda_available, is_musa_available
from mmengine.dist.utils import master_only
from tqdm import tqdm


class OrderedLayerNameHook:
    """get orders of all the target layers via hook."""

    def __init__(self, target_layers: dict):
        self.target_layers = target_layers
        self.ordered_target_layers = []

    def __call__(self, module, input, output):
        self.ordered_target_layers.append(
            list(self.target_layers.keys())[list(self.target_layers.values()).index(module)]
        )


def get_target_layers_ordered(model: torch.nn.Module, target_modules: List[str]):
    target_layers = {}
    hooks = []
    hook_callback = OrderedLayerNameHook(target_layers)

    # get target layers
    for target_module in target_modules:
        for name, layer in model.named_modules():
            if target_module in name.split(".")[-1] and name not in target_layers:
                target_layers[name] = layer
                hooks.append(layer.register_forward_hook(hook_callback))
    with torch.no_grad():
        _ = model(model.dummy_inputs["input_ids"].to(model.device))

    # remove hooks
    while hooks:
        hooks.pop().remove()
    target_layers = hook_callback.ordered_target_layers
    return target_layers


def get_target_layers(model: torch.nn.Module, target_modules: List[str], exclude_rate: float):
    target_layers = []

    # get target layers
    for target_module in target_modules:
        target_layers += [name for name, _ in model.named_modules() if target_module in name.split(".")[-1]]
    target_layers = sorted(list(set(target_layers)))

    # get total target layer number before exclusion
    total_target_layer_number = len(target_layers)

    # get exclude_layers, int always return the floor value, we want to use ceil here
    # since target layers contain both attn and mlp, we divide the exclude rate by 2
    num_exclude_layers = int(len(target_layers) * exclude_rate / 2) + 1
    exclude_layers = [f".{i}." for i in range(num_exclude_layers)]
    layer_to_remove = []
    for layer in target_layers:
        for exclude_layer in exclude_layers:
            if exclude_layer in layer:
                layer_to_remove.append(layer)
    for layer in layer_to_remove:
        target_layers.remove(layer)
    exclude_layers = layer_to_remove
    return target_layers, exclude_layers, total_target_layer_number


def name_contains_keys(name: str, keys: Iterable[str]) -> bool:
    """If `name` contains any sub-string key in `keys`, return True. Otherwise, return False."""
    return any(key in name for key in keys)


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppress_tqdm():
    original_tqdm = tqdm.__init__

    def no_op_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        original_tqdm(*args, **kwargs)

    tqdm.__init__ = no_op_tqdm
    try:
        yield
    finally:
        tqdm.__init__ = original_tqdm


class TimeCounter:
    """adjust from mmengine time counter"""

    instance_dict: dict = dict()

    log_interval: int
    warmup_interval: int
    __count: int
    __pure_inf_time: float

    def __new__(
        cls,
        log_interval: int = 1,
        warmup_interval: int = 1,
        tag: Optional[str] = None,
    ):
        assert warmup_interval >= 1
        if tag is not None and tag in cls.instance_dict:
            return cls.instance_dict[tag]

        instance = super().__new__(cls)
        cls.instance_dict[tag] = instance

        instance.log_interval = log_interval
        instance.warmup_interval = warmup_interval
        instance.with_sync = True

        instance.__count = 0
        instance.__pure_inf_time = 0.0
        instance.__start_time = 0.0

        return instance

    @master_only
    def __call__(self, fn):
        if self.tag is None:
            self.tag = fn.__name__

        def wrapper(*args, **kwargs):
            self.__count += 1

            if self.with_sync:
                if is_cuda_available():
                    torch.cuda.synchronize()
                elif is_musa_available():
                    torch.musa.synchronize()
            start_time = time.perf_counter()

            result = fn(*args, **kwargs)

            if self.with_sync:
                if is_cuda_available():
                    torch.cuda.synchronize()
                elif is_musa_available():
                    torch.musa.synchronize()
            elapsed = time.perf_counter() - start_time
            self.print_time(elapsed)

            return result

        return wrapper

    @master_only
    def __enter__(self):
        self.__count += 1

        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.__start_time = time.perf_counter()

    @master_only
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.__start_time
        self.calc_time(elapsed)

    def calc_time(self, elapsed: Union[int, float]) -> None:
        """print times per count."""
        if self.__count >= self.warmup_interval:
            self.__pure_inf_time += elapsed

            if self.__count % self.log_interval == 0:
                self.times_per_count = self.__pure_inf_time / (self.__count - self.warmup_interval + 1)

    def get_times_per_count(self):
        times_per_count = self.times_per_count
        # self.times_per_count = 0.0
        return times_per_count


def get_cuda_visible_devices() -> list[int]:
    """Get the device ids of all available GPUs."""
    return [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
