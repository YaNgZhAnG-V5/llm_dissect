import os
import time
from typing import Iterable, Optional, Union

import torch
from mmengine.device import is_cuda_available, is_musa_available
from mmengine.dist.utils import master_only


def name_contains_keys(name: str, keys: Iterable[str]) -> bool:
    """If `name` contains any sub-string key in `keys`, return True. Otherwise, return False."""
    return any(key in name for key in keys)


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
