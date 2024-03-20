from typing import Iterable


def name_contains_keys(name: str, keys: Iterable[str]) -> bool:
    """If `name` contains any sub-string key in `keys`, return True. Otherwise, return False."""
    return any(key in name for key in keys)
