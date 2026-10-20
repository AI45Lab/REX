from __future__ import annotations

import contextlib
import time
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm

from .dist_utils import is_local_rank0
from .logger import get_logger, log_rank0
from .pad import pad_to_max_length, pad_to_multiple_of

IGNORE_INDEX = -100


class SharedMemory(_mpshm.SharedMemory):
    def __init__(self, name: str | None = None, create: bool = False, size: int = 0, *, track: bool = False) -> None:
        self._track = track
        if track:
            super().__init__(name=name, create=create, size=size)
            return
        original_register = _mprt.register
        _mprt.register = lambda *args, **kwargs: None
        try:
            super().__init__(name=name, create=create, size=size)
        finally:
            _mprt.register = original_register

    def unlink(self) -> None:
        if self._track:
            super().unlink()
            return

        original_unregister = _mprt.unregister
        _mprt.unregister = lambda *args, **kwargs: None
        try:
            super().unlink()
        finally:
            _mprt.unregister = original_unregister


def trim_memory() -> bool:
    with contextlib.suppress(Exception):
        import ctypes

        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    return False


@contextlib.contextmanager
def profile_time(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        get_logger().debug(f"{label} took {time.perf_counter() - start:.2f}s")
