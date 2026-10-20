# Copyright (c) OpenMMLab. All rights reserved.
import os
import logging
import sys
import threading

from dataset.xtuner_vendor.utils.dist_utils import get_rank


_LOGGER: logging.Logger | None = None
_LOGGER_LOCK = threading.Lock()


def log_format(module: str | None = None, rank: int | None = None) -> str:
    if rank is None:
        prefix = "[XTuner]"
    else:
        prefix = f"[XTuner][RANK {rank}]"

    formatter = f"{prefix}[%(asctime)s][%(levelname)s]"

    if module is not None:
        formatter += f"[{module}]"

    formatter += " %(message)s"
    return formatter


class _Rank0Logger:
    """Logger proxy that emits records only when the current process is rank 0.

    Use this for messages that are identical across distributed ranks — configuration
    warnings, env detection, deprecation notices, one-shot startup info. Non-rank-0
    processes skip the call entirely (so the message is absent from both stderr and the
    per-rank log file). For per-sample, per-rank, or per-process messages that
    genuinely differ across ranks, call ``get_logger()`` directly so every rank keeps
    its own record.
    """

    def debug(self, *args, **kwargs):
        if get_rank() == 0:
            get_logger().debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        if get_rank() == 0:
            get_logger().info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if get_rank() == 0:
            get_logger().warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if get_rank() == 0:
            get_logger().error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if get_rank() == 0:
            get_logger().critical(*args, **kwargs)


log_rank0 = _Rank0Logger()


def get_logger(level="INFO", log_dir=None, tag=None):
    global _LOGGER
    if _LOGGER is None:
        with _LOGGER_LOCK:
            if _LOGGER is None:
                log_level = os.environ.get("XTUNER_LOG_LEVEL", level).upper()
                logger = logging.getLogger("rex.xtuner_vendor")
                logger.propagate = False
                logger.setLevel(getattr(logging, log_level, logging.INFO))
                logger.handlers.clear()
                stream_handler = logging.StreamHandler(sys.stderr)
                stream_handler.setLevel(getattr(logging, log_level, logging.INFO))
                stream_handler.setFormatter(logging.Formatter(log_format()))
                logger.addHandler(stream_handler)
                _LOGGER = logger

    if log_dir is not None:
        log_level = os.environ.get("XTUNER_LOG_LEVEL", level).upper()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"rank_{get_rank()}.log"
        add_log_dir = all(
            not isinstance(handler, logging.FileHandler) or handler.baseFilename != str(log_path)
            for handler in _LOGGER.handlers
        )
        if add_log_dir:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(getattr(logging, log_level, logging.INFO))
            file_handler.setFormatter(logging.Formatter(log_format(rank=get_rank(), module=tag if tag else None)))
            _LOGGER.addHandler(file_handler)
    return _LOGGER
