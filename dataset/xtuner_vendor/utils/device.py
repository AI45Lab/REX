from __future__ import annotations

import torch


def get_torch_device_module():
    if torch.cuda.is_available():
        return torch.cuda
    return torch

