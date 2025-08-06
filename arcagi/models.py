"""Data models for ARC-AGI tasks."""

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    """Dataclass to hold output batch data with proper typing."""

    one: torch.Tensor  # one hot [B, 30, 30, 10]
    fea: torch.Tensor  # order2 features [B, 30, 30, 147]
    col: torch.Tensor  # colors, int array [B, 30, 30]
    msk: torch.Tensor  # masks, bool array [B, 30, 30]
    colf: torch.Tensor  # colors, flattened [B, 30*30]
    mskf: torch.Tensor  # masks, flattened [B, 30*30]
    idx: torch.Tensor  # indices [B]


@dataclass
class BatchData:
    """Dataclass to hold complete batch data with proper typing."""

    inp: Batch
    out: Batch
