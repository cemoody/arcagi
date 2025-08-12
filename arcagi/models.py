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
    transform_idx: torch.Tensor | None = None  # D4 transform indices [B], values 0-7


@dataclass
class BatchData:
    """Dataclass to hold complete batch data with proper typing."""

    inp: Batch
    out: Batch


def tile(batch: BatchData, n: int = 1) -> BatchData:
    """Tile a BatchData by concatenating it n times along the batch dimension.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    if n == 1:
        return batch
    
    # Tile input batch
    inp_tiled = Batch(
        one=batch.inp.one.repeat(n, 1, 1, 1),
        fea=batch.inp.fea.repeat(n, 1, 1, 1),
        col=batch.inp.col.repeat(n, 1, 1),
        msk=batch.inp.msk.repeat(n, 1, 1),
        colf=batch.inp.colf.repeat(n, 1),
        mskf=batch.inp.mskf.repeat(n, 1),
        idx=batch.inp.idx.repeat(n),
        transform_idx=batch.inp.transform_idx.repeat(n) if batch.inp.transform_idx is not None else None,
    )
    
    # Tile output batch
    out_tiled = Batch(
        one=batch.out.one.repeat(n, 1, 1, 1),
        fea=batch.out.fea.repeat(n, 1, 1, 1),
        col=batch.out.col.repeat(n, 1, 1),
        msk=batch.out.msk.repeat(n, 1, 1),
        colf=batch.out.colf.repeat(n, 1),
        mskf=batch.out.mskf.repeat(n, 1),
        idx=batch.out.idx.repeat(n),
        transform_idx=batch.out.transform_idx.repeat(n) if batch.out.transform_idx is not None else None,
    )
    
    return BatchData(inp=inp_tiled, out=out_tiled)
