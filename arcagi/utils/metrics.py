"""Metrics utilities for training and evaluation."""

from typing import TYPE_CHECKING, Dict, List, cast

import torch

if TYPE_CHECKING:
    from arcagi.models import BatchData


def training_index_metrics(
    i: "BatchData",
    col_log: torch.Tensor,
    msk_log: torch.Tensor,
    prefix: str,
    metrics_dict: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """Compute metrics for training indices.

    Tracks per-example averages:
    - Average active pixels per appearance
    - Average incorrect predictions per appearance
    - Count of times seen
    """
    # Get predictions
    col_pred = col_log.argmax(dim=-1)  # [B, 30, 30]
    msk_pred = (msk_log > 0).squeeze(-1)  # [B, 30, 30]

    # Calculate metrics per batch item
    incorrect_color_pixels = col_pred != i.inp.col  # (B, 30, 30)
    incorrect_mask_pixels = msk_pred != i.inp.msk  # (B, 30, 30)
    incorrect_color_mask = incorrect_color_pixels | i.inp.msk  # (B, 30, 30)
    incorrect_num_color = incorrect_color_mask.sum(dim=(1, 2))  # [B]
    incorrect_num_mask = incorrect_mask_pixels.sum(dim=(1, 2))  # [B]

    # Group by unique indices (handles duplicates in batch)
    unique_indices: List[int] = torch.unique(i.inp.idx).tolist()  # type: ignore

    # Aggregate metrics for each unique index
    for unique_index in unique_indices:
        unique_index = cast(int, unique_index)  # Ensure type checker knows it's int
        # Get mask for this example index in the batch
        idx_mask = i.inp.idx == unique_index
        count = int(idx_mask.item())

        n_masked_pixels = i.inp.msk[idx_mask].sum().item()
        n_incorrect_num_color = incorrect_num_color[idx_mask].sum().item()
        n_incorrect_num_mask = incorrect_num_mask[idx_mask].sum().item()

        # Update metrics dict
        if unique_index not in metrics_dict:
            metrics_dict[unique_index] = {}

        # Get existing metrics
        m = metrics_dict[unique_index]

        m[f"{prefix}_n_masked_pixels"] = n_masked_pixels
        m[f"{prefix}_n_incorrect_num_color"] = n_incorrect_num_color
        m[f"{prefix}_n_incorrect_num_mask"] = n_incorrect_num_mask

    return metrics_dict


def get_average_incorrect_pixels(
    metrics_dict: Dict[int, Dict[str, float]], prefix: str, example_idx: int
) -> float:
    """Get average incorrect pixels per example.

    Since metrics are already stored as averages, just return the value directly.
    """
    if example_idx not in metrics_dict:
        return 0.0

    metrics = metrics_dict[example_idx]
    return metrics.get(f"{prefix}_pixels_incorrect", 0.0)