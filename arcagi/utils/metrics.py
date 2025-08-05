"""Metrics utilities for training and evaluation."""

from typing import TYPE_CHECKING, Dict, List, Literal

import torch

if TYPE_CHECKING:
    from arcagi.models import BatchData


def training_index_metrics(
    i: "BatchData",
    col_log: torch.Tensor,
    msk_log: torch.Tensor,
    prefix: Literal["input", "output"],
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

    # Determine which target to use based on prefix
    if prefix == "input":
        target_col = i.inp.col
        target_msk = i.inp.msk
        target_idx = i.inp.idx
    else:  # prefix == "output"
        target_col = i.out.col
        target_msk = i.out.msk
        target_idx = i.out.idx

    # Calculate metrics per batch item
    incorrect_color_pixels = col_pred != target_col  # (B, 30, 30)
    incorrect_mask_pixels = msk_pred != target_msk  # (B, 30, 30)
    # Count incorrect colors only within the mask (where mask is True)
    incorrect_color_within_mask = incorrect_color_pixels & target_msk  # (B, 30, 30)
    incorrect_num_color = incorrect_color_within_mask.sum(dim=(1, 2))  # [B]
    incorrect_num_mask = incorrect_mask_pixels.sum(dim=(1, 2))  # [B]

    # Group by unique indices (handles duplicates in batch)
    unique_indices: List[int] = torch.unique(target_idx).tolist()  # type: ignore

    # Aggregate metrics for each unique index
    for unique_index in unique_indices:
        # Get mask for this example index in the batch
        idx_mask = target_idx == unique_index
        count = int(idx_mask.sum().item())

        n_masked_pixels = target_msk[idx_mask].sum().item()
        n_incorrect_num_color = incorrect_num_color[idx_mask].sum().item()
        n_incorrect_num_mask = incorrect_num_mask[idx_mask].sum().item()

        # Update metrics dict
        if unique_index not in metrics_dict:
            metrics_dict[unique_index] = {}

        # Get existing metrics
        m = metrics_dict[unique_index]

        m[f"{prefix}_n_masked_pixels"] = n_masked_pixels / count
        m[f"{prefix}_n_incorrect_num_color"] = n_incorrect_num_color / count
        m[f"{prefix}_n_incorrect_num_mask"] = n_incorrect_num_mask / count

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
