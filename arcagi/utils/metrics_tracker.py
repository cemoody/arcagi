"""
Training and validation metrics tracking utilities for ARC-AGI models.

This module provides a centralized way to track per-example metrics
across different training modes (single-file vs multi-file).
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from .validation_reporter import MetricsFormatter, ValidationReporter


class MetricsTracker:
    """
    Tracks per-example training and validation metrics for ARC-AGI models.

    Handles both single-file mode (where we track detailed per-example metrics)
    and multi-file mode (where we track aggregated metrics).
    """

    def __init__(
        self,
        single_file_threshold: int = 10,
        wandb_logger: Optional[Callable[[str, float], None]] = None,
    ):
        """Initialize tracker with automatic single-file vs multi-file mode detection."""
        self.single_file_threshold = single_file_threshold

        # Reporting components
        self.validation_reporter = ValidationReporter(wandb_logger)
        self.metrics_formatter = MetricsFormatter()

        self.reset()

    def reset(self) -> None:
        """Clear all accumulated metrics and start fresh for the next epoch."""
        # Training state
        self.training_indices_seen: Set[int] = set()
        self.training_example_metrics: Dict[int, Dict[str, float]] = {}

        # Validation state
        self.validation_outputs: List[Dict[str, Any]] = []

    def track_batch_indices(self, indices: torch.Tensor) -> None:
        """Add batch indices to running set for single-file mode detection."""
        batch_indices = indices.cpu().numpy()
        for idx in batch_indices:
            self.training_indices_seen.add(int(idx))

    def is_single_file_mode(self) -> bool:
        """True when training on few unique examples (enables detailed per-example tracking)."""
        return len(self.training_indices_seen) <= self.single_file_threshold

    def calculate_order2_accuracy(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute accuracy on binary order2 features (dims 0-35) within valid mask region."""
        if not mask.any():
            return torch.tensor(0.0, device=pred_features.device)

        pred_order2 = pred_features[:, :, :36]
        target_order2 = target_features[:, :, :36]
        pred_order2_binary = (pred_order2 > 0.5).float()

        correct = ((pred_order2_binary == target_order2) * mask.unsqueeze(-1)).sum()
        total = mask.sum() * 36

        return (
            correct / total
            if total > 0
            else torch.tensor(0.0, device=pred_features.device)
        )

    def calculate_incorrect_pixels(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Count pixels where NOT all 36 order2 features match exactly."""
        if not mask.any():
            return torch.tensor(0.0, device=pred_features.device)

        pred_order2 = pred_features[:, :, :36]
        target_order2 = target_features[:, :, :36]
        pred_order2_binary = (pred_order2 > 0.5).float()

        # Count pixels where NOT all 36 order2 features match
        all_features_match = (pred_order2_binary == target_order2).all(dim=-1)
        incorrect_pixels_count = ((~all_features_match) * mask).sum()

        return incorrect_pixels_count

    def calculate_mask_accuracy(
        self, pred_mask_logits: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (accuracy_fraction, incorrect_pixel_count) for binary mask prediction."""
        pred_mask = (pred_mask_logits.squeeze(-1) > 0).float()
        mask_correct = (pred_mask == target_mask.float()).sum()
        mask_total = torch.numel(target_mask)
        mask_incorrect = mask_total - mask_correct
        mask_accuracy = (
            mask_correct / mask_total
            if mask_total > 0
            else torch.tensor(0.0, device=pred_mask_logits.device)
        )

        return mask_accuracy, mask_incorrect

    def track_example_metrics(
        self,
        idx: int,
        loss: torch.Tensor,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        target_mask: torch.Tensor,
        color_accuracy: float = 0.0,
        color_incorrect: float = 0.0,
    ) -> None:
        """Accumulate running averages of per-example metrics (only in single-file mode)."""
        if not self.is_single_file_mode():
            return  # Don't track detailed metrics in multi-file mode

        if not target_mask.any():
            return  # Skip examples with no valid pixels

        # Calculate metrics
        order2_accuracy = self.calculate_order2_accuracy(
            pred_features, target_features, target_mask
        )
        mask_accuracy, mask_incorrect = self.calculate_mask_accuracy(
            pred_mask_logits, target_mask
        )

        # Calculate incorrect pixels (pixels where NOT all 36 order2 features match)
        incorrect_pixels = self.calculate_incorrect_pixels(
            pred_features, target_features, target_mask
        )

        # Initialize metrics dict if first time seeing this example
        if idx not in self.training_example_metrics:
            self.training_example_metrics[idx] = {
                "count": 0,
                "loss_sum": 0.0,
                "acc_sum": 0.0,
                "mask_acc_sum": 0.0,
                "incorrect_pixels_sum": 0.0,
                "color_acc_sum": 0.0,
                "mask_incorrect_sum": 0.0,
                "color_incorrect_sum": 0.0,
            }

        # Update metrics
        metrics = self.training_example_metrics[idx]
        metrics["count"] += 1
        metrics["loss_sum"] += loss.item()
        metrics["acc_sum"] += (
            order2_accuracy.item()
            if torch.is_tensor(order2_accuracy)
            else order2_accuracy
        )
        metrics["mask_acc_sum"] += (
            mask_accuracy.item() if torch.is_tensor(mask_accuracy) else mask_accuracy
        )
        metrics["incorrect_pixels_sum"] += (
            incorrect_pixels.item()
            if torch.is_tensor(incorrect_pixels)
            else incorrect_pixels
        )
        metrics["color_acc_sum"] += (
            color_accuracy.item() if torch.is_tensor(color_accuracy) else color_accuracy
        )
        metrics["mask_incorrect_sum"] += (
            mask_incorrect.item() if torch.is_tensor(mask_incorrect) else mask_incorrect
        )
        metrics["color_incorrect_sum"] += (
            color_incorrect.item()
            if torch.is_tensor(color_incorrect)
            else color_incorrect
        )

    def track_validation_step(
        self,
        loss: torch.Tensor,
        order2_accuracy: torch.Tensor,
        mask_accuracy: torch.Tensor,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        target_mask: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Track validation metrics for a single step."""
        # Calculate per-example metrics (enhanced to match training metrics)
        per_example_data = []
        for i in range(len(indices)):
            example_idx = indices[i].item()

            # Calculate non-mask pixels for this example
            valid_pixels = target_mask[i].sum().item()

            # Calculate correctly predicted pixels (for order2 features)
            if valid_pixels > 0:
                incorrect_pixels = self.calculate_incorrect_pixels(
                    pred_features[i : i + 1], target_features[i : i + 1], target_mask[i]
                )

                # Calculate order2 accuracy for this specific example
                example_order2_accuracy = self.calculate_order2_accuracy(
                    pred_features[i : i + 1], target_features[i : i + 1], target_mask[i]
                )
            else:
                incorrect_pixels = torch.tensor(0.0)
                example_order2_accuracy = torch.tensor(0.0)

            # Calculate mask accuracy for this example
            example_mask_accuracy, mask_incorrect = self.calculate_mask_accuracy(
                pred_mask_logits[i : i + 1], target_mask[i]
            )

            # Color accuracy placeholder (same as training)
            color_accuracy = 0.0
            color_incorrect = 0.0

            per_example_data.append(
                {
                    "index": example_idx,
                    "valid_pixels": valid_pixels,
                    "correct_pixels": (
                        incorrect_pixels.item()
                        if torch.is_tensor(incorrect_pixels)
                        else incorrect_pixels
                    ),
                    "order2_accuracy": (
                        example_order2_accuracy.item()
                        if torch.is_tensor(example_order2_accuracy)
                        else example_order2_accuracy
                    ),
                    "mask_accuracy": (
                        example_mask_accuracy.item()
                        if torch.is_tensor(example_mask_accuracy)
                        else example_mask_accuracy
                    ),
                    "color_accuracy": color_accuracy,
                    "mask_incorrect": (
                        mask_incorrect.item()
                        if torch.is_tensor(mask_incorrect)
                        else mask_incorrect
                    ),
                    "color_incorrect": color_incorrect,
                }
            )

        # Store outputs for epoch-level metrics
        self.validation_outputs.append(
            {
                "val_loss": loss,
                "val_order2_acc": order2_accuracy,
                "val_mask_acc": mask_accuracy,
                "per_example_data": per_example_data,
            }
        )

    def should_print_epoch_summary(self) -> bool:
        """True when training on sequential examples (0,1,2,3,4...) indicating single-file training."""
        if not self.training_indices_seen:
            return False

        unique_indices = sorted(self.training_indices_seen)

        # Check if we have indices 0-4, likely training on a single file
        return len(unique_indices) >= 5 and unique_indices == list(
            range(len(unique_indices))
        )

    def print_epoch_summary(self, current_epoch: int) -> None:
        """Display formatted table of per-example performance averaged across augmentations."""
        if not self.should_print_epoch_summary():
            return

        formatted_summary = self.metrics_formatter.format_training_summary(
            self.training_example_metrics, current_epoch
        )
        print(formatted_summary)

    def print_validation_summary(self, current_epoch: int) -> None:
        """Display formatted validation summary using ValidationReporter."""
        if self.is_single_file_mode():
            self.validation_reporter.print_single_file_summary(
                self.validation_outputs, current_epoch
            )
        else:
            self.validation_reporter.print_multi_file_summary(
                self.validation_outputs, current_epoch
            )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return diagnostic info about tracker state and detected training mode."""
        return {
            "unique_indices_count": len(self.training_indices_seen),
            "unique_indices": sorted(self.training_indices_seen),
            "is_single_file_mode": self.is_single_file_mode(),
            "tracked_examples_count": len(self.training_example_metrics),
            "should_print_summary": self.should_print_epoch_summary(),
            "validation_steps_tracked": len(self.validation_outputs),
        }
