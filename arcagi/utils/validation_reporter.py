"""
Validation reporting utilities for ARC-AGI models.

This module handles the complex validation summary reporting that was previously
embedded in the MetricsTracker class.
"""

from typing import Any, Callable, Dict, List, Optional


class ValidationReporter:
    """Handles formatted validation reporting for both single-file and multi-file modes."""

    def __init__(self, wandb_logger: Optional[Callable[[str, float], None]] = None):
        """Initialize reporter with optional wandb logging."""
        self.wandb_logger = wandb_logger

    def print_single_file_summary(
        self, validation_outputs: List[Dict[str, Any]], current_epoch: int
    ) -> None:
        """Print detailed per-example validation summary for single-file mode."""
        if not validation_outputs:
            return

        print(f"\n=== Validation Epoch {current_epoch} Summary ===")

        # Group by example index
        per_example_data = self._group_by_example(validation_outputs)

        # Print header (without loss since it's not per-example)
        print(f"{'Ex':<3} {'Order2':<8} {'Mask':<8} {'Incorrect':<9} {'MaskErr':<8}")
        print("-" * 43)

        # Print per-example averages
        total_order2 = 0.0
        total_mask = 0.0
        total_correct = 0.0
        total_mask_err = 0.0

        for idx in sorted(per_example_data.keys()):
            example_data = per_example_data[idx]
            # Loss is not available per-example, skip it
            avg_order2 = sum(d["order2_accuracy"] for d in example_data) / len(
                example_data
            )
            avg_mask = sum(d["mask_accuracy"] for d in example_data) / len(example_data)
            # Use the incorrect percentage directly (now calculated in MetricsTracker)
            avg_incorrect = sum(d["correct_pixels"] for d in example_data) / len(
                example_data
            )
            avg_mask_err = sum(d["mask_incorrect"] for d in example_data) / len(
                example_data
            )

            print(
                f"{idx:<3} {avg_order2:<8.4f} {avg_mask:<8.4f} {avg_incorrect:<9.1f} {avg_mask_err:<8.1f}"
            )

            total_order2 += avg_order2
            total_mask += avg_mask
            total_correct += avg_incorrect  # Now tracking incorrect pixels
            total_mask_err += avg_mask_err

        # Print overall averages
        num_examples = len(per_example_data)
        if num_examples > 0:
            print("-" * 43)
            print(
                f"{'Avg':<3} {total_order2/num_examples:<8.4f} "
                f"{total_mask/num_examples:<8.4f} {total_correct/num_examples:<9.1f} {total_mask_err/num_examples:<8.1f}"
            )

            # Log to wandb if available (skip loss since it's not per-example)
            if self.wandb_logger:
                self.wandb_logger("val_order2_accuracy", total_order2 / num_examples)
                self.wandb_logger("val_mask_accuracy", total_mask / num_examples)
                self.wandb_logger("val_correct_pixels", total_correct / num_examples)

    def print_multi_file_summary(
        self, validation_outputs: List[Dict[str, Any]], current_epoch: int
    ) -> None:
        """Print aggregated validation summary for multi-file mode."""
        if not validation_outputs:
            return

        print(f"\n=== Validation Epoch {current_epoch} Summary (Multi-file) ===")

        # Calculate overall averages - handle new data structure
        total_loss = 0.0
        total_order2 = 0.0
        total_mask = 0.0
        total_correct = 0.0
        total_mask_err = 0.0
        count = 0

        for output in validation_outputs:
            if "per_example_data" in output:
                for example_data in output["per_example_data"]:
                    total_order2 += example_data["order2_accuracy"]
                    total_mask += example_data["mask_accuracy"]
                    total_correct += example_data["correct_pixels"]
                    total_mask_err += example_data["mask_incorrect"]
                    count += 1
                # Loss is at the output level, not per-example
                total_loss += output["val_loss"] * len(output["per_example_data"])
            else:
                # Fallback for old format
                total_loss += output["loss"]
                total_order2 += output["order2_accuracy"]
                total_mask += output["mask_accuracy"]
                total_correct += output["correct_pixels"]
                total_mask_err += output["mask_incorrect"]
                count += 1

        print(f"Validation samples: {count}")
        print(f"Average loss: {total_loss / count:.4f}")
        print(f"Average order2 accuracy: {total_order2 / count:.4f}")
        print(f"Average mask accuracy: {total_mask / count:.4f}")
        print(f"Average correct pixels: {total_correct / count:.4f}")
        print(f"Average mask errors: {total_mask_err / count:.1f}")

        # Log to wandb if available
        if self.wandb_logger:
            self.wandb_logger("val_loss", total_loss / count)
            self.wandb_logger("val_order2_accuracy", total_order2 / count)
            self.wandb_logger("val_mask_accuracy", total_mask / count)
            self.wandb_logger("val_correct_pixels", total_correct / count)

    def _group_by_example(
        self, validation_outputs: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group validation outputs by example index."""
        grouped = {}
        for output in validation_outputs:
            # Handle both old format (direct idx) and new format (per_example_data)
            if "per_example_data" in output:
                for example_data in output["per_example_data"]:
                    idx = example_data["index"]
                    if idx not in grouped:
                        grouped[idx] = []
                    grouped[idx].append(example_data)
            else:
                # Fallback for old format
                idx = output["idx"]
                if idx not in grouped:
                    grouped[idx] = []
                grouped[idx].append(output)
        return grouped


class MetricsFormatter:
    """Handles consistent formatting of training and validation metrics."""

    @staticmethod
    def format_training_summary(
        per_example_metrics: Dict[int, Dict[str, Any]], current_epoch: int
    ) -> str:
        """Format training metrics into a readable table."""
        if not per_example_metrics:
            return f"\n=== Training Epoch {current_epoch} Summary ===\nNo training data recorded."

        lines = [f"\n=== Training Epoch {current_epoch} Summary ==="]
        lines.append(
            f"{'Ex':<3} {'Loss':<8} {'Order2':<8} {'Mask':<8} {'Incorrect':<9} {'MaskErr':<8}"
        )
        lines.append("-" * 51)

        total_loss = 0.0
        total_order2 = 0.0
        total_mask = 0.0
        total_correct = 0.0
        total_mask_err = 0.0

        for idx in sorted(per_example_metrics.keys()):
            metrics = per_example_metrics[idx]
            count = metrics["count"]
            # Use actual MetricsTracker structure with sum/count
            avg_loss = metrics["loss_sum"] / count if count > 0 else 0.0
            avg_order2 = metrics["acc_sum"] / count if count > 0 else 0.0
            avg_mask = metrics["mask_acc_sum"] / count if count > 0 else 0.0
            avg_incorrect = (
                metrics["incorrect_pixels_sum"] / count if count > 0 else 0.0
            )
            avg_mask_err = metrics["mask_incorrect_sum"] / count if count > 0 else 0.0

            lines.append(
                f"{idx:<3} {avg_loss:<8.4f} {avg_order2:<8.4f} {avg_mask:<8.4f} {avg_incorrect:<9.1f} {avg_mask_err:<8.1f}"
            )

            total_loss += avg_loss
            total_order2 += avg_order2
            total_mask += avg_mask
            total_correct += avg_incorrect  # Now tracking incorrect pixels
            total_mask_err += avg_mask_err

        # Add overall averages
        num_examples = len(per_example_metrics)
        lines.append("-" * 52)
        lines.append(
            f"{'Avg':<3} {total_loss/num_examples:<8.4f} {total_order2/num_examples:<8.4f} "
            f"{total_mask/num_examples:<8.4f} {total_correct/num_examples:<9.1f} {total_mask_err/num_examples:<8.1f}"
        )

        return "\n".join(lines)
