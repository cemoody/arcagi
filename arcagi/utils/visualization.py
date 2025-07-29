"""
Visualization utilities for ARC-AGI models.

This module provides visualization tools for model outputs, including
edge visualization, color model integration, and accuracy metrics display.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Import visualization tools
from utils.terminal_relshow import relshow


class ValidationVisualizer:
    """Handles visualization of validation results including edge visualization and color outputs."""

    def __init__(
        self,
        current_filename: Optional[str] = None,
        color_model_checkpoint_dir: str = "optimized_checkpoints",
    ):
        """Initialize visualizer with optional color model support."""
        self.current_filename = current_filename
        self.color_model_checkpoint_dir = color_model_checkpoint_dir

    def set_filename(self, filename: str) -> None:
        """Set the current filename for color model visualization."""
        self.current_filename = filename

    def visualize_with_color_model(
        self, features: torch.Tensor, filename: str, title: str
    ) -> None:
        """Visualize features using the color model."""
        # Import color model function locally to avoid circular imports
        from color_mapping.ex17 import load_color_model_for_inference

        try:
            color_model = load_color_model_for_inference(
                filename, checkpoint_dir=self.color_model_checkpoint_dir
            )

            with torch.no_grad():
                colors = color_model.predict_colors(features.unsqueeze(0))[0]
                masks = color_model.predict_masks(features.unsqueeze(0))[0]

                # Convert to numpy for visualization
                colors_np = colors.cpu().numpy()
                masks_np = (masks > 0.5).cpu().numpy().astype(bool)

                # Create color map (fallback if get_color_palette doesn't exist)
                try:
                    color_map = color_model.get_color_palette()
                except AttributeError:
                    # Fallback color map
                    color_map = {
                        0: "black",
                        1: "blue",
                        2: "red",
                        3: "green",
                        4: "yellow",
                        5: "gray",
                    }

                # Display the colored output
                self._display_colored_output(colors_np, masks_np, color_map, title)

        except Exception as e:
            print(f"Error in color model visualization for {title}: {e}")

    def _display_colored_output(
        self,
        colors: np.ndarray,
        masks: np.ndarray,
        color_map: Dict[int, str],
        title: str,
    ) -> None:
        """Display colored output in terminal using colored visualization."""
        # Use colored terminal display function (now required import)
        from arcagi.utils.terminal_imshow import imshow

        # Create a display matrix where masked regions are -1 (background)
        # and unmasked regions show the color values
        display_matrix = np.full_like(colors, -1)  # Fill with background value
        display_matrix[masks] = colors[masks]  # Show colors only in mask regions

        # Convert to torch tensor for imshow
        display_tensor = torch.from_numpy(display_matrix).int()

        # Display with colors
        print(f"🎨 Using colored terminal display for {title}")
        imshow(display_tensor, title=f"{title} - {self.current_filename}")

    def visualize_validation_examples(
        self, validation_examples: List[Dict[str, Any]], current_epoch: int
    ) -> None:
        """Visualize validation examples with edge comparison."""
        if not validation_examples:
            return

        print(
            f"\n\033[1m========== Edge Visualization - Test Subset (Epoch {current_epoch}) ==========\033[0m"
        )

        for i, example in enumerate(validation_examples):
            self._visualize_single_example(example, i)

    def _visualize_single_example(
        self, example: Dict[str, Any], viz_index: int
    ) -> None:
        """Visualize a single validation example."""
        example_idx = example["example_idx"]
        pred_edges = example["predicted_edges"]
        target_edges = example["target_edges"]
        mask = example["mask"]

        print(
            f"\n\033[1mTest Example {example_idx} (Validation #{viz_index+1}):\033[0m"
        )

        # Only show non-mask regions for better visualization
        if mask.any():
            self._visualize_masked_regions(
                example, pred_edges, target_edges, mask, example_idx
            )
        else:
            print(f"  (No non-mask pixels to visualize)")

    def _visualize_masked_regions(
        self,
        example: Dict[str, Any],
        pred_edges: torch.Tensor,
        target_edges: torch.Tensor,
        mask: torch.Tensor,
        example_idx: int,
    ) -> None:
        """Visualize the masked regions of predicted vs target edges."""
        # Find bounding box of non-mask region
        mask_coords = torch.where(mask)
        if len(mask_coords[0]) > 0:
            min_y, max_y = mask_coords[0].min().item(), mask_coords[0].max().item()
            min_x, max_x = mask_coords[1].min().item(), mask_coords[1].max().item()

            # Extract the region of interest
            target_region = target_edges[min_y : max_y + 1, min_x : max_x + 1]
            pred_region = pred_edges[min_y : max_y + 1, min_x : max_x + 1]

            # Convert predictions to binary (threshold at 0.5)
            pred_region_binary = (pred_region > 0.5).float()

            try:
                self._display_comparison(
                    example, example_idx, target_region, pred_region_binary
                )
            except Exception as e:
                print(f"Error visualizing example {example_idx}: {e}")

    def _display_comparison(
        self,
        example: Dict[str, Any],
        example_idx: int,
        target_region: torch.Tensor,
        pred_region_binary: torch.Tensor,
    ) -> None:
        """Display comparison between expected and predicted outputs."""
        print(f"Expected output (full 30x30 image):")
        if self.current_filename and "target_full" in example:
            # Show full 30x30 features to color model
            self.visualize_with_color_model(
                example["target_full"],  # Full 30x30 features
                self.current_filename,
                f"Expected_Ex{example_idx}",
            )
        else:
            relshow(target_region, title=None)

        print(f"Predicted output (full 30x30 image):")
        if self.current_filename and "predicted_full" in example:
            # Show full 30x30 features to color model
            self.visualize_with_color_model(
                example["predicted_full"],  # Full 30x30 features
                self.current_filename,
                f"Predicted_Ex{example_idx}",
            )
        else:
            relshow(pred_region_binary, title=None)


class AccuracyMetricsCalculator:
    """Calculates and displays color and mask accuracy metrics using color models."""

    def __init__(self, color_model_checkpoint_dir: str = "optimized_checkpoints"):
        """Initialize with color model checkpoint directory."""
        self.color_model_checkpoint_dir = color_model_checkpoint_dir

    def calculate_and_print_metrics(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
        example_idx: int,
        predicted_mask_logits: torch.Tensor,
        current_filename: str,
    ) -> Dict[str, float]:
        """Calculate and print comprehensive accuracy metrics."""
        # Import color model function locally to avoid circular imports
        from color_mapping.ex17 import load_color_model_for_inference

        try:
            # Load the color model for this filename
            color_model = load_color_model_for_inference(
                current_filename, checkpoint_dir=self.color_model_checkpoint_dir
            )

            # Calculate metrics
            metrics = self._compute_accuracy_metrics(
                predicted_features,
                target_features,
                target_mask,
                predicted_mask_logits,
                color_model,
            )

            # Print formatted results
            self._print_formatted_metrics(metrics, example_idx)

            return metrics

        except Exception as e:
            print(f"  ❌ Could not calculate accuracy metrics: {e}")
            return {}

    def _compute_accuracy_metrics(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
        predicted_mask_logits: torch.Tensor,
        color_model,
    ) -> Dict[str, float]:
        """Compute all accuracy metrics."""
        with torch.no_grad():
            # Get color predictions
            pred_colors = color_model.predict_colors(predicted_features.unsqueeze(0))[0]
            target_colors = color_model.predict_colors(target_features.unsqueeze(0))[0]

            # Convert mask logits to binary mask
            pred_mask = (predicted_mask_logits.squeeze(-1) > 0).float()
            target_mask = target_mask.to(pred_colors.device)

            # Calculate mask accuracy
            mask_correct = (pred_mask == target_mask).sum().item()
            mask_total = target_mask.numel()
            mask_accuracy = mask_correct / mask_total if mask_total > 0 else 0.0

            # Calculate color accuracy in predicted mask region
            pred_mask_area = pred_mask.sum().item()
            if pred_mask_area > 0:
                # Convert to bool to fix bitwise operation
                pred_mask_bool = pred_mask.bool()
                color_match = pred_colors == target_colors
                color_correct_in_pred_mask = (color_match & pred_mask_bool).sum().item()
                color_accuracy_in_pred_mask = (
                    color_correct_in_pred_mask / pred_mask_area
                )
            else:
                color_accuracy_in_pred_mask = 0.0
                color_correct_in_pred_mask = 0

            # Calculate color accuracy in target mask region
            target_mask_area = target_mask.sum().item()
            if target_mask_area > 0:
                # Convert to bool to fix bitwise operation
                target_mask_bool = target_mask.bool()
                color_match = pred_colors == target_colors
                color_correct_in_target_mask = (
                    (color_match & target_mask_bool).sum().item()
                )
                color_accuracy_in_target_mask = (
                    color_correct_in_target_mask / target_mask_area
                )
            else:
                color_accuracy_in_target_mask = 0.0
                color_correct_in_target_mask = 0

            return {
                "mask_accuracy": mask_accuracy,
                "mask_correct": mask_correct,
                "mask_total": mask_total,
                "color_accuracy_pred_mask": color_accuracy_in_pred_mask,
                "color_correct_pred_mask": color_correct_in_pred_mask,
                "pred_mask_area": pred_mask_area,
                "color_accuracy_target_mask": color_accuracy_in_target_mask,
                "color_correct_target_mask": color_correct_in_target_mask,
                "target_mask_area": target_mask_area,
            }

    def _print_formatted_metrics(
        self, metrics: Dict[str, float], example_idx: int
    ) -> None:
        """Print formatted accuracy metrics."""
        print(f"\n📊 Accuracy Metrics for Example {example_idx}:")
        print(
            f"  🎯 Mask Accuracy: {metrics['mask_accuracy']:.1%} "
            f"({metrics['mask_correct']:.0f}/{metrics['mask_total']:.0f} pixels)"
        )
        print(
            f"  🎨 Color Accuracy (in predicted mask): {metrics['color_accuracy_pred_mask']:.1%} "
            f"({metrics['color_correct_pred_mask']:.0f}/{metrics['pred_mask_area']:.0f} pixels)"
        )
        print(
            f"  🎨 Color Accuracy (in target mask): {metrics['color_accuracy_target_mask']:.1%} "
            f"({metrics['color_correct_target_mask']:.0f}/{metrics['target_mask_area']:.0f} pixels)"
        )

        # Additional info about mask differences
        if metrics["pred_mask_area"] != metrics["target_mask_area"]:
            diff = metrics["pred_mask_area"] - metrics["target_mask_area"]
            print(f"  ℹ️  Predicted mask size differs by {diff:+.0f} pixels from target")
