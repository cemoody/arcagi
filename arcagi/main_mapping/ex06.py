import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
from utils.terminal_imshow import imshow
from utils.terminal_relshow import relshow

# Import the color model from ex17
try:
    from color_mapping.ex17 import load_color_model_for_inference
except ImportError:
    print(
        "Warning: Could not import color model from ex17. Visualization will be limited."
    )


class FeatureMappingModel(pl.LightningModule):
    """
    Model for mapping input features to output features using message passing.

    Key features:
    1. Message passing for spatial consistency
    2. Predicts both output_features and output_mask
    3. Heavily weights order2 features (first 36 dims) in the loss
    4. Based on ex17.py architecture but adapted for feature mapping
    5. Uses color models from ex17.py for visualization (dual color+mask prediction)
    """

    def __init__(
        self,
        input_feature_dim: int = 147,  # All concatenated features
        output_feature_dim: int = 147,  # We predict all features
        hidden_dim: int = 512,
        num_message_rounds: int = 24,
        dropout: float = 0.0,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        order2_weight: float = 10.0,  # Weight for order2 features in loss
        mask_weight: float = 1.0,  # Weight for mask prediction loss
    ):
        super().__init__()
        self.save_hyperparameters()

        # Feature dimensions
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.hidden_dim = hidden_dim
        self.num_message_rounds = num_message_rounds
        self.lr = lr
        self.weight_decay = weight_decay
        self.order2_weight = order2_weight
        self.mask_weight = mask_weight

        # Initial feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sinusoidal position embeddings - no parameters needed
        # Pre-compute the position embeddings for efficiency
        self.register_buffer(
            "pos_embed", self._create_sinusoidal_embeddings(hidden_dim)
        )

        # Weight-tied message passing
        self.shared_message_layer = SpatialMessagePassing(hidden_dim, dropout)

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_transform = nn.Linear(hidden_dim, hidden_dim)

        # Feature prediction head
        self.feature_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_feature_dim),
        )

        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        # For tracking validation outputs
        self.validation_outputs = []
        self.training_outputs = []
        self.validation_examples_for_viz = []  # Store examples for edge visualization

        # Store filename for visualization (can be set externally)
        self.current_filename = None

        # Better initialization
        self.apply(self._init_weights)

    def set_filename(self, filename: str) -> None:
        """Set the current filename for visualization."""
        self.current_filename = filename

    def set_color_model_checkpoint_dir(self, checkpoint_dir: str) -> None:
        """Set the checkpoint directory for loading color models."""
        self.color_model_checkpoint_dir = checkpoint_dir

    def _create_sinusoidal_embeddings(self, hidden_dim: int) -> torch.Tensor:
        """Create sinusoidal position embeddings for a 30x30 grid."""
        import math

        # Create position indices for x and y
        x_pos = torch.arange(30).float()
        y_pos = torch.arange(30).float()

        # Create 2D grid of positions
        grid_x, grid_y = torch.meshgrid(x_pos, y_pos, indexing="ij")

        # Initialize embedding tensor
        pos_embed = torch.zeros(30, 30, hidden_dim)

        # Half dimensions for x, half for y
        half_dim = hidden_dim // 2

        # Create div_term for frequency scaling
        div_term_x = torch.exp(
            torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim)
        )
        div_term_y = torch.exp(
            torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim)
        )

        # Apply sinusoidal functions to x coordinates
        pos_embed[:, :, 0:half_dim:2] = torch.sin(grid_x.unsqueeze(-1) * div_term_x)
        pos_embed[:, :, 1:half_dim:2] = torch.cos(grid_x.unsqueeze(-1) * div_term_x)

        # Apply sinusoidal functions to y coordinates
        pos_embed[:, :, half_dim::2] = torch.sin(grid_y.unsqueeze(-1) * div_term_y)
        pos_embed[:, :, half_dim + 1 :: 2] = torch.cos(
            grid_y.unsqueeze(-1) * div_term_y
        )

        return pos_embed

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict output features and mask.

        Args:
            features: Input features of shape [B, 30, 30, input_feature_dim]

        Returns:
            output_features: Predicted features of shape [B, 30, 30, output_feature_dim]
            output_mask: Predicted mask of shape [B, 30, 30, 1]
        """
        # Extract base features
        h = self.feature_extractor(features)  # [B, 30, 30, hidden_dim]

        # Add pre-computed sinusoidal position embeddings
        # pos_embed is already [30, 30, hidden_dim], just need to add batch dimension
        h = h + self.pos_embed.unsqueeze(0)

        # Spatial message passing with residual connections
        for i in range(self.num_message_rounds):
            h_new = self.shared_message_layer(h)
            # Gradually increase residual strength
            alpha = 0.3 + 0.1 * (i / self.num_message_rounds)
            h = h + alpha * h_new

        # Global context
        h_spatial = h.permute(0, 3, 1, 2).contiguous()  # [B, hidden_dim, 30, 30]
        global_context = (
            self.global_pool(h_spatial).squeeze(-1).squeeze(-1)
        )  # [B, hidden_dim]
        global_context = self.global_transform(global_context)  # [B, hidden_dim]

        # Add global context back
        h = h + global_context.unsqueeze(1).unsqueeze(2) * 0.1

        # Predict output features and mask
        output_features = self.feature_head(h)  # [B, 30, 30, output_feature_dim]
        output_mask_logits = self.mask_head(h)  # [B, 30, 30, 1]

        return output_features, output_mask_logits

    def compute_feature_loss(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute feature prediction loss with heavy weighting on order2 features.

        Args:
            pred_features: Predicted features [B, 30, 30, 147]
            target_features: Target features [B, 30, 30, 147]
            valid_mask: Binary mask indicating valid cells [B, 30, 30]

        Returns:
            total_loss: Weighted total loss
            order2_loss: Loss for order2 features (first 36 dims)
            other_loss: Loss for other features
        """
        # Expand valid_mask to match feature dimensions
        valid_mask = valid_mask.unsqueeze(-1)  # [B, 30, 30, 1]

        # Split features into order2 and others
        pred_order2 = pred_features[..., :36]  # First 36 dimensions
        target_order2 = target_features[..., :36]

        pred_other = pred_features[..., 36:]  # Remaining dimensions
        target_other = target_features[..., 36:]

        # Compute losses only on valid cells
        if valid_mask.any():
            # Order2 loss (MSE)
            order2_diff = (pred_order2 - target_order2) ** 2
            order2_loss = (order2_diff * valid_mask).sum() / valid_mask.sum()

            # Other features loss (MSE)
            other_diff = (pred_other - target_other) ** 2
            other_loss = (other_diff * valid_mask).sum() / valid_mask.sum()

            # Weighted total loss
            total_loss = self.order2_weight * order2_loss + other_loss
        else:
            total_loss = torch.tensor(0.0, device=pred_features.device)
            order2_loss = torch.tensor(0.0, device=pred_features.device)
            other_loss = torch.tensor(0.0, device=pred_features.device)

        return total_loss, order2_loss, other_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        (
            input_features,
            output_features,
            inputs_mask,
            outputs_mask,
            indices,
            subset_is_train,
        ) = batch

        # Track indices seen during training
        if not hasattr(self, "training_indices_seen"):
            self.training_indices_seen = set()
            self.training_example_metrics = {}

        # Track unique indices in this batch
        batch_indices = indices.cpu().numpy()
        for idx in batch_indices:
            self.training_indices_seen.add(int(idx))

        # Forward pass
        pred_features, pred_mask_logits = self(input_features)

        # Compute feature loss
        valid_mask = outputs_mask.float()
        feature_loss, order2_loss, other_loss = self.compute_feature_loss(
            pred_features, output_features, valid_mask
        )

        # Compute mask loss
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits.squeeze(-1), outputs_mask.float()
        )

        # Total loss
        loss = feature_loss + self.mask_weight * mask_loss

        # Track per-example metrics for single-file mode
        if len(self.training_indices_seen) <= 10:  # Likely single-file mode
            with torch.no_grad():
                # Calculate per-example accuracy for order2 features
                for i in range(len(indices)):
                    idx = int(indices[i].item())

                    if outputs_mask[i].any():
                        # Calculate order2 accuracy for this example
                        pred_order2 = pred_features[i, :, :, :36]
                        target_order2 = output_features[i, :, :, :36]
                        pred_order2_binary = (pred_order2 > 0.5).float()

                        correct = (
                            (pred_order2_binary == target_order2)
                            * outputs_mask[i].unsqueeze(-1)
                        ).sum()
                        total = outputs_mask[i].sum() * 36
                        accuracy = correct / total if total > 0 else 0.0

                        # Calculate mask accuracy for this example
                        pred_mask_single = (pred_mask_logits[i].squeeze(-1) > 0).float()
                        mask_correct = (
                            pred_mask_single == outputs_mask[i].float()
                        ).sum()
                        mask_total = torch.numel(outputs_mask[i])
                        mask_incorrect = mask_total - mask_correct
                        mask_accuracy = (
                            mask_correct / mask_total if mask_total > 0 else 0.0
                        )

                        # For color accuracy, we need the target colors which aren't directly available
                        # in the training step. For now, we'll set this to 0.0 and implement
                        # proper color accuracy calculation in a separate validation method.
                        color_accuracy = 0.0
                        color_incorrect = 0.0  # Placeholder for color incorrect count

                        # Store metrics
                        if idx not in self.training_example_metrics:
                            self.training_example_metrics[idx] = {
                                "count": 0,
                                "loss_sum": 0.0,
                                "acc_sum": 0.0,
                                "mask_acc_sum": 0.0,
                                "color_acc_sum": 0.0,
                                "mask_incorrect_sum": 0.0,
                                "color_incorrect_sum": 0.0,
                            }

                        self.training_example_metrics[idx]["count"] += 1
                        self.training_example_metrics[idx]["loss_sum"] += loss.item()
                        self.training_example_metrics[idx]["acc_sum"] += (
                            accuracy.item() if torch.is_tensor(accuracy) else accuracy
                        )
                        self.training_example_metrics[idx]["mask_acc_sum"] += (
                            mask_accuracy.item()
                            if torch.is_tensor(mask_accuracy)
                            else mask_accuracy
                        )
                        self.training_example_metrics[idx]["color_acc_sum"] += (
                            color_accuracy.item()
                            if torch.is_tensor(color_accuracy)
                            else color_accuracy
                        )
                        self.training_example_metrics[idx]["mask_incorrect_sum"] += (
                            mask_incorrect.item()
                            if torch.is_tensor(mask_incorrect)
                            else mask_incorrect
                        )
                        self.training_example_metrics[idx]["color_incorrect_sum"] += (
                            color_incorrect.item()
                            if torch.is_tensor(color_incorrect)
                            else color_incorrect
                        )

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_feature_loss",
            feature_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train_order2_loss",
            order2_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train_other_loss", other_loss, on_step=True, on_epoch=True, sync_dist=False
        )
        self.log(
            "train_mask_loss", mask_loss, on_step=True, on_epoch=True, sync_dist=False
        )

        # Additional per-step logging
        self.log(
            "batch_idx", float(batch_idx), on_step=True, on_epoch=False, sync_dist=False
        )
        self.log(
            "global_step",
            float(self.global_step),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

        # Log learning rate
        if self.trainer and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", lr, on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def visualize_with_color_model(
        self,
        features: torch.Tensor,
        filename: str,
        title: str = "Predicted Colors",
        region_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Visualize features using the trained color model.

        Args:
            features: Feature tensor of shape (30, 30, 147)
            filename: Filename for loading the appropriate color model
            title: Title for the visualization
            region_bounds: Optional (min_y, max_y, min_x, max_x) to show only a region
        """
        try:
            # Load the color model for this filename
            checkpoint_dir = getattr(
                self, "color_model_checkpoint_dir", "optimized_checkpoints"
            )
            color_model = load_color_model_for_inference(
                filename, checkpoint_dir=checkpoint_dir
            )

            # Features should already be 30x30
            if features.shape[0] != 30 or features.shape[1] != 30:
                print(f"Warning: Expected 30x30 features, got {features.shape}")

            # Predict colors from features (ex17 model returns both colors and masks)
            predicted_colors = color_model.predict_colors(
                features.unsqueeze(0)
            )  # Add batch dim - this still works the same way

            # Convert to torch tensor for imshow
            predicted_colors_tensor = predicted_colors[
                0
            ]  # Remove batch dim, keep as tensor

            # If region bounds specified, extract that region for visualization
            if region_bounds is not None:
                min_y, max_y, min_x, max_x = region_bounds
                predicted_colors_region = predicted_colors_tensor[
                    min_y : max_y + 1, min_x : max_x + 1
                ]
                region_str = f" (region {min_y}:{max_y+1}, {min_x}:{max_x+1})"
            else:
                predicted_colors_region = predicted_colors_tensor
                region_str = ""

            # Use terminal imshow to display colors
            display_title = f"{title} - {filename}{region_str}"
            imshow(predicted_colors_region, title=display_title, show_legend=True)

        except Exception as e:
            print(f"❌ Could not visualize with color model for {filename}: {e}")
            print("Falling back to relshow...")
            # Fallback to relshow if color model fails
            # For relshow, we need the edge features (last 8 dimensions)
            if features.shape[-1] >= 8:
                # If region bounds specified, extract that region
                if region_bounds is not None:
                    min_y, max_y, min_x, max_x = region_bounds
                    edge_features = features[min_y : max_y + 1, min_x : max_x + 1, -8:]
                else:
                    edge_features = features[..., -8:]
                relshow(edge_features, title=title)
            else:
                print(f"Cannot use relshow either - features shape: {features.shape}")
                print(f"Need at least 8 feature dimensions for relshow")

    def on_train_epoch_end(self) -> None:
        """Show per-example metrics during training for single-file mode."""
        # Check if we're in single-file mode by counting unique indices
        if hasattr(self, "training_indices_seen"):
            unique_indices = sorted(self.training_indices_seen)

            # If we have indices 0-4, we're likely training on a single file
            if len(unique_indices) >= 5 and unique_indices == list(
                range(len(unique_indices))
            ):
                print(
                    f"\nTraining Epoch {self.current_epoch} - Per-Example Performance:"
                )
                print("(Note: These are averaged across all augmented copies)")
                print(
                    f"{'Example':<10} {'Seen':<10} {'Avg Loss':<15} {'Avg Order2 Acc':<20} {'Avg Mask Acc':<15} {'Avg Color Acc':<15} {'Mask Incorrect':<15} {'Color Incorrect':<15}"
                )
                print("-" * 125)

                for idx in unique_indices:
                    if idx in self.training_example_metrics:
                        metrics = self.training_example_metrics[idx]
                        count = metrics["count"]
                        avg_loss = metrics["loss_sum"] / count
                        avg_acc = metrics["acc_sum"] / count
                        avg_mask_acc = metrics["mask_acc_sum"] / count
                        avg_color_acc = metrics["color_acc_sum"] / count
                        avg_mask_incorrect = metrics["mask_incorrect_sum"] / count
                        avg_color_incorrect = metrics["color_incorrect_sum"] / count
                        print(
                            f"{idx:<10} {count:<10} {avg_loss:<15.3f} {avg_acc:<20.2%} {avg_mask_acc:<15.2%} {avg_color_acc:<15.2%} {avg_mask_incorrect:<15.1f} {avg_color_incorrect:<15.1f}"
                        )

            # Reset for next epoch
            self.training_indices_seen = set()
            self.training_example_metrics = {}

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        (
            input_features,
            output_features,
            inputs_mask,
            outputs_mask,
            indices,
            subset_is_train,
        ) = batch

        # Forward pass
        pred_features, pred_mask_logits = self(input_features)

        # Compute feature loss
        valid_mask = outputs_mask.float()
        feature_loss, order2_loss, other_loss = self.compute_feature_loss(
            pred_features, output_features, valid_mask
        )

        # Compute mask loss
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits.squeeze(-1), outputs_mask.float()
        )

        # Total loss
        loss = feature_loss + self.mask_weight * mask_loss

        # Compute mask accuracy
        pred_mask = (pred_mask_logits.squeeze(-1) > 0).float()
        mask_accuracy = (pred_mask == outputs_mask.float()).float().mean()

        # Compute feature accuracy for order2 features (binary features)
        if valid_mask.any():
            pred_order2 = pred_features[..., :36]
            target_order2 = output_features[..., :36]

            # For binary features, round predictions
            pred_order2_binary = (pred_order2 > 0.5).float()
            order2_accuracy = (
                (pred_order2_binary == target_order2) * valid_mask.unsqueeze(-1)
            ).sum() / (valid_mask.sum() * 36)
        else:
            order2_accuracy = torch.tensor(0.0)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_feature_loss", feature_loss, on_step=False, on_epoch=True)
        self.log("val_order2_loss", order2_loss, on_step=False, on_epoch=True)
        self.log("val_other_loss", other_loss, on_step=False, on_epoch=True)
        self.log("val_mask_loss", mask_loss, on_step=False, on_epoch=True)
        self.log(
            "val_mask_acc", mask_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_order2_acc",
            order2_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Calculate per-example metrics (enhanced to match training metrics)
        per_example_data = []
        for i in range(len(indices)):
            example_idx = indices[i].item()

            # Calculate non-mask pixels for this example
            valid_pixels = outputs_mask[i].sum().item()

            # Calculate correctly predicted pixels (for order2 features)
            if valid_pixels > 0:
                pred_order2_ex = pred_features[i, :, :, :36]
                target_order2_ex = output_features[i, :, :, :36]
                pred_order2_binary_ex = (pred_order2_ex > 0.5).float()

                # Count pixels where ALL 36 order2 features match
                all_features_match = (pred_order2_binary_ex == target_order2_ex).all(
                    dim=-1
                )
                correct_pixels = (all_features_match * outputs_mask[i]).sum()

                # Calculate order2 accuracy
                order2_correct = (
                    (pred_order2_binary_ex == target_order2_ex)
                    * outputs_mask[i].unsqueeze(-1)
                ).sum()
                order2_total = outputs_mask[i].sum() * 36
                order2_accuracy = (
                    order2_correct / order2_total if order2_total > 0 else 0.0
                )
            else:
                correct_pixels = 0
                order2_accuracy = 0.0

            # Calculate mask accuracy for this example
            pred_mask_single = (pred_mask_logits[i].squeeze(-1) > 0).float()
            mask_correct = (pred_mask_single == outputs_mask[i].float()).sum()
            mask_total = torch.numel(outputs_mask[i])
            mask_incorrect = mask_total - mask_correct
            mask_accuracy = mask_correct / mask_total if mask_total > 0 else 0.0

            # Color accuracy placeholder (same as training)
            color_accuracy = 0.0
            color_incorrect = 0.0

            per_example_data.append(
                {
                    "index": example_idx,
                    "valid_pixels": valid_pixels,
                    "correct_pixels": (
                        correct_pixels.item()
                        if torch.is_tensor(correct_pixels)
                        else correct_pixels
                    ),
                    "order2_accuracy": (
                        order2_accuracy.item()
                        if torch.is_tensor(order2_accuracy)
                        else order2_accuracy
                    ),
                    "mask_accuracy": (
                        mask_accuracy.item()
                        if torch.is_tensor(mask_accuracy)
                        else mask_accuracy
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

        # Store first few examples for edge visualization (limit to avoid memory issues)
        # Only store examples from test subset (subset_is_train is False)
        if len(self.validation_examples_for_viz) < 5:
            for i in range(min(2, len(indices))):  # Store up to 2 examples per batch
                if len(self.validation_examples_for_viz) < 5 and not subset_is_train[i]:
                    # Extract order2 edge features (last 8 dimensions: indices 36-43)
                    # These represent center-to-neighbor mask detection features
                    pred_edges = pred_features[i, :, :, 36:44]  # Shape: (30, 30, 8)
                    target_edges = output_features[i, :, :, 36:44]  # Shape: (30, 30, 8)
                    mask = outputs_mask[i]  # Shape: (30, 30)

                    # Also store full feature vectors for color visualization
                    pred_full = pred_features[i].detach().cpu()  # Shape: (30, 30, 147)
                    target_full = output_features[i].cpu()  # Shape: (30, 30, 147)
                    pred_mask_logits_full = pred_mask_logits[i].detach().cpu()

                    self.validation_examples_for_viz.append(
                        {
                            "example_idx": indices[i].item(),
                            "predicted_edges": pred_edges.detach().cpu(),
                            "target_edges": target_edges.cpu(),
                            "predicted_full": pred_full,
                            "target_full": target_full,
                            "mask": mask.cpu(),
                            "is_test_subset": True,
                            "predicted_mask_logits": pred_mask_logits_full,
                        }
                    )

        return {"val_loss": loss}

    def on_validation_epoch_start(self) -> None:
        """Reset validation outputs at the start of each epoch."""
        self.validation_outputs = []
        self.validation_examples_for_viz = []

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log validation metrics."""
        if not self.validation_outputs:
            return

        # Calculate average metrics
        avg_loss = torch.stack(
            [out["val_loss"] for out in self.validation_outputs]
        ).mean()
        avg_order2_acc = torch.stack(
            [out["val_order2_acc"] for out in self.validation_outputs]
        ).mean()
        avg_mask_acc = torch.stack(
            [out["val_mask_acc"] for out in self.validation_outputs]
        ).mean()

        self.log("val_epoch_loss", avg_loss, prog_bar=True)
        self.log("val_epoch_order2_acc", avg_order2_acc, prog_bar=True)
        self.log("val_epoch_mask_acc", avg_mask_acc, prog_bar=True)

        # Check if we're training on a single file by looking at unique indices
        all_indices = []
        for output in self.validation_outputs:
            for example_data in output["per_example_data"]:
                all_indices.append(example_data["index"])

        unique_indices = sorted(set(all_indices))

        # If we have many indices (0,1,2,3,4...) from a single file, show per-example metrics
        # If we have just indices 0,1 repeated many times, show aggregated metrics
        single_file_mode = len(unique_indices) > 2 or (
            len(unique_indices) <= 2 and len(all_indices) <= 10
        )

        if single_file_mode:
            # Single file mode: show enhanced metrics for each individual example
            print(f"\nValidation Epoch {self.current_epoch} - Per-Example Performance:")
            print("(Enhanced metrics matching training format)")
            print(
                f"{'Example':<10} {'Valid Pixels':<15} {'Order2 Acc':<15} {'Mask Acc':<15} {'Color Acc':<15} {'Mask Incorrect':<15} {'Color Incorrect':<15}"
            )
            print("-" * 115)

            # Collect metrics for each example in order
            example_metrics = {}
            for output in self.validation_outputs:
                for example_data in output["per_example_data"]:
                    idx = example_data["index"]
                    if idx not in example_metrics:
                        example_metrics[idx] = example_data

            # Display in order
            for idx in sorted(example_metrics.keys()):
                metrics = example_metrics[idx]
                valid_pixels = metrics["valid_pixels"]
                correct_pixels = metrics["correct_pixels"]
                order2_acc = metrics.get("order2_accuracy", 0.0)
                mask_acc = metrics.get("mask_accuracy", 0.0)
                color_acc = metrics.get("color_accuracy", 0.0)
                mask_incorrect = metrics.get("mask_incorrect", 0.0)
                color_incorrect = metrics.get("color_incorrect", 0.0)

                print(
                    f"{idx:<10} {valid_pixels:<15} {order2_acc:<15.2%} {mask_acc:<15.2%} {color_acc:<15.2%} {mask_incorrect:<15.1f} {color_incorrect:<15.1f}"
                )

                # Enhanced wandb logging
                self.log(f"val_example_{idx}_valid_pixels", float(valid_pixels))
                self.log(f"val_example_{idx}_correct_pixels", float(correct_pixels))
                self.log(f"val_example_{idx}_order2_accuracy", float(order2_acc))
                self.log(f"val_example_{idx}_mask_accuracy", float(mask_acc))
                self.log(f"val_example_{idx}_color_accuracy", float(color_acc))
                self.log(f"val_example_{idx}_mask_incorrect", float(mask_incorrect))
                self.log(f"val_example_{idx}_color_incorrect", float(color_incorrect))
        else:
            # Multi-file mode: aggregate enhanced metrics by index
            index_metrics = {}
            for output in self.validation_outputs:
                for example_data in output["per_example_data"]:
                    idx = example_data["index"]
                    if idx not in index_metrics:
                        index_metrics[idx] = {
                            "valid_pixels_sum": example_data["valid_pixels"],
                            "correct_pixels_sum": example_data["correct_pixels"],
                            "order2_acc_sum": example_data.get("order2_accuracy", 0.0),
                            "mask_acc_sum": example_data.get("mask_accuracy", 0.0),
                            "color_acc_sum": example_data.get("color_accuracy", 0.0),
                            "mask_incorrect_sum": example_data.get(
                                "mask_incorrect", 0.0
                            ),
                            "color_incorrect_sum": example_data.get(
                                "color_incorrect", 0.0
                            ),
                            "count": 1,
                        }
                    else:
                        # Same index can appear multiple times in validation set
                        index_metrics[idx]["valid_pixels_sum"] += example_data[
                            "valid_pixels"
                        ]
                        index_metrics[idx]["correct_pixels_sum"] += example_data[
                            "correct_pixels"
                        ]
                        index_metrics[idx]["order2_acc_sum"] += example_data.get(
                            "order2_accuracy", 0.0
                        )
                        index_metrics[idx]["mask_acc_sum"] += example_data.get(
                            "mask_accuracy", 0.0
                        )
                        index_metrics[idx]["color_acc_sum"] += example_data.get(
                            "color_accuracy", 0.0
                        )
                        index_metrics[idx]["mask_incorrect_sum"] += example_data.get(
                            "mask_incorrect", 0.0
                        )
                        index_metrics[idx]["color_incorrect_sum"] += example_data.get(
                            "color_incorrect", 0.0
                        )
                        index_metrics[idx]["count"] += 1

            # Report enhanced per-index metrics in ascending order
            if index_metrics:
                print(
                    f"\nValidation Epoch {self.current_epoch} - Per-Index Performance (averaged):"
                )
                print("(Enhanced metrics matching training format)")
                print(
                    f"{'Index':<10} {'Count':<10} {'Avg Valid Pixels':<18} {'Avg Order2 Acc':<18} {'Avg Mask Acc':<15} {'Avg Color Acc':<15} {'Avg Mask Incorrect':<20} {'Avg Color Incorrect':<20}"
                )
                print("-" * 140)

                for idx in sorted(index_metrics.keys()):
                    metrics = index_metrics[idx]
                    count = metrics["count"]
                    avg_valid_pixels = metrics["valid_pixels_sum"] / count
                    avg_correct_pixels = metrics["correct_pixels_sum"] / count
                    avg_order2_acc = metrics["order2_acc_sum"] / count
                    avg_mask_acc = metrics["mask_acc_sum"] / count
                    avg_color_acc = metrics["color_acc_sum"] / count
                    avg_mask_incorrect = metrics["mask_incorrect_sum"] / count
                    avg_color_incorrect = metrics["color_incorrect_sum"] / count

                    print(
                        f"{idx:<10} {count:<10} {avg_valid_pixels:<18.1f} {avg_order2_acc:<18.2%} {avg_mask_acc:<15.2%} {avg_color_acc:<15.2%} {avg_mask_incorrect:<20.1f} {avg_color_incorrect:<20.1f}"
                    )

                    # Enhanced wandb logging
                    self.log(f"val_idx_{idx}_count", float(count))
                    self.log(f"val_idx_{idx}_avg_valid_pixels", float(avg_valid_pixels))
                    self.log(
                        f"val_idx_{idx}_avg_correct_pixels", float(avg_correct_pixels)
                    )
                    self.log(
                        f"val_idx_{idx}_avg_order2_accuracy", float(avg_order2_acc)
                    )
                    self.log(f"val_idx_{idx}_avg_mask_accuracy", float(avg_mask_acc))
                    self.log(f"val_idx_{idx}_avg_color_accuracy", float(avg_color_acc))
                    self.log(
                        f"val_idx_{idx}_avg_mask_incorrect", float(avg_mask_incorrect)
                    )
                    self.log(
                        f"val_idx_{idx}_avg_color_incorrect", float(avg_color_incorrect)
                    )

        # Edge visualization for validation examples (test subset only)
        if self.validation_examples_for_viz:
            print(
                f"\n\033[1m========== Edge Visualization - Test Subset (Epoch {self.current_epoch}) ==========\033[0m"
            )

            for i, example in enumerate(self.validation_examples_for_viz):
                example_idx = example["example_idx"]
                pred_edges = example["predicted_edges"]
                target_edges = example["target_edges"]
                mask = example["mask"]

                print(
                    f"\n\033[1mTest Example {example_idx} (Validation #{i+1}):\033[0m"
                )

                # Only show non-mask regions for better visualization
                if mask.any():
                    # Find bounding box of non-mask region
                    mask_coords = torch.where(mask)
                    if len(mask_coords[0]) > 0:
                        min_y, max_y = (
                            mask_coords[0].min().item(),
                            mask_coords[0].max().item(),
                        )
                        min_x, max_x = (
                            mask_coords[1].min().item(),
                            mask_coords[1].max().item(),
                        )

                        # Add small padding
                        min_y = max(0, min_y - 1)
                        max_y = min(29, max_y + 1)
                        min_x = max(0, min_x - 1)
                        max_x = min(29, max_x + 1)

                        # Extract regions
                        pred_region = pred_edges[
                            min_y : max_y + 1, min_x : max_x + 1, :
                        ]
                        target_region = target_edges[
                            min_y : max_y + 1, min_x : max_x + 1, :
                        ]

                        # Convert predictions to binary (threshold at 0.5)
                        pred_region_binary = (pred_region > 0.5).float()

                        try:
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

                            # Calculate and print accuracies using color model
                            if (
                                self.current_filename
                                and "predicted_full" in example
                                and "target_full" in example
                            ):
                                self._print_accuracy_metrics(
                                    example["predicted_full"],
                                    example["target_full"],
                                    example["mask"],
                                    example_idx,
                                    example["predicted_mask_logits"],
                                )

                        except Exception as e:
                            print(f"Error visualizing example {example_idx}: {e}")
                else:
                    print(f"  (No non-mask pixels to visualize)")

        # Clear validation outputs
        self.validation_outputs = []
        self.validation_examples_for_viz = []

    def _print_accuracy_metrics(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
        example_idx: int,
        predicted_mask_logits: torch.Tensor,
    ) -> None:
        """Calculate and print color and mask accuracy metrics."""
        try:
            # Load the color model for this filename
            checkpoint_dir = getattr(
                self, "color_model_checkpoint_dir", "optimized_checkpoints"
            )
            color_model = load_color_model_for_inference(
                self.current_filename, checkpoint_dir=checkpoint_dir
            )

            # Predict colors and masks from both predicted and target features
            with torch.no_grad():
                # Get predictions for both features (add batch dimension)
                pred_colors = color_model.predict_colors(
                    predicted_features.unsqueeze(0)
                )[
                    0
                ]  # Remove batch dim
                target_colors = color_model.predict_colors(
                    target_features.unsqueeze(0)
                )[
                    0
                ]  # Remove batch dim

                # Convert the passed-in logits to a binary mask
                pred_mask = (predicted_mask_logits.squeeze(-1) > 0).float()

                # Convert target_mask to same device and type as predictions
                target_mask = target_mask.to(pred_colors.device)

                # 1. Calculate mask accuracy (predicted mask vs target mask)
                mask_correct = (pred_mask == target_mask).sum().item()
                mask_total = target_mask.numel()
                mask_accuracy = mask_correct / mask_total if mask_total > 0 else 0.0

                # 2. Calculate color accuracy within the PREDICTED mask region only
                pred_mask_area = pred_mask.sum().item()
                if pred_mask_area > 0:
                    # Only consider pixels where predicted mask is True
                    color_correct_in_pred_mask = (
                        ((pred_colors == target_colors) & pred_mask).sum().item()
                    )
                    color_accuracy_in_pred_mask = (
                        color_correct_in_pred_mask / pred_mask_area
                    )
                else:
                    color_accuracy_in_pred_mask = 0.0
                    color_correct_in_pred_mask = 0

                # 3. Calculate color accuracy within the TARGET mask region (for comparison)
                target_mask_area = target_mask.sum().item()
                if target_mask_area > 0:
                    color_correct_in_target_mask = (
                        ((pred_colors == target_colors) & target_mask).sum().item()
                    )
                    color_accuracy_in_target_mask = (
                        color_correct_in_target_mask / target_mask_area
                    )
                else:
                    color_accuracy_in_target_mask = 0.0
                    color_correct_in_target_mask = 0

                # Print the metrics
                print(f"\n📊 Accuracy Metrics for Example {example_idx}:")
                print(
                    f"  🎯 Mask Accuracy: {mask_accuracy:.1%} ({mask_correct}/{mask_total} pixels)"
                )
                print(
                    f"  🎨 Color Accuracy (in predicted mask): {color_accuracy_in_pred_mask:.1%} ({color_correct_in_pred_mask}/{pred_mask_area} pixels)"
                )
                print(
                    f"  🎨 Color Accuracy (in target mask): {color_accuracy_in_target_mask:.1%} ({color_correct_in_target_mask}/{target_mask_area} pixels)"
                )

                # Additional info about mask differences
                if pred_mask_area != target_mask_area:
                    diff = pred_mask_area - target_mask_area
                    print(
                        f"  ℹ️  Predicted mask size differs by {diff:+d} pixels from target"
                    )

        except Exception as e:
            print(f"  ❌ Could not calculate accuracy metrics: {e}")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate schedule
        def lr_lambda(epoch: int) -> float:
            if epoch < 5:
                return (epoch + 1) / 5  # Warmup
            elif epoch < 50:
                return 1.0
            elif epoch < 100:
                return 0.5
            else:
                return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class SpatialMessagePassing(nn.Module):
    """Efficient spatial message passing for local consistency."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple 3x3 convolution for message passing
        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim // 16,  # Group convolution for efficiency
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, 30, 30, hidden_dim]
        # Convert to conv format
        h_conv = h.permute(0, 3, 1, 2)  # [B, hidden_dim, 30, 30]

        # Message passing
        messages = self.conv(h_conv)

        messages = messages.permute(0, 2, 3, 1)  # [B, 30, 30, hidden_dim]

        # Apply layer norm and activation
        messages = self.norm(messages)
        messages = self.activation(messages)
        messages = self.dropout(messages)

        return messages


def load_feature_data(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    filename_filter: Optional[str] = None,
    data_augment_factor: int = 1,
    use_train_subset: Optional[bool] = None,
) -> DataLoader:
    """Load feature mapping data from NPZ file with optional filename and subset filtering.

    Args:
        npz_path: Path to NPZ file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        filename_filter: Filter to specific filename (without .json extension)
        data_augment_factor: Number of times to repeat the dataset per epoch
        use_train_subset: If True, use only "train" subset; if False, use only "test" subset; if None, use all
    """
    data = np.load(npz_path)

    # Load the features and masks
    inputs_features = torch.from_numpy(data["inputs_features"]).float()
    outputs_features = torch.from_numpy(data["outputs_features"]).float()
    inputs_mask = torch.from_numpy(data["inputs_mask"]).bool()
    outputs_mask = torch.from_numpy(data["outputs_mask"]).bool()
    indices = torch.from_numpy(data["indices"]).long()

    # Load subset information if available
    subset_is_train = None
    subset_is_train_tensor = None
    if "subset_example_index_is_train" in data:
        subset_is_train = data["subset_example_index_is_train"]
        subset_is_train_tensor = torch.from_numpy(subset_is_train).bool()

    print(f"Loaded data from {npz_path}")

    # Apply filename filter if specified
    if filename_filter is not None:
        # Load filenames from the npz file
        filenames = data["filenames"]
        # Create mask for matching filenames (with .json extension)
        filename_with_ext = f"{filename_filter}.json"
        mask = filenames == filename_with_ext
        filter_indices = np.where(mask)[0]

        if len(filter_indices) == 0:
            print(f"  Warning: No examples found for filename: {filename_filter}")
            # Return empty dataloader
            empty_dataset = TensorDataset(
                torch.empty(0, 30, 30, 147),
                torch.empty(0, 30, 30, 147),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.bool),  # subset_is_train
            )
            return DataLoader(empty_dataset, batch_size=batch_size)

        # Filter all arrays
        inputs_features = inputs_features[filter_indices]
        outputs_features = outputs_features[filter_indices]
        inputs_mask = inputs_mask[filter_indices]
        outputs_mask = outputs_mask[filter_indices]
        indices = indices[filter_indices]
        if subset_is_train is not None:
            subset_is_train = subset_is_train[filter_indices]
            subset_is_train_tensor = subset_is_train_tensor[filter_indices]

        print(f"  Filtered to filename: {filename_filter}")
        print(f"  Found {len(filter_indices)} examples")

    # Apply subset filtering if requested
    if use_train_subset is not None and subset_is_train is not None:
        # Create mask for the desired subset
        if use_train_subset:
            subset_mask = subset_is_train
            subset_name = "train"
        else:
            subset_mask = ~subset_is_train
            subset_name = "test"

        # Find indices that match the subset
        subset_indices = np.where(subset_mask)[0]

        if len(subset_indices) == 0:
            print(f"  Warning: No examples found in {subset_name} subset")
            # Return empty dataloader
            empty_dataset = TensorDataset(
                torch.empty(0, 30, 30, 147),
                torch.empty(0, 30, 30, 147),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.bool),  # subset_is_train
            )
            return DataLoader(empty_dataset, batch_size=batch_size)

        # Filter to subset
        inputs_features = inputs_features[subset_indices]
        outputs_features = outputs_features[subset_indices]
        inputs_mask = inputs_mask[subset_indices]
        outputs_mask = outputs_mask[subset_indices]
        indices = indices[subset_indices]
        if subset_is_train_tensor is not None:
            subset_is_train_tensor = subset_is_train_tensor[subset_indices]

        print(f"  Filtered to {subset_name} subset: {len(subset_indices)} examples")

    # Apply data augmentation by repeating the dataset
    if data_augment_factor > 1:
        inputs_features = inputs_features.repeat(data_augment_factor, 1, 1, 1)
        outputs_features = outputs_features.repeat(data_augment_factor, 1, 1, 1)
        inputs_mask = inputs_mask.repeat(data_augment_factor, 1, 1)
        outputs_mask = outputs_mask.repeat(data_augment_factor, 1, 1)
        indices = indices.repeat(data_augment_factor)
        if subset_is_train_tensor is not None:
            subset_is_train_tensor = subset_is_train_tensor.repeat(data_augment_factor)
        print(
            f"  Data augmented {data_augment_factor}x: {len(inputs_features)} total examples"
        )

    print(f"  Input features shape: {inputs_features.shape}")
    print(f"  Output features shape: {outputs_features.shape}")
    print(f"  Examples: {len(inputs_features)}")

    # Create dataset
    if subset_is_train_tensor is not None:
        dataset = TensorDataset(
            inputs_features,
            outputs_features,
            inputs_mask,
            outputs_mask,
            indices,
            subset_is_train_tensor,
        )
    else:
        # Create a dummy tensor for consistency
        dummy_subset = torch.zeros(len(inputs_features), dtype=torch.bool)
        dataset = TensorDataset(
            inputs_features,
            outputs_features,
            inputs_mask,
            outputs_mask,
            indices,
            dummy_subset,
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


def run_visualization(args):
    """
    Runs a visualization of the message passing process.
    """
    print("--- Running Message Passing Visualization ---")
    if not args.filename or args.filename == "all":
        raise ValueError("--filename must be specified for visualization.")
    if not args.load_checkpoint:
        raise ValueError("--load_checkpoint must be specified for visualization.")

    # Load the model from the checkpoint
    model = FeatureMappingModel.load_from_checkpoint(args.load_checkpoint)
    model.eval()
    print(f"Loaded model from {args.load_checkpoint}")

    # Load the specific test example
    val_loader = load_feature_data(
        os.path.join(args.data_dir, "train_all.npz"),
        batch_size=1,
        shuffle=False,
        filename_filter=args.filename,
        use_train_subset=False,  # Use the test set
    )

    if len(val_loader) == 0:
        print(f"No test data found for filename {args.filename}. Trying eval_all.npz.")
        val_loader = load_feature_data(
            os.path.join(args.data_dir, "eval_all.npz"),
            batch_size=1,
            shuffle=False,
            filename_filter=args.filename,
        )

    if len(val_loader) == 0:
        raise ValueError(
            f"No data found for filename {args.filename} in train or eval splits."
        )

    batch = next(iter(val_loader))
    input_features, target_features, _, target_mask, _, _ = batch

    # Move data to the same device as the model
    input_features = input_features.to(model.device)
    target_features = target_features.to(model.device)
    target_mask = target_mask.to(model.device)

    # Load the color model for visualization
    try:
        color_model = load_color_model_for_inference(
            args.filename, checkpoint_dir=args.color_model_checkpoint_dir
        )
        color_model.to(model.device)
    except FileNotFoundError:
        print(
            f"Warning: Color model for {args.filename} not found. Visualization will be feature-based."
        )
        color_model = None

    pred_color_frames = []
    pred_mask_frames = []
    pred_order2_frames = []

    # Get target data once
    target_order2 = target_features[:, :, :, :36].cpu().numpy()
    target_mask_np = target_mask.cpu().numpy()

    if color_model:
        with torch.no_grad():
            target_colors = color_model.predict_colors(target_features)[0].cpu().numpy()
    else:
        target_colors = target_features[0, :, :, 0].cpu().numpy()

    for i in range(65):
        print(f"Generating frame for {i} message passing rounds...")
        model.num_message_rounds = i
        with torch.no_grad():
            pred_features, pred_mask_logits = model(input_features)

        # Store predicted order2 features
        pred_order2 = pred_features[:, :, :, :36]
        pred_order2_frames.append(pred_order2.squeeze(0).cpu().numpy())

        # Store predicted mask
        pred_mask = (pred_mask_logits.squeeze(-1) > 0).float()
        pred_mask_frames.append(pred_mask.squeeze(0).cpu().numpy())

        if color_model:
            with torch.no_grad():
                pred_colors = color_model.predict_colors(pred_features)[0]
                pred_color_frames.append(pred_colors.cpu().numpy())
        else:
            # Fallback to a simple representation if no color model
            pred_color_frames.append(pred_features[0, :, :, 0].cpu().numpy())

    # Save all the collected data to an NPZ file
    output_filename = f"{args.filename}_visualization_details.npz"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        output_path,
        predicted_colors=np.stack(pred_color_frames),
        predicted_masks=np.stack(pred_mask_frames),
        predicted_order2_features=np.stack(pred_order2_frames),
        target_colors=target_colors,
        target_mask=target_mask_np,
        target_order2_features=target_order2,
    )
    print(f"Saved detailed visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature mapping with message passing")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed_data",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="main_mapping_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="Maximum number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num_message_rounds",
        type=int,
        default=12,
        help="Number of message passing rounds",
    )
    parser.add_argument(
        "--order2_weight",
        type=float,
        default=10.0,
        help="Weight for order2 features in loss",
    )
    parser.add_argument(
        "--mask_weight", type=float, default=1.0, help="Weight for mask prediction loss"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="arc-main-mapping",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="28e73c20",
        help="Filter to examples from a specific filename (without .json extension). Use 'all' to train on all files.",
    )
    parser.add_argument(
        "--data_augment_factor",
        type=int,
        default=1,
        help="Number of times to repeat the dataset per epoch (useful for small datasets)",
    )
    parser.add_argument(
        "--single_file_mode",
        action="store_true",
        help="Train on 'train' subset and evaluate on 'test' subset from the same file",
    )
    parser.add_argument(
        "--color_model_checkpoint_dir",
        type=str,
        default="optimized_checkpoints",
        help="Directory where color model checkpoints are saved (ex17 models, default: optimized_checkpoints)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run message passing visualization instead of training. Requires --filename and --load_checkpoint.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint to load for visualization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization_outputs",
        help="Directory to save visualization NPZ files.",
    )

    args = parser.parse_args()

    # Handle filename filtering
    filename_filter = None if args.filename == "all" else args.filename

    # If visualization is requested, run it and exit
    if args.visualize:
        run_visualization(args)
        return

    if args.single_file_mode:
        # Single file mode: use training data file for both train and eval
        # Train on "train" subset, evaluate on "test" subset
        if filename_filter is None:
            raise ValueError(
                "--filename must be specified when using --single_file_mode"
            )

        print(f"\n=== Single File Mode ===")
        print(f"Training on 'train' subset of {filename_filter}")
        print(f"Evaluating on 'test' subset of {filename_filter}")

        # Both loaders use the training data file
        train_path = os.path.join(args.data_dir, "train_all.npz")
        val_path = train_path  # Same file

        # Load training data (train subset only)
        train_loader = load_feature_data(
            train_path,
            batch_size=args.batch_size,
            shuffle=True,
            filename_filter=filename_filter,
            data_augment_factor=args.data_augment_factor,
            use_train_subset=True,  # Only train subset
        )

        # Load validation data (test subset only)
        val_loader = load_feature_data(
            val_path,
            batch_size=args.batch_size,
            shuffle=False,
            filename_filter=filename_filter,
            data_augment_factor=1,  # Don't augment validation data
            use_train_subset=False,  # Only test subset
        )
    else:
        # Standard mode: use separate train and eval files
        train_path = os.path.join(args.data_dir, "train_all.npz")
        val_path = os.path.join(args.data_dir, "eval_all.npz")

        train_loader = load_feature_data(
            train_path,
            batch_size=args.batch_size,
            shuffle=True,
            filename_filter=filename_filter,
            data_augment_factor=args.data_augment_factor,
            use_train_subset=None,  # Use all examples
        )
        val_loader = load_feature_data(
            val_path,
            batch_size=args.batch_size,
            shuffle=False,
            filename_filter=filename_filter,
            data_augment_factor=1,  # Don't augment validation data
            use_train_subset=None,  # Use all examples
        )

    # Get feature dimensions from first batch
    first_batch = next(iter(train_loader))
    input_feature_dim = first_batch[0].shape[-1]
    output_feature_dim = first_batch[1].shape[-1]

    print(f"\nFeature dimensions:")
    print(f"  Input: {input_feature_dim}")
    print(f"  Output: {output_feature_dim}")

    # Create model
    model = FeatureMappingModel(
        input_feature_dim=input_feature_dim,
        output_feature_dim=output_feature_dim,
        hidden_dim=args.hidden_dim,
        num_message_rounds=args.num_message_rounds,
        lr=args.lr,
        order2_weight=args.order2_weight,
        mask_weight=args.mask_weight,
    )

    # Set filename for visualization if filtering to a specific file
    if filename_filter:
        model.set_filename(filename_filter)
        print(f"Set model filename to: {filename_filter} for visualization")

    # Set color model checkpoint directory
    model.set_color_model_checkpoint_dir(args.color_model_checkpoint_dir)
    print(f"Color model checkpoint directory: {args.color_model_checkpoint_dir}")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Create wandb logger if enabled
    logger = None
    if not args.no_wandb:
        wandb_config = {
            "architecture": "MessagePassingFeatureMapper",
            "dataset": "ARC-AGI",
            "filename": args.filename,
            "single_file_mode": args.single_file_mode,
            "hidden_dim": args.hidden_dim,
            "num_message_rounds": args.num_message_rounds,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "order2_weight": args.order2_weight,
            "mask_weight": args.mask_weight,
            "input_feature_dim": input_feature_dim,
            "output_feature_dim": output_feature_dim,
        }

        # Configure wandb for immediate logging
        os.environ["WANDB_MODE"] = "online"  # Ensure online mode

        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config=wandb_config,
            log_model=True,
            save_dir="./wandb",
        )

        # Set wandb to log frequently
        if wandb.run:
            wandb.run.settings.update({"_log_every_n_steps": 1})

    # Create trainer with callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_order2_acc",
        dirpath=os.path.join(args.checkpoint_dir, "checkpoints"),
        filename="feature_mapping-{epoch:02d}-{val_epoch_loss:.4f}-{val_epoch_order2_acc:.4f}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=args.checkpoint_dir,
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=1,  # Log every single step
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
