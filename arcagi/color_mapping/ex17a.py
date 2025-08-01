import argparse
import os

# Import from parent module
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
from utils.terminal_imshow import imshow


class OptimizedMemorizationModel(pl.LightningModule):
    """
    Optimized model for perfect memorization with minimal parameters.

    This version (ex17) predicts both colors and masks with separate accuracies and losses.

    Key optimizations:
    1. Example-aware feature transformation
    2. Efficient spatial message passing
    3. Adaptive learning rate with strong warmup
    4. Focused architecture for 2 training examples
    5. Batch normalization for faster convergence
    6. Better weight initialization
    7. Sinusoidal position embeddings (parameter-free)
    8. Dual prediction heads for colors and masks
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 512,
        num_message_rounds: int = 6,  # OPTIMIZED: Reduced from 6 to 3 rounds
        num_classes: int = 10,
        dropout: float = 0.0,
        lr: float = 0.1,  # Even higher learning rate
        weight_decay: float = 0.0,
        temperature: float = 0.01,  # Lower temperature for sharper predictions
        filename: str = "3345333e",
        num_train_examples: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_message_rounds = num_message_rounds
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.filename = filename
        self.num_train_examples = num_train_examples

        # Track validation metrics across entire epoch
        self.validation_outputs = []

        # Track per-index metrics for validation
        self.index_metrics = {}

        # Track per-index metrics for training
        self.training_index_metrics = {}

        # Feature extraction with layer normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Binary noise layer for training-time regularization
        self.noise_prob = 0.05  # 5% probability of flipping each feature

        # Sinusoidal position embeddings - no parameters needed
        # Pre-compute the position embeddings for efficiency
        self.register_buffer(
            "pos_embed", self._create_sinusoidal_embeddings(hidden_dim)
        )

        # Efficient message passing with weight tying
        # Use a single shared layer that gets called multiple times
        # self.shared_message_layer = UnetSpatialMessagePassing(
        #     hidden_dim, dropout=dropout
        # )
        self.shared_message_layer = SpatialMessagePassing(hidden_dim, dropout=dropout)
        self.shared_linear = nn.Linear(hidden_dim, hidden_dim)
        self.shared_dropout = nn.Dropout(dropout)

        # Color-specific prediction heads for better specialization
        self.color_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_classes)]
        )

        # Mask prediction head
        self.mask_head = nn.Linear(hidden_dim, 1)

        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_transform = nn.Linear(hidden_dim, hidden_dim)

        # Store available colors
        self.available_colors: Optional[List[int]] = None

        # Better initialization
        self._init_weights()

    def apply_binary_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Apply binary noise to features during training only."""
        if self.training and self.noise_prob > 0:
            # Create noise mask with probability noise_prob
            noise_mask = torch.rand_like(features) < self.noise_prob

            # For binary features (0/1), flip them
            # For continuous features, add small random perturbation
            binary_features = (features == 0) | (features == 1)

            # Flip binary features
            features = torch.where(
                noise_mask & binary_features, 1 - features, features  # Flip 0->1, 1->0
            )

            # Add small noise to continuous features
            continuous_noise = torch.randn_like(features) * 0.01
            features = torch.where(
                noise_mask & ~binary_features, features + continuous_noise, features
            )

        return features

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

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, features: torch.Tensor, example_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict both colors and masks.

        Args:
            features: Input features of shape [B, 30, 30, feature_dim]
            example_idx: Optional example indices for example-aware processing

        Returns:
            color_logits: Color predictions of shape [B, 30, 30, num_classes]
            mask_logits: Mask predictions of shape [B, 30, 30, 1]
        """
        # Apply binary noise during training
        features = self.apply_binary_noise(features)

        # Extract base features
        h = self.feature_extractor(features)  # [B, 30, 30, hidden_dim]

        # Add position embeddings
        h = h + self.pos_embed.unsqueeze(0)

        # Spatial message passing with stronger residuals
        # Use the same shared layer multiple times
        for i in range(self.num_message_rounds):
            h_new = self.shared_message_layer(h)
            # h2 = self.shared_dropout(self.shared_linear(h_new))
            # Gradually decrease residual strength
            alpha = 1.0 - (i / self.num_message_rounds)
            h = h + alpha * h_new

        # Global context
        h_spatial = h.permute(0, 3, 1, 2).contiguous()  # [B, hidden_dim, 30, 30]
        global_context = (
            self.global_pool(h_spatial).squeeze(-1).squeeze(-1)
        )  # [B, hidden_dim]
        global_context = self.global_transform(global_context)  # [B, hidden_dim]

        # Add global context back
        h = h + global_context.unsqueeze(1).unsqueeze(2) * 0.1

        # Color-specific predictions
        logits_list: List[torch.Tensor] = []
        for color_head in self.color_heads:
            logits_list.append(color_head(h))  # [B, 30, 30, 1]

        color_logits = torch.cat(logits_list, dim=-1)  # [B, 30, 30, num_classes]

        # Temperature scaling for colors
        color_logits = color_logits / self.temperature

        # Apply color constraints
        color_logits = self.apply_color_constraints(color_logits)

        # Mask prediction
        mask_logits = self.mask_head(h)  # [B, 30, 30, 1]

        return color_logits, mask_logits

    def apply_color_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hard constraints to ensure only available colors are predicted."""
        if self.available_colors is not None:
            mask = torch.ones(self.num_classes, device=logits.device) * -1e10
            mask[self.available_colors] = 0
            logits = logits + mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        return logits

    def predict_colors(self, features: torch.Tensor) -> torch.Tensor:
        """Predict colors from features. Returns color indices."""
        with torch.no_grad():
            color_logits, _ = self(features)
            return color_logits.argmax(dim=-1)

    def predict_masks(self, features: torch.Tensor) -> torch.Tensor:
        """Predict masks from features. Returns binary mask."""
        with torch.no_grad():
            _, mask_logits = self(features)
            return (mask_logits > 0).squeeze(-1)

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        (
            input_features,
            output_features,
            input_colors,
            output_colors,
            input_masks,
            output_masks,
            _,
            _,
        ) = batch

        # Get example indices for this batch
        example_indices = (
            torch.arange(len(input_features), device=self.device)
            % self.num_train_examples
        )

        # Process inputs with example information
        input_color_logits, input_mask_logits = self(input_features, example_indices)

        # Color loss for inputs
        input_color_loss = F.cross_entropy(
            input_color_logits.reshape(-1, input_color_logits.size(-1)),
            input_colors.reshape(-1),
        )

        # Mask loss for inputs (binary cross entropy)
        input_mask_loss = F.binary_cross_entropy_with_logits(
            input_mask_logits.reshape(-1), input_masks.reshape(-1).float()
        )

        # Process outputs with example information
        output_color_logits, output_mask_logits = self(output_features, example_indices)

        # Color loss for outputs
        output_color_loss = F.cross_entropy(
            output_color_logits.reshape(-1, output_color_logits.size(-1)),
            output_colors.reshape(-1),
        )

        # Mask loss for outputs (binary cross entropy)
        output_mask_loss = F.binary_cross_entropy_with_logits(
            output_mask_logits.reshape(-1), output_masks.reshape(-1).float()
        )

        # Combined loss
        total_loss = (
            input_color_loss + output_color_loss + input_mask_loss + output_mask_loss
        )

        # Calculate color accuracy
        all_color_logits = torch.cat(
            [input_color_logits.flatten(0, 2), output_color_logits.flatten(0, 2)]
        )
        all_color_targets = torch.cat([input_colors.flatten(), output_colors.flatten()])
        color_predictions = all_color_logits.argmax(dim=-1)

        valid_color_mask = all_color_targets != -1
        if valid_color_mask.any():
            color_accuracy = (
                (
                    color_predictions[valid_color_mask]
                    == all_color_targets[valid_color_mask]
                )
                .float()
                .mean()
            )
        else:
            color_accuracy = torch.tensor(0.0)

        # Calculate mask accuracy
        all_mask_logits = torch.cat(
            [input_mask_logits.flatten(), output_mask_logits.flatten()]
        )
        all_mask_targets = torch.cat([input_masks.flatten(), output_masks.flatten()])
        mask_predictions = (all_mask_logits > 0).float()

        mask_accuracy = (mask_predictions == all_mask_targets.float()).float().mean()

        self.log("train_loss", total_loss)
        self.log("train_color_acc", color_accuracy)
        self.log("train_mask_acc", mask_accuracy)

        # Track per-index training metrics
        for i in range(len(input_features)):
            idx = example_indices[i].item()

            # Calculate input color accuracy for this example
            input_color_valid_mask_i = input_colors[i] != -1
            input_color_correct_pixels = 0
            input_color_valid_pixels = input_color_valid_mask_i.sum().item()
            if input_color_valid_pixels > 0:
                input_color_predictions_i = input_color_logits[i].argmax(dim=-1)
                input_color_correct_pixels = (
                    (
                        input_color_predictions_i[input_color_valid_mask_i]
                        == input_colors[i][input_color_valid_mask_i]
                    )
                    .sum()
                    .item()
                )

            # Calculate output color accuracy for this example
            output_color_valid_mask_i = output_colors[i] != -1
            output_color_correct_pixels = 0
            output_color_valid_pixels = output_color_valid_mask_i.sum().item()
            if output_color_valid_pixels > 0:
                output_color_predictions_i = output_color_logits[i].argmax(dim=-1)
                output_color_correct_pixels = (
                    (
                        output_color_predictions_i[output_color_valid_mask_i]
                        == output_colors[i][output_color_valid_mask_i]
                    )
                    .sum()
                    .item()
                )

            # Calculate mask accuracies for this example
            input_mask_predictions_i = (input_mask_logits[i] > 0).squeeze(-1)
            output_mask_predictions_i = (output_mask_logits[i] > 0).squeeze(-1)
            input_mask_correct_pixels = (
                (input_mask_predictions_i == input_masks[i]).sum().item()
            )
            output_mask_correct_pixels = (
                (output_mask_predictions_i == output_masks[i]).sum().item()
            )

            # Store metrics for this index
            if idx not in self.training_index_metrics:
                self.training_index_metrics[idx] = {
                    "input_color_correct": 0,
                    "input_color_total": 0,
                    "output_color_correct": 0,
                    "output_color_total": 0,
                    "input_mask_correct": 0,
                    "input_mask_total": 0,
                    "output_mask_correct": 0,
                    "output_mask_total": 0,
                    "count": 0,
                }

            self.training_index_metrics[idx][
                "input_color_correct"
            ] += input_color_correct_pixels
            self.training_index_metrics[idx][
                "input_color_total"
            ] += input_color_valid_pixels
            self.training_index_metrics[idx][
                "output_color_correct"
            ] += output_color_correct_pixels
            self.training_index_metrics[idx][
                "output_color_total"
            ] += output_color_valid_pixels
            self.training_index_metrics[idx][
                "input_mask_correct"
            ] += input_mask_correct_pixels
            self.training_index_metrics[idx]["input_mask_total"] += 900  # 30x30
            self.training_index_metrics[idx][
                "output_mask_correct"
            ] += output_mask_correct_pixels
            self.training_index_metrics[idx]["output_mask_total"] += 900  # 30x30
            self.training_index_metrics[idx]["count"] += 1

        # Visualize every 10 epochs
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            # self.visualize_predictions(
            #     input_features[0:1], input_colors[0:1], input_color_logits[0:1], "input"
            # )
            for i in range(len(output_features)):
                self.visualize_predictions(
                    output_features[i : i + 1],
                    output_colors[i : i + 1],
                    output_color_logits[i : i + 1],
                    "output",
                )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Print training per-index accuracy at the end of each epoch."""
        if self.training_index_metrics:
            print(f"\nTraining Per-Index Accuracy (Epoch {self.current_epoch}):")
            print(
                f"{'Index':<8} {'Input Color':<12} {'Output Color':<12} {'Input Mask':<12} {'Output Mask':<12} {'All Perfect':<12}"
            )
            print("-" * 70)

            for idx in sorted(self.training_index_metrics.keys()):
                metrics = self.training_index_metrics[idx]

                input_color_acc = 0.0
                if metrics["input_color_total"] > 0:
                    input_color_acc = (
                        metrics["input_color_correct"] / metrics["input_color_total"]
                    )

                output_color_acc = 0.0
                if metrics["output_color_total"] > 0:
                    output_color_acc = (
                        metrics["output_color_correct"] / metrics["output_color_total"]
                    )

                input_mask_acc = 0.0
                if metrics["input_mask_total"] > 0:
                    input_mask_acc = (
                        metrics["input_mask_correct"] / metrics["input_mask_total"]
                    )

                output_mask_acc = 0.0
                if metrics["output_mask_total"] > 0:
                    output_mask_acc = (
                        metrics["output_mask_correct"] / metrics["output_mask_total"]
                    )

                all_perfect = (
                    (input_color_acc >= 1.0)
                    and (output_color_acc >= 1.0)
                    and (input_mask_acc >= 1.0)
                    and (output_mask_acc >= 1.0)
                )

                print(
                    f"{idx:<8} {input_color_acc:<12.2%} {output_color_acc:<12.2%} {input_mask_acc:<12.2%} {output_mask_acc:<12.2%} {'✓' if all_perfect else '✗':<12}"
                )

        # Clear training metrics for next epoch
        self.training_index_metrics = {}

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        (
            input_features,
            output_features,
            input_colors,
            output_colors,
            input_masks,
            output_masks,
            example_indices,
            _,
        ) = batch

        # For validation, we don't use example indices
        input_color_logits, input_mask_logits = self(input_features)
        output_color_logits, output_mask_logits = self(output_features)

        # Compute color losses
        input_color_loss = F.cross_entropy(
            input_color_logits.reshape(-1, input_color_logits.size(-1)),
            input_colors.reshape(-1),
        )
        output_color_loss = F.cross_entropy(
            output_color_logits.reshape(-1, output_color_logits.size(-1)),
            output_colors.reshape(-1),
        )

        # Compute mask losses
        input_mask_loss = F.binary_cross_entropy_with_logits(
            input_mask_logits.reshape(-1), input_masks.reshape(-1).float()
        )
        output_mask_loss = F.binary_cross_entropy_with_logits(
            output_mask_logits.reshape(-1), output_masks.reshape(-1).float()
        )

        total_loss = (
            input_color_loss + output_color_loss + input_mask_loss + output_mask_loss
        )

        # Compute color accuracy
        input_color_predictions = input_color_logits.argmax(dim=-1)  # [B, 30, 30]
        output_color_predictions = output_color_logits.argmax(dim=-1)  # [B, 30, 30]

        # Create separate masks for input and output colors
        input_color_valid_mask = input_colors != -1
        output_color_valid_mask = output_colors != -1

        # Calculate color accuracies
        input_color_accuracy = torch.tensor(0.0)
        output_color_accuracy = torch.tensor(0.0)

        if input_color_valid_mask.any():
            input_color_accuracy = (
                (
                    input_color_predictions[input_color_valid_mask]
                    == input_colors[input_color_valid_mask]
                )
                .float()
                .mean()
            )

        if output_color_valid_mask.any():
            output_color_accuracy = (
                (
                    output_color_predictions[output_color_valid_mask]
                    == output_colors[output_color_valid_mask]
                )
                .float()
                .mean()
            )

        # Combined color accuracy for logging
        all_color_predictions = torch.cat(
            [input_color_predictions.flatten(), output_color_predictions.flatten()]
        )
        all_color_targets = torch.cat([input_colors.flatten(), output_colors.flatten()])
        valid_color_mask = all_color_targets != -1

        if valid_color_mask.any():
            color_accuracy = (
                (
                    all_color_predictions[valid_color_mask]
                    == all_color_targets[valid_color_mask]
                )
                .float()
                .mean()
            )
        else:
            color_accuracy = torch.tensor(0.0)

        # Compute mask accuracy
        input_mask_predictions = (input_mask_logits > 0).squeeze(-1)  # [B, 30, 30]
        output_mask_predictions = (output_mask_logits > 0).squeeze(-1)  # [B, 30, 30]

        # Calculate mask accuracies
        input_mask_accuracy = (input_mask_predictions == input_masks).float().mean()
        output_mask_accuracy = (output_mask_predictions == output_masks).float().mean()

        # Combined mask accuracy
        all_mask_predictions = torch.cat(
            [input_mask_predictions.flatten(), output_mask_predictions.flatten()]
        )
        all_mask_targets = torch.cat([input_masks.flatten(), output_masks.flatten()])
        mask_accuracy = (all_mask_predictions == all_mask_targets).float().mean()

        self.log("val_loss", total_loss)
        self.log("val_color_acc", color_accuracy)
        self.log("val_mask_acc", mask_accuracy)

        # Calculate per-index metrics
        for i in range(len(example_indices)):
            idx = example_indices[i].item()

            # Calculate input color accuracy for this example
            input_color_valid_pixels = input_color_valid_mask[i].sum().item()
            input_color_correct_pixels = 0
            if input_color_valid_pixels > 0:
                input_color_correct_pixels = (
                    (
                        input_color_predictions[i][input_color_valid_mask[i]]
                        == input_colors[i][input_color_valid_mask[i]]
                    )
                    .sum()
                    .item()
                )

            # Calculate output color accuracy for this example
            output_color_valid_pixels = output_color_valid_mask[i].sum().item()
            output_color_correct_pixels = 0
            if output_color_valid_pixels > 0:
                output_color_correct_pixels = (
                    (
                        output_color_predictions[i][output_color_valid_mask[i]]
                        == output_colors[i][output_color_valid_mask[i]]
                    )
                    .sum()
                    .item()
                )

            # Calculate mask accuracies for this example
            input_mask_correct_pixels = (
                (input_mask_predictions[i] == input_masks[i]).sum().item()
            )
            output_mask_correct_pixels = (
                (output_mask_predictions[i] == output_masks[i]).sum().item()
            )

            # Store metrics for this index
            if idx not in self.index_metrics:
                self.index_metrics[idx] = {
                    "input_color_correct": 0,
                    "input_color_total": 0,
                    "output_color_correct": 0,
                    "output_color_total": 0,
                    "input_mask_correct": 0,
                    "input_mask_total": 0,
                    "output_mask_correct": 0,
                    "output_mask_total": 0,
                    "count": 0,
                }

            self.index_metrics[idx]["input_color_correct"] += input_color_correct_pixels
            self.index_metrics[idx]["input_color_total"] += input_color_valid_pixels
            self.index_metrics[idx][
                "output_color_correct"
            ] += output_color_correct_pixels
            self.index_metrics[idx]["output_color_total"] += output_color_valid_pixels
            self.index_metrics[idx]["input_mask_correct"] += input_mask_correct_pixels
            self.index_metrics[idx]["input_mask_total"] += 900  # 30x30
            self.index_metrics[idx]["output_mask_correct"] += output_mask_correct_pixels
            self.index_metrics[idx]["output_mask_total"] += 900  # 30x30
            self.index_metrics[idx]["count"] += 1

        # Store outputs for epoch-level metrics
        self.validation_outputs.append(
            {
                "val_loss": total_loss,
                "val_color_acc": color_accuracy,
                "val_mask_acc": mask_accuracy,
                "input_color_predictions": input_color_predictions[
                    input_color_valid_mask
                ],
                "input_color_targets": input_colors[input_color_valid_mask],
                "output_color_predictions": output_color_predictions[
                    output_color_valid_mask
                ],
                "output_color_targets": output_colors[output_color_valid_mask],
                "input_mask_predictions": input_mask_predictions.flatten(),
                "input_mask_targets": input_masks.flatten(),
                "output_mask_predictions": output_mask_predictions.flatten(),
                "output_mask_targets": output_masks.flatten(),
                # Store full grids for visualization
                "full_output_color_predictions": output_color_predictions,
                "full_output_color_targets": output_colors,
                "batch_size": input_color_valid_mask.sum()
                + output_color_valid_mask.sum(),
            }
        )

        return {
            "val_loss": total_loss,
            "val_color_acc": color_accuracy,
            "val_mask_acc": mask_accuracy,
        }

    def on_validation_epoch_start(self) -> None:
        """Reset validation outputs at the start of each epoch."""
        self.validation_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Calculate epoch-level metrics from all validation batches."""
        if not self.validation_outputs:
            return

        # Gather INPUT color predictions and targets
        input_color_predictions = torch.cat(
            [out["input_color_predictions"] for out in self.validation_outputs]
        )
        input_color_targets = torch.cat(
            [out["input_color_targets"] for out in self.validation_outputs]
        )

        # Gather OUTPUT color predictions and targets
        output_color_predictions = torch.cat(
            [out["output_color_predictions"] for out in self.validation_outputs]
        )
        output_color_targets = torch.cat(
            [out["output_color_targets"] for out in self.validation_outputs]
        )

        # Gather mask predictions and targets
        input_mask_predictions = torch.cat(
            [out["input_mask_predictions"] for out in self.validation_outputs]
        )
        input_mask_targets = torch.cat(
            [out["input_mask_targets"] for out in self.validation_outputs]
        )
        output_mask_predictions = torch.cat(
            [out["output_mask_predictions"] for out in self.validation_outputs]
        )
        output_mask_targets = torch.cat(
            [out["output_mask_targets"] for out in self.validation_outputs]
        )

        # Calculate separate color accuracies
        input_color_accuracy = (
            (input_color_predictions == input_color_targets).float().mean()
        )
        output_color_accuracy = (
            (output_color_predictions == output_color_targets).float().mean()
        )

        # Calculate separate mask accuracies
        input_mask_accuracy = (
            (input_mask_predictions == input_mask_targets).float().mean()
        )
        output_mask_accuracy = (
            (output_mask_predictions == output_mask_targets).float().mean()
        )

        # Overall accuracies
        all_color_predictions = torch.cat(
            [input_color_predictions, output_color_predictions]
        )
        all_color_targets = torch.cat([input_color_targets, output_color_targets])
        overall_color_accuracy = (
            (all_color_predictions == all_color_targets).float().mean()
        )

        all_mask_predictions = torch.cat(
            [input_mask_predictions, output_mask_predictions]
        )
        all_mask_targets = torch.cat([input_mask_targets, output_mask_targets])
        overall_mask_accuracy = (
            (all_mask_predictions == all_mask_targets).float().mean()
        )

        # Calculate epoch loss
        epoch_loss = torch.stack(
            [out["val_loss"] for out in self.validation_outputs]
        ).mean()

        # Log all metrics
        self.log("val_epoch_color_acc", overall_color_accuracy, prog_bar=True)
        self.log("val_epoch_mask_acc", overall_mask_accuracy, prog_bar=True)
        self.log("val_epoch_loss", epoch_loss, prog_bar=True)
        self.log("val_input_color_acc", input_color_accuracy, prog_bar=True)
        self.log("val_output_color_acc", output_color_accuracy, prog_bar=True)
        self.log("val_input_mask_acc", input_mask_accuracy, prog_bar=True)
        self.log("val_output_mask_acc", output_mask_accuracy, prog_bar=True)

        # For early stopping, we need BOTH input and output colors AND masks to be 100%
        colors_perfect = (input_color_accuracy >= 1.0) and (
            output_color_accuracy >= 1.0
        )
        masks_perfect = (input_mask_accuracy >= 1.0) and (output_mask_accuracy >= 1.0)
        both_perfect = colors_perfect and masks_perfect
        self.log("val_both_perfect", float(both_perfect), prog_bar=True)

        # Debug output every 20 epochs
        if self.current_epoch % 20 == 0:
            print(f"\nEpoch {self.current_epoch} validation results:")
            print(
                f"  INPUT color accuracy: {input_color_accuracy:.4f} ({input_color_targets.numel()} pixels)"
            )
            print(
                f"  OUTPUT color accuracy: {output_color_accuracy:.4f} ({output_color_targets.numel()} pixels)"
            )
            print(
                f"  INPUT mask accuracy: {input_mask_accuracy:.4f} ({input_mask_targets.numel()} pixels)"
            )
            print(
                f"  OUTPUT mask accuracy: {output_mask_accuracy:.4f} ({output_mask_targets.numel()} pixels)"
            )
            print(f"  Colors perfect: {colors_perfect}, Masks perfect: {masks_perfect}")

            # Visualization every 10 epochs (same frequency as training)
        if self.current_epoch % 10 == 0 and self.validation_outputs:
            # Get first validation batch for visualization
            first_batch = self.validation_outputs[0]
            if "full_output_color_predictions" in first_batch:
                # Use the full grids (first example from the batch)
                full_pred = first_batch["full_output_color_predictions"]
                full_true = first_batch["full_output_color_targets"]

                if len(full_pred) > 0:
                    # Show validation visualization for each example in the first batch
                    for i in range(len(full_pred)):
                        output_pred = full_pred[i]  # [30, 30]
                        output_true = full_true[i]  # [30, 30]

                        # Show validation visualization using the same function
                        self._print_prediction_results(
                            output_pred,
                            output_true,
                            f"VALIDATION OUTPUT - Epoch {self.current_epoch}",
                        )

        # Print per-index accuracy
        if self.index_metrics:
            print(f"\nPer-Index Accuracy (Epoch {self.current_epoch}):")
            print(
                f"{'Index':<8} {'Input Color':<12} {'Output Color':<12} {'Input Mask':<12} {'Output Mask':<12} {'All Perfect':<12}"
            )
            print("-" * 70)

            for idx in sorted(self.index_metrics.keys()):
                metrics = self.index_metrics[idx]

                input_color_acc = 0.0
                if metrics["input_color_total"] > 0:
                    input_color_acc = (
                        metrics["input_color_correct"] / metrics["input_color_total"]
                    )

                output_color_acc = 0.0
                if metrics["output_color_total"] > 0:
                    output_color_acc = (
                        metrics["output_color_correct"] / metrics["output_color_total"]
                    )

                input_mask_acc = 0.0
                if metrics["input_mask_total"] > 0:
                    input_mask_acc = (
                        metrics["input_mask_correct"] / metrics["input_mask_total"]
                    )

                output_mask_acc = 0.0
                if metrics["output_mask_total"] > 0:
                    output_mask_acc = (
                        metrics["output_mask_correct"] / metrics["output_mask_total"]
                    )

                all_perfect = (
                    (input_color_acc >= 1.0)
                    and (output_color_acc >= 1.0)
                    and (input_mask_acc >= 1.0)
                    and (output_mask_acc >= 1.0)
                )

                print(
                    f"{idx:<8} {input_color_acc:<12.2%} {output_color_acc:<12.2%} {input_mask_acc:<12.2%} {output_mask_acc:<12.2%} {'✓' if all_perfect else '✗':<12}"
                )

        # Clear validation outputs and index metrics for next epoch
        self.validation_outputs = []
        self.index_metrics = {}

    def configure_optimizers(self) -> Dict[str, Any]:
        # Use SGD with momentum for faster convergence on small dataset
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Very aggressive learning rate schedule
        def lr_lambda(epoch: int) -> float:
            if epoch < 2:
                return (epoch + 1) / 2  # Quick warmup
            elif epoch < 30:
                return 1.0  # Full LR
            elif epoch < 50:
                return 0.5  # Half LR
            else:
                return 0.1  # Low LR for fine-tuning

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def set_available_colors(self, colors: List[int]) -> None:
        """Set the available colors for this model."""
        self.available_colors = colors

    def visualize_predictions(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        prefix: str,
    ) -> None:
        """Visualize predictions vs ground truth for training."""
        self._print_prediction_results(
            predictions[0].argmax(dim=-1).cpu(),
            targets[0].cpu(),
            f"{prefix.upper()} - Epoch {self.current_epoch}",
        )

    def _print_prediction_results(
        self, pred_colors: torch.Tensor, true_colors: torch.Tensor, header: str
    ) -> None:
        """Shared visualization function for both training and validation."""
        # Create visualization
        print(f"\n{'='*60}")
        print(header)
        print(f"{'='*60}")

        # Show ground truth
        print("\nGround Truth colors:")
        imshow(true_colors, title=None, show_legend=True)

        # Show predictions
        print("\nPredicted colors:")
        imshow(pred_colors, title=None, show_legend=True)

        # Calculate accuracy for this example
        valid_mask = true_colors != -1
        if valid_mask.any():
            accuracy = (
                (pred_colors[valid_mask] == true_colors[valid_mask]).float().mean()
            )
            print(f"\nAccuracy: {accuracy:.2%}")

            # Per-color accuracy
            print("\nPer-color accuracy:")
            unique_colors = torch.unique(true_colors[valid_mask])
            for color in unique_colors:
                color_mask = true_colors == color
                if color_mask.any():
                    color_acc = (pred_colors[color_mask] == color).float().mean()
                    print(
                        f"  Color {color.item()}: {color_acc:.2%} ({color_mask.sum().item()} pixels)"
                    )


class UnetSpatialMessagePassing(nn.Module):
    """Residual U-Net style spatial message passing for multi-scale feature learning."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Calculate group sizes for efficient group convolutions
        # Use groups of 16 for better balance between efficiency and expressiveness
        groups_base = min(16, hidden_dim)
        groups_2x = min(16, hidden_dim * 2)
        groups_4x = min(16, hidden_dim * 4)

        # RESIDUAL OPTIMIZATION: Add residual connections throughout the U-Net
        # This improves gradient flow and training stability

        # Encoder layers - progressively downsample
        # 30x30 -> 30x30 (no stride) - RESIDUAL BLOCK
        self.enc1_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=groups_base
        )  # Group conv for efficiency
        self.enc1_norm = nn.LayerNorm([hidden_dim, 30, 30])
        self.enc1_activation = nn.GELU()
        self.enc1_dropout = nn.Dropout(dropout)

        # 30x30 -> 15x15 (stride=2) - RESIDUAL BLOCK
        self.enc2_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=groups_base,
        )  # Group conv for efficiency
        self.enc2_norm = nn.LayerNorm([hidden_dim * 2, 15, 15])
        self.enc2_activation = nn.GELU()
        self.enc2_dropout = nn.Dropout(dropout)

        # Residual connection for enc2 (needs projection due to dimension change)
        self.enc2_residual = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, stride=2),
            nn.LayerNorm([hidden_dim * 2, 15, 15]),
        )

        # Bottleneck - 15x15 -> 15x15 - RESIDUAL BLOCK
        self.bottleneck_conv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,  # Keep same dimensions for residual
            kernel_size=3,
            padding=1,
            groups=groups_2x,
        )  # Group conv for efficiency
        self.bottleneck_norm = nn.LayerNorm([hidden_dim * 2, 15, 15])
        self.bottleneck_activation = nn.GELU()
        self.bottleneck_dropout = nn.Dropout(dropout)

        # Decoder layers - progressively upsample
        # 15x15 -> 30x30: (15-1)*2 - 2*1 + 3 + 1 = 30 - RESIDUAL BLOCK
        self.dec2_conv = nn.ConvTranspose2d(
            hidden_dim * 4,  # Input has skip connection: hidden_dim*2 + hidden_dim*2
            hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=groups_base,  # Group conv for efficiency
        )
        self.dec2_norm = nn.LayerNorm([hidden_dim, 30, 30])
        self.dec2_activation = nn.GELU()
        self.dec2_dropout = nn.Dropout(dropout)

        # Residual connection for dec2 (upsample the skip connection)
        self.dec2_residual = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim * 4, hidden_dim, kernel_size=1, stride=2, output_padding=1
            ),
            nn.LayerNorm([hidden_dim, 30, 30]),
        )

        # Final layer - 30x30 -> 30x30 (no stride) - RESIDUAL BLOCK
        self.dec1_conv = nn.Conv2d(
            hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, groups=groups_base
        )  # Group conv for efficiency
        self.dec1_norm = nn.LayerNorm([hidden_dim, 30, 30])
        self.dec1_activation = nn.GELU()
        self.dec1_dropout = nn.Dropout(dropout)

        # Residual connection for dec1 (needs projection due to dimension change)
        self.dec1_residual = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.LayerNorm([hidden_dim, 30, 30]),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, 30, 30, hidden_dim]
        # Convert to conv format
        x = h.permute(0, 3, 1, 2)  # [B, hidden_dim, 30, 30]

        # RESIDUAL ENCODER PATH with skip connections
        # Encoder 1: 30x30 -> 30x30 (same dimensions, direct residual)
        residual = x
        e1 = self.enc1_conv(x)
        e1 = self.enc1_norm(e1)
        e1 = self.enc1_activation(e1)
        e1 = self.enc1_dropout(e1)
        e1 = e1 + residual  # RESIDUAL CONNECTION

        # Encoder 2: 30x30 -> 15x15 (dimension change, projected residual)
        residual = self.enc2_residual(e1)
        e2 = self.enc2_conv(e1)
        e2 = self.enc2_norm(e2)
        e2 = self.enc2_activation(e2)
        e2 = self.enc2_dropout(e2)
        e2 = e2 + residual  # RESIDUAL CONNECTION

        # Bottleneck: 15x15 -> 15x15 (same dimensions, direct residual)
        residual = e2
        b = self.bottleneck_conv(e2)
        b = self.bottleneck_norm(b)
        b = self.bottleneck_activation(b)
        b = self.bottleneck_dropout(b)
        b = b + residual  # RESIDUAL CONNECTION

        # RESIDUAL DECODER PATH with U-Net skip connections
        # Skip connection from encoder + residual connection
        d3_input = torch.cat([b, e2], dim=1)  # [B, hidden_dim*4, 15, 15]

        # Decoder 2: 15x15 -> 30x30 (dimension change, projected residual)
        residual = self.dec2_residual(d3_input)
        d2 = self.dec2_conv(d3_input)
        d2 = self.dec2_norm(d2)
        d2 = self.dec2_activation(d2)
        d2 = self.dec2_dropout(d2)
        d2 = d2 + residual  # RESIDUAL CONNECTION

        # Add U-Net skip connection
        d2_input = torch.cat([d2, e1], dim=1)  # [B, hidden_dim*2, 30, 30]

        # Decoder 1: 30x30 -> 30x30 (dimension change, projected residual)
        residual = self.dec1_residual(d2_input)
        d1 = self.dec1_conv(d2_input)
        d1 = self.dec1_norm(d1)
        d1 = self.dec1_activation(d1)
        d1 = self.dec1_dropout(d1)
        d1 = d1 + residual  # RESIDUAL CONNECTION

        # Convert back to original format
        output = d1.permute(0, 2, 3, 1)  # [B, 30, 30, hidden_dim]

        return output


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


class PerfectAccuracyEarlyStopping(Callback):
    """Custom callback that stops training when both INPUT and OUTPUT achieve 100% accuracy for patience epochs."""

    def __init__(self, patience: int = 5, min_epochs: int = 200):
        super().__init__()
        self.patience = patience
        self.min_epochs = min_epochs
        self.perfect_epochs = 0
        self.best_epoch = -1

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the current epoch metrics
        input_color_acc = trainer.callback_metrics.get("val_input_color_acc", 0.0)
        output_color_acc = trainer.callback_metrics.get("val_output_color_acc", 0.0)
        input_mask_acc = trainer.callback_metrics.get("val_input_mask_acc", 0.0)
        output_mask_acc = trainer.callback_metrics.get("val_output_mask_acc", 0.0)

        # Check if both colors and masks are perfect (>= 1.0)
        colors_perfect = (input_color_acc >= 1.0) and (output_color_acc >= 1.0)
        masks_perfect = (input_mask_acc >= 1.0) and (output_mask_acc >= 1.0)
        both_perfect = colors_perfect and masks_perfect

        if both_perfect:
            self.perfect_epochs += 1
            if self.best_epoch == -1:
                self.best_epoch = trainer.current_epoch

            print(
                f"\nBoth INPUT and OUTPUT achieved 100% accuracy for colors AND masks! ({self.perfect_epochs}/{self.patience} epochs)"
            )

            # Only stop if we've reached min_epochs and maintained perfect accuracy for patience epochs
            if (
                self.perfect_epochs >= self.patience
                and trainer.current_epoch >= self.min_epochs
            ):
                print(
                    f"Stopping training - perfect accuracy maintained for {self.patience} consecutive epochs (after {trainer.current_epoch} total epochs)"
                )
                trainer.should_stop = True
            elif trainer.current_epoch < self.min_epochs:
                print(
                    f"Perfect accuracy achieved but continuing to minimum {self.min_epochs} epochs (current: {trainer.current_epoch})"
                )
        else:
            # Reset counter if not perfect
            if self.perfect_epochs > 0:
                print(f"\nLost perfect accuracy after {self.perfect_epochs} epochs")
            self.perfect_epochs = 0
            self.best_epoch = -1


def load_single_file_data(
    filename: str, data_dir: str = "/tmp/arc_data", subset: Optional[str] = None
) -> Tuple[TensorDataset, List[int]]:
    """Load data for a single filename with optional subset filtering.

    Args:
        filename: The filename to load (without .json extension)
        data_dir: Directory containing the data
        subset: 'train' or 'test' to filter by subset, None for all

    Returns:
        dataset: TensorDataset with the loaded data
        available_colors: List of unique color indices found in the data
    """
    # Load preprocessed data
    train_data = np.load("processed_data/train_all.npz")

    # Get indices for this filename
    filenames = train_data["filenames"]
    file_indices = np.where(filenames == filename + ".json")[0]

    if len(file_indices) == 0:
        raise ValueError(f"No training examples found for {filename}")

    # Apply subset filtering if requested
    if subset is not None and "subset_example_index_is_train" in train_data:
        subset_is_train = train_data["subset_example_index_is_train"][file_indices]
        if subset == "train":
            subset_mask = subset_is_train
        else:  # test
            subset_mask = ~subset_is_train

        file_indices = file_indices[subset_mask]

        if len(file_indices) == 0:
            raise ValueError(f"No {subset} examples found for {filename}")

    subset_name = f" {subset}" if subset else ""
    print(f"Found {len(file_indices)}{subset_name} examples for {filename}")

    # Extract features and colors
    input_features = torch.from_numpy(
        train_data["inputs_features"][file_indices]
    ).float()
    output_features = torch.from_numpy(
        train_data["outputs_features"][file_indices]
    ).float()
    input_colors = torch.from_numpy(train_data["inputs"][file_indices]).long()
    output_colors = torch.from_numpy(train_data["outputs"][file_indices]).long()

    # Extract masks
    input_masks = torch.from_numpy(train_data["inputs_mask"][file_indices]).bool()
    output_masks = torch.from_numpy(train_data["outputs_mask"][file_indices]).bool()

    # Extract example indices (the index within the file)
    example_indices = torch.from_numpy(train_data["indices"][file_indices]).long()

    # Get available colors
    all_colors = torch.cat([input_colors.flatten(), output_colors.flatten()])
    available_colors = torch.unique(all_colors[all_colors != -1]).tolist()

    print(f"Available colors for {filename}: {available_colors}")
    print(f"Example indices: {example_indices.tolist()}")

    # Create dataset
    dataset = TensorDataset(
        input_features,
        output_features,
        input_colors,
        output_colors,
        input_masks,
        output_masks,
        example_indices,  # Use actual example indices
        torch.zeros(len(file_indices)),  # Dummy
    )

    return dataset, available_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/tmp/arc_data", help="Data directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="optimized_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500, help="Maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num_message_rounds",
        type=int,
        default=12,
        help="Number of message passing rounds",
    )
    parser.add_argument("--dropout", type=float, default=0.02, help="Dropout rate")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for scaling"
    )
    parser.add_argument(
        "--filename", type=str, default="3345333e", help="Filename to train on"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (epochs to wait after reaching 100% accuracy)",
    )
    parser.add_argument(
        "--single_file_mode",
        action="store_true",
        help="Train on 'train' subset and evaluate on 'test' subset from the same file",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.10,
        help="Probability of applying binary noise to features during training (default: 0.05)",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=200,
        help="Minimum number of epochs to train before early stopping (default: 200)",
    )

    args = parser.parse_args()

    # Target filename
    filename = args.filename

    if args.single_file_mode:
        print(f"\n=== Single File Mode ===")
        print(f"Training on 'train' subset of {filename}")
        print(f"Evaluating on 'test' subset of {filename}")

        # Load train and test subsets separately
        train_dataset, available_colors = load_single_file_data(
            filename, args.data_dir, subset="train"
        )
        test_dataset, _ = load_single_file_data(filename, args.data_dir, subset="test")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=True
        )
        val_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )
    else:
        # Load all data (no subset filtering)
        train_dataset, available_colors = load_single_file_data(filename, args.data_dir)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        test_loader = None

    # Create model
    model = OptimizedMemorizationModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_message_rounds=args.num_message_rounds,
        num_classes=10,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        filename=filename,
        num_train_examples=len(train_dataset),
    )

    # Set noise probability
    model.noise_prob = args.noise_prob

    # Set available colors
    model.set_available_colors(available_colors)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Create trainer with callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_color_acc",  # Monitor epoch-level color accuracy
        dirpath=os.path.join(args.checkpoint_dir, "checkpoints"),
        filename=f"optimized-{{epoch:02d}}-{{val_epoch_loss:.4f}}-{{val_epoch_color_acc:.4f}}-{{val_epoch_mask_acc:.4f}}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    # Add custom callback for perfect accuracy
    perfect_accuracy_callback = PerfectAccuracyEarlyStopping(
        patience=args.patience, min_epochs=args.min_epochs
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, perfect_accuracy_callback],
        default_root_dir=args.checkpoint_dir,
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed for {filename}")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")

    # Evaluate on test set if in single file mode
    if args.single_file_mode and test_loader is not None:
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)

        # Run test evaluation
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                (
                    input_features,
                    output_features,
                    input_colors,
                    output_colors,
                    input_masks,
                    output_masks,
                    _,
                    _,
                ) = batch

                # Get predictions
                input_color_logits, input_mask_logits = model(input_features)
                output_color_logits, output_mask_logits = model(output_features)

                input_color_predictions = input_color_logits.argmax(dim=-1)
                output_color_predictions = output_color_logits.argmax(dim=-1)
                input_mask_predictions = (input_mask_logits > 0).squeeze(-1)
                output_mask_predictions = (output_mask_logits > 0).squeeze(-1)

                # Calculate color accuracies
                input_color_valid_mask = input_colors != -1
                output_color_valid_mask = output_colors != -1

                input_color_correct = 0
                input_color_total = 0
                output_color_correct = 0
                output_color_total = 0

                if input_color_valid_mask.any():
                    input_color_correct = (
                        (
                            input_color_predictions[input_color_valid_mask]
                            == input_colors[input_color_valid_mask]
                        )
                        .sum()
                        .item()
                    )
                    input_color_total = input_color_valid_mask.sum().item()

                if output_color_valid_mask.any():
                    output_color_correct = (
                        (
                            output_color_predictions[output_color_valid_mask]
                            == output_colors[output_color_valid_mask]
                        )
                        .sum()
                        .item()
                    )
                    output_color_total = output_color_valid_mask.sum().item()

                # Calculate mask accuracies
                input_mask_correct = (
                    (input_mask_predictions == input_masks).sum().item()
                )
                output_mask_correct = (
                    (output_mask_predictions == output_masks).sum().item()
                )
                input_mask_total = input_masks.numel()
                output_mask_total = output_masks.numel()

                # Print results
                print(f"\nTest Set Results:")
                print(
                    f"INPUT color accuracy:  {input_color_correct}/{input_color_total} = {input_color_correct/input_color_total*100:.1f}%"
                    if input_color_total > 0
                    else "No input color pixels"
                )
                print(
                    f"OUTPUT color accuracy: {output_color_correct}/{output_color_total} = {output_color_correct/output_color_total*100:.1f}%"
                    if output_color_total > 0
                    else "No output color pixels"
                )
                print(
                    f"INPUT mask accuracy:  {input_mask_correct}/{input_mask_total} = {input_mask_correct/input_mask_total*100:.1f}%"
                )
                print(
                    f"OUTPUT mask accuracy: {output_mask_correct}/{output_mask_total} = {output_mask_correct/output_mask_total*100:.1f}%"
                )

                total_color_correct = input_color_correct + output_color_correct
                total_color_pixels = input_color_total + output_color_total
                total_mask_correct = input_mask_correct + output_mask_correct
                total_mask_pixels = input_mask_total + output_mask_total

                if total_color_pixels > 0:
                    print(
                        f"OVERALL color accuracy: {total_color_correct}/{total_color_pixels} = {total_color_correct/total_color_pixels*100:.1f}%"
                    )
                print(
                    f"OVERALL mask accuracy: {total_mask_correct}/{total_mask_pixels} = {total_mask_correct/total_mask_pixels*100:.1f}%"
                )

                # Show visual comparison for first example
                if len(input_features) > 0:
                    print("\nVisual comparison of first test example:")
                    # model.visualize_predictions(
                    #     input_features[0:1],
                    #     input_colors[0:1],
                    #     input_color_logits[0:1],
                    #     "TEST INPUT",
                    # )
                    model.visualize_predictions(
                        output_features[0:1],
                        output_colors[0:1],
                        output_color_logits[0:1],
                        "TEST OUTPUT",
                    )

    # Save the final trained model with filename-specific name
    final_model_path = os.path.join(args.checkpoint_dir, f"color_model_{filename}.pt")

    # Save both model state dict and metadata
    model_save_data = {
        "model_state_dict": model.state_dict(),
        "available_colors": available_colors,
        "filename": filename,
        "hyperparameters": model.hparams,
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


def load_color_model_for_inference(
    filename: str, checkpoint_dir: str = "color_mapping_outputs"
) -> OptimizedMemorizationModel:
    """Load a trained color and mask model for inference."""
    model_path = os.path.join(checkpoint_dir, f"color_model_{filename}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for filename: {filename} at {model_path}"
        )

    # Load saved data (weights_only=False for our trusted models to handle custom classes)
    saved_data = torch.load(model_path, map_location="cpu", weights_only=False)

    # Create model with saved hyperparameters
    hparams = saved_data["hyperparameters"]
    model = OptimizedMemorizationModel(
        feature_dim=hparams.get("feature_dim", 147),
        hidden_dim=hparams.get("hidden_dim", 512),
        num_message_rounds=hparams.get("num_message_rounds", 12),
        num_classes=hparams.get("num_classes", 10),
        dropout=hparams.get("dropout", 0.0),
        lr=hparams.get("lr", 0.1),
        weight_decay=hparams.get("weight_decay", 0.0),
        temperature=hparams.get("temperature", 0.01),
        filename=saved_data["filename"],
        num_train_examples=hparams.get("num_train_examples", 2),
    )

    # Load model weights
    model.load_state_dict(saved_data["model_state_dict"])

    # Set available colors
    model.set_available_colors(saved_data["available_colors"])

    # Set to evaluation mode
    model.eval()

    print(
        f"Loaded color and mask model for {filename} with colors: {saved_data['available_colors']}"
    )

    return model


if __name__ == "__main__":
    main()
