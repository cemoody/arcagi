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

    This version (ex16) uses sinusoidal position embeddings instead of learnable ones.

    Key optimizations:
    1. Example-aware feature transformation
    2. Efficient spatial message passing
    3. Adaptive learning rate with strong warmup
    4. Focused architecture for 2 training examples
    5. Batch normalization for faster convergence
    6. Better weight initialization
    7. Sinusoidal position embeddings (parameter-free)
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 512,
        num_message_rounds: int = 6,
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

        # Track per-index metrics
        self.index_metrics = {}

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
        self.shared_message_layer = SpatialMessagePassing(hidden_dim, dropout=dropout)

        # Color-specific prediction heads for better specialization
        self.color_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_classes)]
        )

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
    ) -> torch.Tensor:
        batch_size = features.shape[0]

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

        # Color-specific predictions
        logits_list = []
        for color_head in self.color_heads:
            logits_list.append(color_head(h))  # [B, 30, 30, 1]

        logits = torch.cat(logits_list, dim=-1)  # [B, 30, 30, num_classes]

        # Temperature scaling
        logits = logits / self.temperature

        # Apply color constraints
        logits = self.apply_color_constraints(logits)

        return logits

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
            logits = self(features)
            return logits.argmax(dim=-1)

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        (
            input_features,
            output_features,
            input_colors,
            output_colors,
            _,
            _,
        ) = batch

        # Get example indices for this batch
        example_indices = (
            torch.arange(len(input_features), device=self.device)
            % self.num_train_examples
        )

        # Process inputs with example information
        input_logits = self(input_features, example_indices)
        input_loss = F.cross_entropy(
            input_logits.reshape(-1, input_logits.size(-1)), input_colors.reshape(-1)
        )

        # Process outputs with example information
        output_logits = self(output_features, example_indices)
        output_loss = F.cross_entropy(
            output_logits.reshape(-1, output_logits.size(-1)), output_colors.reshape(-1)
        )

        # Combined loss with focus on harder examples
        loss = input_loss + output_loss

        # Calculate accuracy
        all_logits = torch.cat(
            [input_logits.flatten(0, 2), output_logits.flatten(0, 2)]
        )
        all_targets = torch.cat([input_colors.flatten(), output_colors.flatten()])
        predictions = all_logits.argmax(dim=-1)

        valid_mask = all_targets != -1
        if valid_mask.any():
            accuracy = (
                (predictions[valid_mask] == all_targets[valid_mask]).float().mean()
            )
        else:
            accuracy = torch.tensor(0.0)

        self.log("train_loss", loss)
        self.log("train_acc", accuracy)

        # Visualize every 10 epochs
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            self.visualize_predictions(
                input_features[0:1], input_colors[0:1], input_logits[0:1], "input"
            )
            self.visualize_predictions(
                output_features[0:1], output_colors[0:1], output_logits[0:1], "output"
            )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        (
            input_features,
            output_features,
            input_colors,
            output_colors,
            example_indices,
            _,
        ) = batch

        # For validation, we don't use example indices
        input_logits = self(input_features)
        output_logits = self(output_features)

        # Compute losses
        input_loss = F.cross_entropy(
            input_logits.reshape(-1, input_logits.size(-1)),
            input_colors.reshape(-1),
        )
        output_loss = F.cross_entropy(
            output_logits.reshape(-1, output_logits.size(-1)),
            output_colors.reshape(-1),
        )

        total_loss = input_loss + output_loss

        # Compute accuracy
        input_predictions = input_logits.argmax(dim=-1)  # [B, 30, 30]
        output_predictions = output_logits.argmax(dim=-1)  # [B, 30, 30]

        # Create separate masks for input and output
        input_valid_mask = input_colors != -1
        output_valid_mask = output_colors != -1

        # Calculate accuracies
        input_accuracy = torch.tensor(0.0)
        output_accuracy = torch.tensor(0.0)

        if input_valid_mask.any():
            input_accuracy = (
                (input_predictions[input_valid_mask] == input_colors[input_valid_mask])
                .float()
                .mean()
            )

        if output_valid_mask.any():
            output_accuracy = (
                (
                    output_predictions[output_valid_mask]
                    == output_colors[output_valid_mask]
                )
                .float()
                .mean()
            )

        # Combined accuracy for logging
        all_predictions = torch.cat(
            [input_predictions.flatten(), output_predictions.flatten()]
        )
        all_targets = torch.cat([input_colors.flatten(), output_colors.flatten()])
        valid_mask = all_targets != -1

        if valid_mask.any():
            accuracy = (
                (all_predictions[valid_mask] == all_targets[valid_mask]).float().mean()
            )
        else:
            accuracy = torch.tensor(0.0)

        self.log("val_loss", total_loss)
        self.log("val_acc", accuracy)

        # Calculate per-index metrics
        for i in range(len(example_indices)):
            idx = example_indices[i].item()

            # Calculate input accuracy for this example
            input_valid_pixels = input_valid_mask[i].sum().item()
            input_correct_pixels = 0
            if input_valid_pixels > 0:
                input_correct_pixels = (
                    (
                        input_predictions[i][input_valid_mask[i]]
                        == input_colors[i][input_valid_mask[i]]
                    )
                    .sum()
                    .item()
                )

            # Calculate output accuracy for this example
            output_valid_pixels = output_valid_mask[i].sum().item()
            output_correct_pixels = 0
            if output_valid_pixels > 0:
                output_correct_pixels = (
                    (
                        output_predictions[i][output_valid_mask[i]]
                        == output_colors[i][output_valid_mask[i]]
                    )
                    .sum()
                    .item()
                )

            # Store metrics for this index
            if idx not in self.index_metrics:
                self.index_metrics[idx] = {
                    "input_correct": 0,
                    "input_total": 0,
                    "output_correct": 0,
                    "output_total": 0,
                    "count": 0,
                }

            self.index_metrics[idx]["input_correct"] += input_correct_pixels
            self.index_metrics[idx]["input_total"] += input_valid_pixels
            self.index_metrics[idx]["output_correct"] += output_correct_pixels
            self.index_metrics[idx]["output_total"] += output_valid_pixels
            self.index_metrics[idx]["count"] += 1

        # Store outputs for epoch-level metrics
        self.validation_outputs.append(
            {
                "val_loss": total_loss,
                "val_acc": accuracy,
                "input_predictions": input_predictions[input_valid_mask],
                "input_targets": input_colors[input_valid_mask],
                "output_predictions": output_predictions[output_valid_mask],
                "output_targets": output_colors[output_valid_mask],
                "batch_size": input_valid_mask.sum() + output_valid_mask.sum(),
            }
        )

        return {"val_loss": total_loss, "val_acc": accuracy}

    def on_validation_epoch_start(self) -> None:
        """Reset validation outputs at the start of each epoch."""
        self.validation_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Calculate epoch-level metrics from all validation batches."""
        if not self.validation_outputs:
            return

        # Gather INPUT predictions and targets
        input_predictions = torch.cat(
            [out["input_predictions"] for out in self.validation_outputs]
        )
        input_targets = torch.cat(
            [out["input_targets"] for out in self.validation_outputs]
        )

        # Gather OUTPUT predictions and targets
        output_predictions = torch.cat(
            [out["output_predictions"] for out in self.validation_outputs]
        )
        output_targets = torch.cat(
            [out["output_targets"] for out in self.validation_outputs]
        )

        # Calculate separate accuracies
        input_accuracy = (input_predictions == input_targets).float().mean()
        output_accuracy = (output_predictions == output_targets).float().mean()

        # Overall accuracy (for backward compatibility)
        all_predictions = torch.cat([input_predictions, output_predictions])
        all_targets = torch.cat([input_targets, output_targets])
        overall_accuracy = (all_predictions == all_targets).float().mean()

        # Calculate epoch loss
        epoch_loss = torch.stack(
            [out["val_loss"] for out in self.validation_outputs]
        ).mean()

        # Log all metrics
        self.log("val_epoch_acc", overall_accuracy, prog_bar=True)
        self.log("val_epoch_loss", epoch_loss, prog_bar=True)
        self.log("val_input_acc", input_accuracy, prog_bar=True)
        self.log("val_output_acc", output_accuracy, prog_bar=True)

        # For early stopping, we need BOTH input and output to be 100%
        both_perfect = (input_accuracy >= 1.0) and (output_accuracy >= 1.0)
        self.log("val_both_perfect", float(both_perfect), prog_bar=True)

        # Debug output every 20 epochs
        if self.current_epoch % 20 == 0:
            print(f"\nEpoch {self.current_epoch} validation results:")
            print(
                f"  INPUT accuracy: {input_accuracy:.4f} ({input_targets.numel()} pixels)"
            )
            print(
                f"  OUTPUT accuracy: {output_accuracy:.4f} ({output_targets.numel()} pixels)"
            )
            print(f"  Both perfect: {both_perfect}")

        # Print per-index accuracy
        if self.index_metrics:
            print(f"\nPer-Index Accuracy (Epoch {self.current_epoch}):")
            print(
                f"{'Index':<8} {'Input Acc':<12} {'Output Acc':<12} {'Both Perfect':<12}"
            )
            print("-" * 50)

            for idx in sorted(self.index_metrics.keys()):
                metrics = self.index_metrics[idx]

                input_acc = 0.0
                if metrics["input_total"] > 0:
                    input_acc = metrics["input_correct"] / metrics["input_total"]

                output_acc = 0.0
                if metrics["output_total"] > 0:
                    output_acc = metrics["output_correct"] / metrics["output_total"]

                both_perfect = (input_acc >= 1.0) and (output_acc >= 1.0)

                print(
                    f"{idx:<8} {input_acc:<12.2%} {output_acc:<12.2%} {'✓' if both_perfect else '✗':<12}"
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
        """Visualize predictions vs ground truth."""
        pred_colors = predictions[0].argmax(dim=-1).cpu()
        true_colors = targets[0].cpu()

        # Create visualization
        print(f"\n{'='*60}")
        print(f"{prefix.upper()} - Epoch {self.current_epoch}")
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
        input_acc = trainer.callback_metrics.get("val_input_acc", 0.0)
        output_acc = trainer.callback_metrics.get("val_output_acc", 0.0)

        # Check if both are perfect (>= 1.0)
        both_perfect = (input_acc >= 1.0) and (output_acc >= 1.0)

        if both_perfect:
            self.perfect_epochs += 1
            if self.best_epoch == -1:
                self.best_epoch = trainer.current_epoch

            print(
                f"\nBoth INPUT and OUTPUT achieved 100% accuracy! ({self.perfect_epochs}/{self.patience} epochs)"
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
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num_message_rounds",
        type=int,
        default=12,
        help="Number of message passing rounds",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Temperature for scaling"
    )
    parser.add_argument(
        "--filename", type=str, default="3345333e", help="Filename to train on"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
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
        default=0.05,
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
            train_dataset, batch_size=len(train_dataset), shuffle=False
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
        monitor="val_epoch_acc",  # Monitor epoch-level accuracy
        dirpath=os.path.join(args.checkpoint_dir, "checkpoints"),
        filename=f"optimized-{{epoch:02d}}-{{val_epoch_loss:.4f}}-{{val_epoch_acc:.4f}}",
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
                    _,
                    _,
                ) = batch

                # Get predictions
                input_logits = model(input_features)
                output_logits = model(output_features)

                input_predictions = input_logits.argmax(dim=-1)
                output_predictions = output_logits.argmax(dim=-1)

                # Calculate accuracies
                input_valid_mask = input_colors != -1
                output_valid_mask = output_colors != -1

                input_correct = 0
                input_total = 0
                output_correct = 0
                output_total = 0

                if input_valid_mask.any():
                    input_correct = (
                        (
                            input_predictions[input_valid_mask]
                            == input_colors[input_valid_mask]
                        )
                        .sum()
                        .item()
                    )
                    input_total = input_valid_mask.sum().item()

                if output_valid_mask.any():
                    output_correct = (
                        (
                            output_predictions[output_valid_mask]
                            == output_colors[output_valid_mask]
                        )
                        .sum()
                        .item()
                    )
                    output_total = output_valid_mask.sum().item()

                # Print results
                print(f"\nTest Set Results:")
                print(
                    f"INPUT accuracy:  {input_correct}/{input_total} = {input_correct/input_total*100:.1f}%"
                    if input_total > 0
                    else "No input pixels"
                )
                print(
                    f"OUTPUT accuracy: {output_correct}/{output_total} = {output_correct/output_total*100:.1f}%"
                    if output_total > 0
                    else "No output pixels"
                )

                total_correct = input_correct + output_correct
                total_pixels = input_total + output_total
                if total_pixels > 0:
                    print(
                        f"OVERALL accuracy: {total_correct}/{total_pixels} = {total_correct/total_pixels*100:.1f}%"
                    )

                # Show visual comparison for first example
                if len(input_features) > 0:
                    print("\nVisual comparison of first test example:")
                    model.visualize_predictions(
                        input_features[0:1],
                        input_colors[0:1],
                        input_logits[0:1],
                        "TEST INPUT",
                    )
                    model.visualize_predictions(
                        output_features[0:1],
                        output_colors[0:1],
                        output_logits[0:1],
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
    """Load a trained color model for inference."""
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
        num_message_rounds=hparams.get("num_message_rounds", 6),
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
        f"Loaded color model for {filename} with colors: {saved_data['available_colors']}"
    )

    return model


if __name__ == "__main__":
    main()
