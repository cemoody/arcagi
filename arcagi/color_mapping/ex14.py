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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OptimizedMemorizationModel(pl.LightningModule):
    """
    Optimized model for perfect memorization with minimal parameters.

    Key optimizations:
    1. Example-aware feature transformation
    2. Efficient spatial message passing
    3. Adaptive learning rate with strong warmup
    4. Focused architecture for 2 training examples
    5. Batch normalization for faster convergence
    6. Better weight initialization
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

        # Example-specific transformation (key for memorization)
        self.example_embeddings = nn.Parameter(
            torch.randn(num_train_examples, hidden_dim) * 0.1
        )

        # Feature extraction with layer normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Position embeddings (learnable for fast adaptation)
        self.pos_embedding = nn.Parameter(torch.randn(30, 30, hidden_dim) * 0.02)

        # Efficient message passing
        self.message_layers = nn.ModuleList(
            [
                SpatialMessagePassing(hidden_dim, dropout=dropout)
                for _ in range(num_message_rounds)
            ]
        )

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

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, features: torch.Tensor, example_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = features.shape[0]

        # Extract base features
        h = self.feature_extractor(features)  # [B, 30, 30, hidden_dim]

        # Add example-specific modulation if available
        if example_idx is not None and self.training:
            # Use example embeddings to modulate features
            example_embeds = self.example_embeddings[example_idx]  # [B, hidden_dim]
            h = h + example_embeds.unsqueeze(1).unsqueeze(2)

        # Add position embeddings
        h = h + self.pos_embedding.unsqueeze(0)

        # Spatial message passing with stronger residuals
        for i, message_layer in enumerate(self.message_layers):
            h_new = message_layer(h)
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

        # Add L2 regularization on example embeddings to prevent overfitting
        reg_loss = 0.001 * self.example_embeddings.pow(2).mean()
        loss = loss + reg_loss

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
            _,
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

        self.log("val_loss", total_loss)
        self.log("val_acc", accuracy)

        return {"val_loss": total_loss, "val_acc": accuracy}

    def configure_optimizers(self) -> Dict[str, Any]:
        # Use SGD with momentum for faster convergence on small dataset
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.95,
            weight_decay=self.weight_decay,
            nesterov=True,
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
        pred_colors = predictions[0].argmax(dim=-1).cpu().numpy()
        true_colors = targets[0].cpu().numpy()

        # Create visualization
        print(f"\n{'='*60}")
        print(f"{prefix.upper()} - Epoch {self.current_epoch}")
        print(f"{'='*60}")

        # Show ground truth
        print("\nGround Truth colors:")
        self._print_grid(true_colors)

        # Show predictions
        print("\nPredicted colors:")
        self._print_grid(pred_colors)

        # Calculate accuracy for this example
        valid_mask = true_colors != -1
        if valid_mask.any():
            accuracy = (pred_colors[valid_mask] == true_colors[valid_mask]).mean()
            print(f"\nAccuracy: {accuracy:.2%}")

            # Per-color accuracy
            print("\nPer-color accuracy:")
            unique_colors = np.unique(true_colors[valid_mask])
            for color in unique_colors:
                color_mask = true_colors == color
                if color_mask.any():
                    color_acc = (pred_colors[color_mask] == color).mean()
                    print(
                        f"  Color {color}: {color_acc:.2%} ({color_mask.sum()} pixels)"
                    )

    def _print_grid(self, grid: np.ndarray) -> None:
        """Print a grid with colors."""
        # Only show the central 20x20 region for readability
        center = grid[5:25, 5:25]

        for row in center:
            row_str = ""
            for val in row:
                if val == -1:
                    row_str += "  "
                else:
                    row_str += f"{val:2d}"
                row_str += " "
            print(row_str)

        # Show legend of colors present
        unique_colors = np.unique(grid[grid != -1])
        if len(unique_colors) > 0:
            print("\nLegend:")
            print(" ".join([f"{c:2d}" for c in unique_colors]))


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


def load_single_file_data(
    filename: str, data_dir: str = "/tmp/arc_data"
) -> Tuple[List[torch.Tensor], List[int]]:
    """Load data for a single filename."""
    # Load preprocessed data
    train_data = np.load("processed_data/train_all.npz")

    # Get indices for this filename
    filenames = train_data["filenames"]
    file_indices = np.where(filenames == filename + ".json")[0]

    if len(file_indices) == 0:
        raise ValueError(f"No training examples found for {filename}")

    print(f"Found {len(file_indices)} train examples for {filename}")

    # Extract features and colors
    input_features = torch.from_numpy(
        train_data["inputs_features"][file_indices]
    ).float()
    output_features = torch.from_numpy(
        train_data["outputs_features"][file_indices]
    ).float()
    input_colors = torch.from_numpy(train_data["inputs"][file_indices]).long()
    output_colors = torch.from_numpy(train_data["outputs"][file_indices]).long()

    # Get available colors
    all_colors = torch.cat([input_colors.flatten(), output_colors.flatten()])
    available_colors = torch.unique(all_colors[all_colors != -1]).tolist()

    print(f"Available colors for {filename}: {available_colors}")

    # Create dataset
    dataset = TensorDataset(
        input_features,
        output_features,
        input_colors,
        output_colors,
        torch.zeros(len(file_indices)),  # Dummy
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
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
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

    args = parser.parse_args()

    # Target filename
    filename = args.filename

    # Load data
    train_dataset, available_colors = load_single_file_data(filename, args.data_dir)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

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

    # Set available colors
    model.set_available_colors(available_colors)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, "checkpoints"),
        filename="optimized-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=args.checkpoint_dir,
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed for {filename}")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
