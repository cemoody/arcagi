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


class PerfectMemorizationModel(pl.LightningModule):
    """
    Extremely aggressive model for perfect memorization of a single filename.

    Key strategies:
    1. Example-specific sub-networks (one for each training example)
    2. Very deep architecture with skip connections
    3. Spatial attention mechanisms
    4. Gradient accumulation for stable training
    5. Custom loss weighting for rare colors
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_classes: int = 10,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        temperature: float = 0.1,
        available_colors: Optional[List[int]] = None,
        num_examples: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.num_examples = num_examples

        # Register available colors as buffer
        if available_colors is not None:
            colors_tensor = torch.zeros(num_classes)
            colors_tensor[available_colors] = 1.0
            self.register_buffer("available_colors", colors_tensor)
        else:
            self.register_buffer("available_colors", torch.ones(num_classes))

        # Create example-specific networks
        self.example_networks = nn.ModuleList(
            [self._create_example_network() for _ in range(num_examples)]
        )

        # Global feature extractor with spatial attention
        self.global_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, 30, hidden_dim) * 0.02)

        # Example identifier network
        self.example_identifier = nn.Sequential(
            nn.Linear(feature_dim * 900, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_examples),
            nn.Softmax(dim=-1),
        )

        # Color-specific heads
        self.color_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_classes)]
        )

        # Learnable temperature for each example
        self.example_temperatures = nn.Parameter(torch.ones(num_examples) * temperature)

    def _create_example_network(self) -> nn.Module:
        """Create a deep network for a single example."""
        layers = []

        # Input projection
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ReLU())

        # Deep layers with residual connections
        for i in range(self.num_layers):
            layers.append(ResidualBlock(self.hidden_dim, self.dropout))

        # Output projection
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))

        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, is_input: bool = True) -> torch.Tensor:
        batch_size = features.shape[0]

        # Extract global features
        global_features = self.global_feature_extractor(
            features
        )  # [B, 30, 30, hidden_dim]

        # Add position embeddings
        global_features = global_features + self.pos_embedding

        # Apply spatial attention
        features_reshaped = global_features.permute(
            0, 3, 1, 2
        )  # [B, hidden_dim, 30, 30]
        attention_weights = self.spatial_attention(features_reshaped)  # [B, 1, 30, 30]
        attended_features = features_reshaped * attention_weights
        attended_features = attended_features.permute(
            0, 2, 3, 1
        )  # [B, 30, 30, hidden_dim]

        # Identify which example this is
        flat_features = features.reshape(batch_size, -1)
        example_probs = self.example_identifier(flat_features)  # [B, num_examples]

        # Process through example-specific networks
        outputs = []
        for i, network in enumerate(self.example_networks):
            example_output = network(attended_features)
            outputs.append(
                example_output * example_probs[:, i : i + 1].unsqueeze(-1).unsqueeze(-1)
            )

        # Combine outputs
        combined = sum(outputs)

        # Apply color-specific heads
        logits_list = []
        for color_head in self.color_heads:
            color_logit = color_head(combined)
            logits_list.append(color_logit)

        logits = torch.cat(logits_list, dim=-1)  # [B, 30, 30, num_classes]

        # Apply temperature scaling based on example
        avg_temperature = (
            example_probs.unsqueeze(-1) * self.example_temperatures.unsqueeze(0)
        ).sum(dim=1)
        logits = logits / avg_temperature.view(-1, 1, 1, 1)

        # Apply color constraints
        logits = self.apply_color_constraints(logits)

        return logits

    def apply_color_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hard constraints to eliminate unavailable colors."""
        available_colors: torch.Tensor = self.available_colors  # type: ignore
        unavailable_mask = 1 - available_colors

        # Set logits for unavailable colors to very negative value
        batch_size = logits.shape[0]
        unavailable_mask_expanded = (
            unavailable_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        )
        unavailable_mask_expanded = unavailable_mask_expanded.expand(
            batch_size, 30, 30, -1
        )

        constrained_logits = logits.clone()
        constrained_logits[unavailable_mask_expanded.bool()] = -1e9

        return constrained_logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss with special handling for rare colors."""
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.num_classes)
        targets_flat = targets.reshape(-1)

        # Mask for valid pixels
        valid_mask = targets_flat != -1

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute class weights based on frequency
        unique_colors, counts = torch.unique(
            targets_flat[valid_mask], return_counts=True
        )
        class_weights = torch.ones(self.num_classes, device=logits.device)

        # Increase weight for rare colors
        total_pixels = counts.sum()
        for color, count in zip(unique_colors, counts):
            # Inverse frequency weighting with smoothing
            weight = (total_pixels / (count + 1.0)).sqrt()
            class_weights[color] = weight

        # Compute weighted loss
        loss = F.cross_entropy(
            logits_flat[valid_mask],
            targets_flat[valid_mask],
            weight=class_weights,
            label_smoothing=0.01,
        )

        return loss

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

        # Process inputs
        input_logits = self(input_features, is_input=True)
        input_loss = self.compute_loss(input_logits, input_colors)

        # Process outputs
        output_logits = self(output_features, is_input=False)
        output_loss = self.compute_loss(output_logits, output_colors)

        # Combined loss
        total_loss = input_loss + output_loss

        # Calculate accuracy
        with torch.no_grad():
            all_preds = torch.cat(
                [
                    torch.argmax(input_logits, dim=-1).flatten(),
                    torch.argmax(output_logits, dim=-1).flatten(),
                ]
            )
            all_targets = torch.cat([input_colors.flatten(), output_colors.flatten()])

            valid_mask = all_targets != -1
            if valid_mask.any():
                accuracy = (
                    (all_preds[valid_mask] == all_targets[valid_mask]).float().mean()
                )
            else:
                accuracy = torch.tensor(0.0)

        # Visualize every 10 epochs
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            self.visualize_predictions(
                input_features[0:1], input_colors[0:1], input_logits[0:1], "input"
            )
            self.visualize_predictions(
                output_features[0:1], output_colors[0:1], output_logits[0:1], "output"
            )

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        (
            input_features,
            output_features,
            input_colors,
            output_colors,
            _,
            _,
        ) = batch

        # Process inputs
        input_logits = self(input_features, is_input=True)
        input_loss = self.compute_loss(input_logits, input_colors)

        # Process outputs
        output_logits = self(output_features, is_input=False)
        output_loss = self.compute_loss(output_logits, output_colors)

        # Combined loss
        total_loss = input_loss + output_loss

        # Calculate accuracy
        all_preds = torch.cat(
            [
                torch.argmax(input_logits, dim=-1).flatten(),
                torch.argmax(output_logits, dim=-1).flatten(),
            ]
        )
        all_targets = torch.cat([input_colors.flatten(), output_colors.flatten()])

        valid_mask = all_targets != -1
        if valid_mask.any():
            accuracy = (all_preds[valid_mask] == all_targets[valid_mask]).float().mean()
        else:
            accuracy = torch.tensor(0.0)

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Use AdamW with different learning rates for different parts
        param_groups = [
            {"params": self.example_networks.parameters(), "lr": self.lr},
            {"params": self.global_feature_extractor.parameters(), "lr": self.lr * 0.5},
            {"params": self.spatial_attention.parameters(), "lr": self.lr * 0.5},
            {"params": self.color_heads.parameters(), "lr": self.lr * 2.0},
            {
                "params": [self.pos_embedding, self.example_temperatures],
                "lr": self.lr * 0.1,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def visualize_predictions(
        self,
        features: torch.Tensor,
        ground_truth: torch.Tensor,
        logits: torch.Tensor,
        prefix: str,
    ) -> None:
        """Visualize predictions for debugging."""
        from arcagi.utils.terminal_imshow import imshow

        predictions = torch.argmax(logits, dim=-1)

        print(f"\n{'='*60}")
        print(f"{prefix.upper()} - Epoch {self.current_epoch}")
        print(f"{'='*60}")

        print(f"\nGround Truth {prefix} colors:")
        imshow(ground_truth.cpu())

        print(f"\nPredicted {prefix} colors:")
        imshow(predictions.cpu())

        # Calculate accuracy for this example
        valid_mask = ground_truth != -1
        if valid_mask.any():
            correct = (
                (predictions[valid_mask] == ground_truth[valid_mask]).float().mean()
            )
            print(f"\nAccuracy: {correct.item():.2%}")

        # Show per-color accuracy
        unique_colors = torch.unique(ground_truth[valid_mask])
        print(f"\nPer-color accuracy:")
        for color in unique_colors:
            color_mask = ground_truth == color
            if color_mask.any():
                color_acc = (predictions[color_mask] == color).float().mean()
                color_count = color_mask.sum().item()
                print(
                    f"  Color {color.item()}: {color_acc.item():.2%} ({color_count} pixels)"
                )


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


def load_single_filename_data(
    filename: str,
    train_path: str = "processed_data/train_all.npz",
    eval_path: str = "processed_data/eval_all.npz",
) -> Tuple[Dict[str, np.ndarray], List[int], int]:
    """Load data for a single filename from both train and eval sets."""
    # Load data
    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    # Find indices for target filename
    train_indices = np.where(train_data["filenames"] == filename)[0]
    eval_indices = np.where(eval_data["filenames"] == filename)[0]

    print(
        f"Found {len(train_indices)} train examples and {len(eval_indices)} eval examples for {filename}"
    )

    # Combine data
    all_inputs = []
    all_outputs = []
    all_input_features = []
    all_output_features = []

    # Add train data
    for idx in train_indices:
        all_inputs.append(train_data["inputs"][idx])
        all_outputs.append(train_data["outputs"][idx])
        all_input_features.append(train_data["inputs_features"][idx])
        all_output_features.append(train_data["outputs_features"][idx])

    # Add eval data
    for idx in eval_indices:
        all_inputs.append(eval_data["inputs"][idx])
        all_outputs.append(eval_data["outputs"][idx])
        all_input_features.append(eval_data["inputs_features"][idx])
        all_output_features.append(eval_data["outputs_features"][idx])

    # Get available colors for this filename
    filename_idx = np.where(train_data["filename_colors_keys"] == filename)[0]
    if len(filename_idx) > 0:
        color_mask = train_data["filename_colors_values"][filename_idx[0]]
        # Convert binary mask to list of available color indices
        available_colors = [i for i, available in enumerate(color_mask) if available == 1]
    else:
        # If not in train, check eval
        filename_idx = np.where(eval_data["filename_colors_keys"] == filename)[0]
        if len(filename_idx) > 0:
            color_mask = eval_data["filename_colors_values"][filename_idx[0]]
            # Convert binary mask to list of available color indices
            available_colors = [i for i, available in enumerate(color_mask) if available == 1]
        else:
            # Default to all colors
            available_colors = list(range(10))

    print(f"Available colors for {filename}: {available_colors}")

    # Count total examples
    num_examples = len(all_inputs)

    return (
        {
            "inputs": np.array(all_inputs),
            "outputs": np.array(all_outputs),
            "inputs_features": np.array(all_input_features),
            "outputs_features": np.array(all_output_features),
        },
        available_colors,
        num_examples,
    )


def create_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    # Convert to tensors
    inputs_tensor = torch.from_numpy(data["inputs"]).long()
    outputs_tensor = torch.from_numpy(data["outputs"]).long()
    inputs_features_tensor = torch.from_numpy(data["inputs_features"]).float()
    outputs_features_tensor = torch.from_numpy(data["outputs_features"]).float()

    # Create dummy filename indices (all same filename)
    num_examples = len(inputs_tensor)
    filename_indices = torch.zeros(num_examples, dtype=torch.long)

    # Create dataset
    dataset = TensorDataset(
        inputs_features_tensor,
        outputs_features_tensor,
        inputs_tensor,
        outputs_tensor,
        filename_indices,
        filename_indices,  # Dummy for compatibility
    )

    # Use all data for both train and val (we want to memorize)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train perfect memorization model on single filename"
    )
    parser.add_argument(
        "--target_filename",
        type=str,
        default="3345333e.json",
        help="Target filename to memorize",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500, help="Maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for scaling"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="color_mapping_perfect_memorization",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    # Load data for single filename
    data, available_colors, num_examples = load_single_filename_data(
        args.target_filename
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    model = PerfectMemorizationModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=10,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        available_colors=available_colors,
        num_examples=num_examples,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/checkpoints",
        filename="perfect_memorization-{epoch:02d}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,  # Gradient accumulation for stability
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed for {args.target_filename}")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
