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


class ColorConstrainedMappingModel(pl.LightningModule):
    """
    PyTorch Lightning model that predicts input colors from features with filename-based color constraints.

    Key features:
    1. Uses filename color masks to constrain predictions to valid colors only
    2. Embeds the color mask as additional features
    3. Applies soft constraints during training and hard constraints during inference
    4. Includes a constraint violation loss to encourage learning valid predictions

    Args:
        feature_dim: Number of input features (147)
        pos_embed_dim: Dimension of position embeddings for x and y coordinates
        filename_embed_dim: Dimension of filename-specific latent embeddings
        color_mask_embed_dim: Dimension for embedding the 10-dim color mask
        hidden_dim: Hidden dimension for the MLP
        num_classes: Number of color classes (10: colors 0-9)
        num_filenames: Number of unique filenames
        constraint_weight: Weight for the constraint violation loss
        soft_constraint_value: Value to subtract from invalid color logits during training
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    """

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 32,
        color_mask_embed_dim: int = 16,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_filenames: int = 400,
        constraint_weight: float = 0.1,
        soft_constraint_value: float = 10.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.pos_embed_dim = pos_embed_dim
        self.filename_embed_dim = filename_embed_dim
        self.color_mask_embed_dim = color_mask_embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_filenames = num_filenames
        self.constraint_weight = constraint_weight
        self.soft_constraint_value = soft_constraint_value
        self.lr = lr
        self.weight_decay = weight_decay

        # Sinusoidal positional embeddings for x and y coordinates (0-29)
        self.register_buffer(
            "x_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )
        self.register_buffer(
            "y_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )

        # Filename-specific latent variables for memorization
        self.filename_embedding = nn.Embedding(num_filenames, filename_embed_dim)

        # Embedding for the 10-dimensional color mask
        self.color_mask_mlp = nn.Sequential(
            nn.Linear(10, color_mask_embed_dim),
            nn.ReLU(),
            nn.Linear(color_mask_embed_dim, color_mask_embed_dim),
        )

        # Input dimension: features + 2 * pos_embed_dim + filename_embed_dim + color_mask_embed_dim
        input_dim = (
            feature_dim + 2 * pos_embed_dim + filename_embed_dim + color_mask_embed_dim
        )

        # Deeper MLP for color prediction
        self.mlp = nn.Sequential(
            # First layer: expand features
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Second layer: process features
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Third layer: refine features
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Fourth layer: compress
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Output layer
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Loss function with class weights to handle imbalance
        class_weights = torch.tensor(
            [
                1.0,  # color 0 (most common)
                8.6,  # color 1
                11.2,  # color 2
                12.1,  # color 3
                16.5,  # color 4
                12.6,  # color 5
                29.7,  # color 6
                47.2,  # color 7
                9.1,  # color 8
                55.2,  # color 9 (rarest)
            ]
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

    def _create_sinusoidal_embeddings(
        self, max_len: int, embed_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal positional embeddings."""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def apply_color_constraints(
        self, logits: torch.Tensor, color_masks: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """
        Apply color constraints to logits based on valid colors for each filename.

        Args:
            logits: Raw logits of shape (batch_size, 30, 30, num_classes)
            color_masks: Binary masks of shape (batch_size, num_classes) indicating valid colors
            training: Whether in training mode (use soft constraints) or inference (use hard constraints)

        Returns:
            Constrained logits of the same shape
        """
        batch_size = logits.shape[0]

        # Expand color masks to match spatial dimensions
        color_masks_expanded = (
            color_masks.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, 30, 30, self.num_classes)
        )  # (batch_size, 30, 30, num_classes)

        # Create constraint mask (1 for invalid colors, 0 for valid colors)
        invalid_mask = 1 - color_masks_expanded

        if training:
            # Soft constraints: subtract a large value from invalid colors
            constraint_penalty = invalid_mask * self.soft_constraint_value
            constrained_logits = logits - constraint_penalty
        else:
            # Hard constraints: set invalid colors to very large negative value
            # Using -1e10 instead of -inf to avoid numerical issues on some devices
            constrained_logits = logits.clone()
            constrained_logits[invalid_mask.bool()] = -1e10

        return constrained_logits

    def compute_constraint_loss(
        self, logits: torch.Tensor, color_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a loss that penalizes predictions of invalid colors.

        Args:
            logits: Raw logits of shape (batch_size, 30, 30, num_classes)
            color_masks: Binary masks of shape (batch_size, num_classes) indicating valid colors

        Returns:
            Constraint violation loss (scalar)
        """
        batch_size = logits.shape[0]

        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # (batch_size, 30, 30, num_classes)

        # Expand color masks
        color_masks_expanded = (
            color_masks.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, 30, 30, self.num_classes)
        )

        # Invalid mask (1 for invalid colors, 0 for valid)
        invalid_mask = 1 - color_masks_expanded

        # Compute probability mass on invalid colors
        invalid_probs = probs * invalid_mask
        constraint_loss = invalid_probs.sum(
            dim=-1
        ).mean()  # Average over all positions and batch

        return constraint_loss

    def forward(
        self,
        features: torch.Tensor,
        filename_ids: torch.Tensor,
        color_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            features: Input features of shape (batch_size, 30, 30, feature_dim)
            filename_ids: Filename IDs of shape (batch_size,)
            color_masks: Binary color masks of shape (batch_size, num_classes)

        Returns:
            logits: Output logits of shape (batch_size, 30, 30, num_classes)
        """
        batch_size, height, width, _ = features.shape

        # Create position grids
        x_pos = (
            torch.arange(width, device=features.device).unsqueeze(0).expand(height, -1)
        )
        y_pos = (
            torch.arange(height, device=features.device).unsqueeze(1).expand(-1, width)
        )

        # Get sinusoidal position embeddings
        x_embeds = self.x_pos_embed[x_pos].unsqueeze(0).expand(batch_size, -1, -1, -1)
        y_embeds = self.y_pos_embed[y_pos].unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Get filename embeddings
        filename_embeds = self.filename_embedding(filename_ids)
        filename_embeds = (
            filename_embeds.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, -1)
        )

        # Embed color masks
        color_mask_embeds = self.color_mask_mlp(
            color_masks.float()
        )  # (batch_size, color_mask_embed_dim)
        color_mask_embeds = (
            color_mask_embeds.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, -1)
        )

        # Concatenate all embeddings with features
        combined_features = torch.cat(
            [features, x_embeds, y_embeds, filename_embeds, color_mask_embeds], dim=-1
        )

        # Apply MLP
        logits = self.mlp(combined_features)

        return logits

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        features, colors, filename_ids, color_masks = batch

        # Forward pass
        logits = self(features, filename_ids, color_masks)

        # Apply soft constraints
        constrained_logits = self.apply_color_constraints(
            logits, color_masks, training=True
        )

        # Reshape for loss computation
        logits_flat = constrained_logits.view(-1, self.num_classes)
        colors_flat = colors.view(-1)

        # Compute main classification loss
        cls_loss = self.criterion(logits_flat, colors_flat)

        # Compute constraint violation loss
        constraint_loss = self.compute_constraint_loss(logits, color_masks)

        # Total loss
        loss = cls_loss + self.constraint_weight * constraint_loss

        # Compute accuracy (ignoring mask values)
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=1)
            mask = colors_flat != -1
            if mask.sum() > 0:
                acc = (preds[mask] == colors_flat[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log("train_constraint_loss", constraint_loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        features, colors, filename_ids, color_masks = batch

        # Forward pass
        logits = self(features, filename_ids, color_masks)

        # Apply hard constraints for validation
        constrained_logits = self.apply_color_constraints(
            logits, color_masks, training=False
        )

        # Debug visualization for first batch (disabled after debugging)
        if False and batch_idx == 0 and self.current_epoch == 0:
            # Import visualization tool
            import os
            import sys

            sys.path.append(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
            from arcagi.utils.terminal_imshow import imshow

            # Take first example in batch
            idx = 0

            # Get predictions before and after constraints
            preds_before = torch.argmax(logits[idx], dim=-1)
            preds_after = torch.argmax(constrained_logits[idx], dim=-1)
            ground_truth = colors[idx]

            print("\n" + "=" * 60)
            print("DEBUG: Color Constraint Visualization (First Validation Example)")
            print("=" * 60)

            # Show which colors are valid for this filename
            valid_colors = color_masks[idx].nonzero(as_tuple=True)[0].tolist()
            print(f"Valid colors for this filename: {valid_colors}")
            print(f"Filename ID: {filename_ids[idx].item()}")
            print(f"Color mask: {color_masks[idx].tolist()}")

            # Debug the constraint application
            print("\nDEBUG: Checking constraint application...")
            test_logits = logits[idx : idx + 1].clone()  # Shape: (1, 30, 30, 10)
            test_mask = color_masks[idx : idx + 1]  # Shape: (1, 10)

            # Manually apply constraints
            invalid_mask = 1 - test_mask
            print(f"Invalid mask: {invalid_mask[0].tolist()}")

            # Check a specific pixel
            pixel_logits = test_logits[0, 15, 15]
            print(f"\nPixel (15,15) logits before constraints: {pixel_logits.tolist()}")

            # Apply hard constraints manually
            constrained_pixel = pixel_logits.clone()
            for c in range(10):
                if invalid_mask[0, c] == 1:
                    constrained_pixel[c] = float("-inf")
            print(
                f"Pixel (15,15) logits after manual constraints: {constrained_pixel.tolist()}"
            )
            print(
                f"Argmax after manual constraints: {torch.argmax(constrained_pixel).item()}"
            )

            # Show ground truth
            print("\nGround Truth:")
            imshow(ground_truth, show_legend=False)

            # Show predictions before constraints
            print("\nPredictions BEFORE color constraints:")
            imshow(preds_before, show_legend=False)

            # Show predictions after constraints
            print("\nPredictions AFTER color constraints:")
            imshow(preds_after, show_legend=False)

            # Check for constraint violations
            violations_before = 0
            violations_after = 0
            non_mask_pixels = (ground_truth != -1).sum().item()

            for i in range(30):
                for j in range(30):
                    if ground_truth[i, j] != -1:
                        if preds_before[i, j].item() not in valid_colors:
                            violations_before += 1
                        if preds_after[i, j].item() not in valid_colors:
                            violations_after += 1

            print(
                f"\nConstraint violations BEFORE: {violations_before}/{non_mask_pixels} ({100*violations_before/non_mask_pixels:.1f}%)"
            )
            print(
                f"Constraint violations AFTER: {violations_after}/{non_mask_pixels} ({100*violations_after/non_mask_pixels:.1f}%)"
            )

            # Show raw logits for a specific pixel
            center_i, center_j = 15, 15
            if ground_truth[center_i, center_j] != -1:
                print(f"\nLogits at pixel ({center_i}, {center_j}):")
                print(f"Ground truth color: {ground_truth[center_i, center_j].item()}")
                raw_logits = logits[idx, center_i, center_j]
                constrained = constrained_logits[idx, center_i, center_j]

                for c in range(10):
                    is_valid = c in valid_colors
                    print(
                        f"  Color {c}: raw={raw_logits[c].item():.2f}, constrained={constrained[c].item():.2f} {'(valid)' if is_valid else '(INVALID)'}"
                    )

            print("=" * 60 + "\n")

        # Reshape for loss computation
        logits_flat = constrained_logits.view(-1, self.num_classes)
        colors_flat = colors.view(-1)

        # Compute main classification loss
        cls_loss = self.criterion(logits_flat, colors_flat)

        # Compute constraint violation loss (should be near 0 with hard constraints)
        constraint_loss = self.compute_constraint_loss(logits, color_masks)

        # Total loss
        loss = cls_loss + self.constraint_weight * constraint_loss

        # Compute accuracy (ignoring mask values)
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=1)
            mask = colors_flat != -1
            if mask.sum() > 0:
                acc = (preds[mask] == colors_flat[mask]).float().mean()

                # Vectorized constraint violation check
                # Create a mask of valid predictions
                batch_size = features.shape[0]

                # Expand color masks to match prediction shape
                valid_mask = torch.zeros(
                    batch_size, self.num_classes, device=self.device
                )
                for i in range(batch_size):
                    valid_mask[i] = color_masks[i]

                # Get validity of each prediction
                preds_reshaped = preds.view(batch_size, -1)  # (batch_size, 900)
                colors_reshaped = colors.view(batch_size, -1)  # (batch_size, 900)

                # For each prediction, check if it's valid
                batch_indices = (
                    torch.arange(batch_size, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, 900)
                )
                pred_validity = valid_mask[
                    batch_indices.flatten(), preds_reshaped.flatten()
                ].view(batch_size, -1)

                # Only count non-mask pixels
                non_mask = colors_reshaped != -1
                violations = ((~pred_validity.bool()) & non_mask).sum().item()
                total_preds = non_mask.sum().item()

                constraint_violation_rate = (
                    violations / total_preds if total_preds > 0 else 0.0
                )
                self.log(
                    "val_constraint_violations",
                    constraint_violation_rate,
                    on_step=False,
                    on_epoch=True,
                )
            else:
                acc = torch.tensor(0.0, device=self.device)

        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log("val_constraint_loss", constraint_loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def prepare_color_constrained_data(
    train_npz_path: str,
    eval_npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_train_as_eval: bool = False,
) -> Tuple[DataLoader, DataLoader, int, Dict[str, torch.Tensor]]:
    """
    Prepare data loaders for color-constrained mapping task.

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_filenames: Number of unique filenames
        filename_to_color_mask: Dict mapping filename to color mask tensor
    """
    # Load training data
    print("Loading training data...")
    train_data = np.load(train_npz_path)

    # Extract data
    train_filenames = train_data["filenames"].tolist()
    train_inputs = torch.from_numpy(train_data["inputs"])
    train_outputs = torch.from_numpy(train_data["outputs"])
    train_features = torch.from_numpy(train_data["inputs_features"]).float()

    # Load filename color mappings
    filename_colors_keys = train_data["filename_colors_keys"]
    filename_colors_values = train_data["filename_colors_values"]

    # Create mapping from filename to color mask
    filename_to_color_mask = {}
    for fname, color_mask in zip(filename_colors_keys, filename_colors_values):
        filename_to_color_mask[fname] = torch.from_numpy(color_mask).float()

    print(f"Loaded color masks for {len(filename_to_color_mask)} unique filenames")

    if use_train_as_eval:
        print(
            "Using training data for both training and evaluation (overfitting test)..."
        )
        val_filenames = train_filenames
        val_inputs = train_inputs
        val_features = train_features
    else:
        # Load validation data
        print("Loading validation data...")
        val_data = np.load(eval_npz_path)
        val_filenames = val_data["filenames"].tolist()
        val_inputs = torch.from_numpy(val_data["inputs"])
        val_outputs = torch.from_numpy(val_data["outputs"])
        val_features = torch.from_numpy(val_data["inputs_features"]).float()

        # Update color mask mapping with validation filenames if needed
        val_filename_colors_keys = val_data["filename_colors_keys"]
        val_filename_colors_values = val_data["filename_colors_values"]

        for fname, color_mask in zip(
            val_filename_colors_keys, val_filename_colors_values
        ):
            if fname not in filename_to_color_mask:
                filename_to_color_mask[fname] = torch.from_numpy(color_mask).float()

    # The inputs are already in the correct format (-1 for mask, 0-9 for colors)
    train_colors = train_inputs.long()
    val_colors = val_inputs.long()

    # Create filename to ID mapping
    all_filenames = list(set(train_filenames + val_filenames))
    filename_to_id = {fname: i for i, fname in enumerate(sorted(all_filenames))}
    num_filenames = len(all_filenames)

    print(f"Found {num_filenames} unique filenames")
    print(f"Training examples: {len(train_filenames)}")
    print(f"Validation examples: {len(val_filenames)}")

    # Convert filenames to IDs and get color masks for each example
    train_filename_ids = []
    train_color_masks = []
    for fname in train_filenames:
        train_filename_ids.append(filename_to_id[fname])
        train_color_masks.append(filename_to_color_mask[fname])

    train_filename_ids = torch.tensor(train_filename_ids, dtype=torch.long)
    train_color_masks = torch.stack(train_color_masks)

    val_filename_ids = []
    val_color_masks = []
    for fname in val_filenames:
        val_filename_ids.append(filename_to_id[fname])
        val_color_masks.append(filename_to_color_mask[fname])

    val_filename_ids = torch.tensor(val_filename_ids, dtype=torch.long)
    val_color_masks = torch.stack(val_color_masks)

    # Convert to proper data types
    train_colors = train_colors.long()
    val_colors = val_colors.long()

    print(f"Training data shapes:")
    print(f"  features: {train_features.shape}")
    print(f"  colors: {train_colors.shape}")
    print(f"  filename_ids: {train_filename_ids.shape}")
    print(f"  color_masks: {train_color_masks.shape}")

    # Create datasets
    train_dataset = TensorDataset(
        train_features, train_colors, train_filename_ids, train_color_masks
    )
    val_dataset = TensorDataset(
        val_features, val_colors, val_filename_ids, val_color_masks
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, num_filenames, filename_to_color_mask


def main():
    parser = argparse.ArgumentParser(
        description="Train color-constrained mapping model"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="processed_data/train_all.npz",
        help="Path to training NPZ file",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="processed_data/eval_all.npz",
        help="Path to evaluation NPZ file",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--pos_embed_dim", type=int, default=16, help="Position embedding dimension"
    )
    parser.add_argument(
        "--filename_embed_dim",
        type=int,
        default=32,
        help="Filename embedding dimension",
    )
    parser.add_argument(
        "--color_mask_embed_dim",
        type=int,
        default=16,
        help="Color mask embedding dimension",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--constraint_weight",
        type=float,
        default=0.1,
        help="Weight for constraint violation loss",
    )
    parser.add_argument(
        "--soft_constraint_value",
        type=float,
        default=10.0,
        help="Value to subtract from invalid color logits during training",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="color_mapping_outputs_v6",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--use_train_as_eval",
        action="store_true",
        help="Use training data for both training and evaluation (overfitting test)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data loaders
    train_loader, val_loader, num_filenames, filename_to_color_mask = (
        prepare_color_constrained_data(
            args.train_data,
            args.eval_data,
            args.batch_size,
            args.num_workers,
            use_train_as_eval=args.use_train_as_eval,
        )
    )

    # Initialize model
    model = ColorConstrainedMappingModel(
        feature_dim=147,
        pos_embed_dim=args.pos_embed_dim,
        filename_embed_dim=args.filename_embed_dim,
        color_mask_embed_dim=args.color_mask_embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        num_filenames=num_filenames,
        constraint_weight=args.constraint_weight,
        soft_constraint_value=args.soft_constraint_value,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="color_constrained-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        verbose=True,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Train model
    print("\nStarting training with color constraints...")
    print(f"Constraint weight: {args.constraint_weight}")
    print(f"Soft constraint value: {args.soft_constraint_value}")

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
