import argparse
import os

# Import from parent module
import sys
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcagi.data_loader import load_npz_data, one_hot_to_categorical


class UNetBlock(nn.Module):
    """Basic U-Net block with conv, norm, activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleUNetColorMappingModel(pl.LightningModule):
    """
    Simplified U-Net based model that predicts input colors from features.
    Uses a more stable architecture to avoid tensor contiguity issues.

    Args:
        feature_dim: Number of input features (147)
        pos_embed_dim: Dimension of position embeddings for x and y coordinates
        filename_embed_dim: Dimension of filename-specific latent embeddings
        base_channels: Base number of channels for U-Net
        num_classes: Number of color classes (10: colors 0-9)
        num_filenames: Number of unique filenames
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        dropout: Dropout rate
    """

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 32,
        base_channels: int = 64,
        num_classes: int = 10,
        num_filenames: int = 400,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.pos_embed_dim = pos_embed_dim
        self.filename_embed_dim = filename_embed_dim
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.num_filenames = num_filenames
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        # Sinusoidal positional embeddings for x and y coordinates (0-29)
        self.register_buffer(
            "x_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )
        self.register_buffer(
            "y_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )

        # Filename-specific latent variables for memorization
        self.filename_embedding = nn.Embedding(num_filenames, filename_embed_dim)

        # Input dimension calculation
        input_dim = feature_dim + 2 * pos_embed_dim + filename_embed_dim

        # Initial feature processing - convert to spatial format first
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_dim, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Simplified U-Net Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Simplified U-Net Decoder
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(
                base_channels * 4, base_channels * 2, kernel_size=3, padding=1
            ),  # After concatenation
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(
                base_channels * 2, base_channels, kernel_size=3, padding=1
            ),  # After concatenation
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # Dropout layer
        self.dropout2d = nn.Dropout2d(dropout)

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

    def forward(
        self, features: torch.Tensor, filename_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the simplified U-Net model.

        Args:
            features: Input features of shape (batch_size, 30, 30, feature_dim)
            filename_ids: Filename IDs of shape (batch_size,)

        Returns:
            logits: Output logits of shape (batch_size, 30, 30, num_classes)
        """
        batch_size, height, width, _ = features.shape

        # Create position grids
        x_pos = (
            torch.arange(width, device=features.device).unsqueeze(0).expand(height, -1)
        )  # (30, 30)
        y_pos = (
            torch.arange(height, device=features.device).unsqueeze(1).expand(-1, width)
        )  # (30, 30)

        # Get sinusoidal position embeddings
        x_embeds = self.x_pos_embed[x_pos]  # (30, 30, pos_embed_dim)
        y_embeds = self.y_pos_embed[y_pos]  # (30, 30, pos_embed_dim)

        # Expand to batch size
        x_embeds = x_embeds.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch_size, 30, 30, pos_embed_dim)
        y_embeds = y_embeds.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch_size, 30, 30, pos_embed_dim)

        # Get filename embeddings
        filename_embeds = self.filename_embedding(
            filename_ids
        )  # (batch_size, filename_embed_dim)
        # Expand to all spatial positions
        filename_embeds = (
            filename_embeds.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, -1)
        )  # (batch_size, 30, 30, filename_embed_dim)

        # Concatenate all embeddings with features
        combined_features = torch.cat(
            [features, x_embeds, y_embeds, filename_embeds], dim=-1
        )  # (batch_size, 30, 30, total_dim)

        # Convert to conv format: (batch_size, channels, height, width)
        # Use permute and make contiguous to avoid view issues
        x = combined_features.permute(0, 3, 1, 2).contiguous()

        # Initial convolution
        x = self.input_conv(x)  # (batch_size, base_channels, 30, 30)

        # Encoder
        enc1 = self.enc1(x)  # (batch_size, base_channels, 30, 30)
        x = self.down1(enc1)  # (batch_size, base_channels*2, 15, 15)

        enc2 = self.enc2(x)  # (batch_size, base_channels*2, 15, 15)
        x = self.down2(enc2)  # (batch_size, base_channels*4, 8, 8)

        # Bottleneck
        x = self.bottleneck(x)  # (batch_size, base_channels*4, 8, 8)
        x = self.dropout2d(x)

        # Decoder
        x = self.up2(x)  # (batch_size, base_channels*2, 16, 16)
        # Crop x to match enc2 size (15x15) - remove one pixel from right and bottom
        x = x[:, :, :15, :15]  # Crop to 15x15
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.dec2(x)  # (batch_size, base_channels*2, 15, 15)

        x = self.up1(x)  # (batch_size, base_channels, 30, 30)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.dec1(x)  # (batch_size, base_channels, 30, 30)

        # Output projection
        logits = self.output_conv(x)  # (batch_size, num_classes, 30, 30)

        # Convert back to (batch_size, 30, 30, num_classes) format
        # Use permute and make contiguous
        logits = logits.permute(0, 2, 3, 1).contiguous()

        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors, filename_ids = batch

        # Forward pass
        logits = self(features, filename_ids)  # (batch_size, 30, 30, num_classes)

        # Reshape for loss computation
        logits_flat = logits.reshape(
            -1, self.num_classes
        )  # (batch_size * 30 * 30, num_classes)
        colors_flat = colors.reshape(-1)  # (batch_size * 30 * 30,)

        # Compute loss
        loss = self.criterion(logits_flat, colors_flat)

        # Compute accuracy (ignoring mask values)
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=1)
            mask = colors_flat != -1  # Don't count mask values
            if mask.sum() > 0:
                acc = (preds[mask] == colors_flat[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors, filename_ids = batch

        # Forward pass
        logits = self(features, filename_ids)  # (batch_size, 30, 30, num_classes)

        # Reshape for loss computation
        logits_flat = logits.reshape(
            -1, self.num_classes
        )  # (batch_size * 30 * 30, num_classes)
        colors_flat = colors.reshape(-1)  # (batch_size * 30 * 30,)

        # Compute loss
        loss = self.criterion(logits_flat, colors_flat)

        # Compute accuracy (ignoring mask values)
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=1)
            mask = colors_flat != -1  # Don't count mask values
            if mask.sum() > 0:
                acc = (preds[mask] == colors_flat[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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


# Reuse data preparation functions from ex02.py
def create_filename_mapping(filenames: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """Create mapping from filename to integer ID."""
    unique_filenames = sorted(list(set(filenames)))
    filename_to_id = {fname: i for i, fname in enumerate(unique_filenames)}
    return filename_to_id, unique_filenames


def prepare_unet_data(
    train_npz_path: str,
    eval_npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_train_as_eval: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare data loaders for U-Net color mapping task.

    Args:
        train_npz_path: Path to training NPZ file with features
        eval_npz_path: Path to evaluation NPZ file with features
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        use_train_as_eval: If True, use training data for both train and eval (for overfitting test)

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader (same as train if use_train_as_eval=True)
        num_filenames: Number of unique filenames
    """
    # Load training data
    print("Loading training data...")
    train_filenames, train_indices, train_inputs, train_outputs, train_features, _ = (
        load_npz_data(train_npz_path, use_features=True)
    )

    if use_train_as_eval:
        print(
            "Using training data for both training and evaluation (overfitting test)..."
        )
        # Use training data for both train and eval
        val_filenames = train_filenames
        val_inputs = train_inputs
        val_features = train_features
    else:
        # Load validation data normally
        print("Loading validation data...")
        val_filenames, val_indices, val_inputs, val_outputs, val_features, _ = (
            load_npz_data(eval_npz_path, use_features=True)
        )

    # Convert one-hot encoded inputs back to categorical colors
    train_colors = one_hot_to_categorical(train_inputs, last_value=10)
    val_colors = one_hot_to_categorical(val_inputs, last_value=10)

    # Convert 10 back to -1 for mask
    train_colors[train_colors == 10] = -1
    val_colors[val_colors == 10] = -1

    # Check that features were loaded
    if train_features is None or val_features is None:
        raise ValueError(
            "Features not found in NPZ files. Make sure to use files with features."
        )

    # Create filename mappings
    all_filenames = train_filenames + val_filenames
    filename_to_id, unique_filenames = create_filename_mapping(all_filenames)
    num_filenames = len(unique_filenames)

    print(f"Found {num_filenames} unique filenames")
    print(f"Training examples: {len(train_filenames)}")
    print(f"Validation examples: {len(val_filenames)}")

    if use_train_as_eval:
        print("Note: Validation set is identical to training set for overfitting test")

    # Convert filenames to IDs
    train_filename_ids = torch.tensor(
        [filename_to_id[fname] for fname in train_filenames], dtype=torch.long
    )
    val_filename_ids = torch.tensor(
        [filename_to_id[fname] for fname in val_filenames], dtype=torch.long
    )

    # Convert to proper data types
    train_features = train_features.float()
    val_features = val_features.float()
    train_colors = train_colors.long()
    val_colors = val_colors.long()

    print(
        f"Training data: features={train_features.shape}, colors={train_colors.shape}, filename_ids={train_filename_ids.shape}"
    )
    print(
        f"Validation data: features={val_features.shape}, colors={val_colors.shape}, filename_ids={val_filename_ids.shape}"
    )

    # Create datasets with filename IDs
    train_dataset = TensorDataset(train_features, train_colors, train_filename_ids)
    val_dataset = TensorDataset(val_features, val_colors, val_filename_ids)

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
        shuffle=False,  # Don't shuffle validation set
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, num_filenames


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net based color mapping model"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (smaller due to U-Net memory)",
    )
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
        "--base_channels", type=int, default=64, help="Base channels for U-Net"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="color_mapping_outputs_unet",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--use_train_as_eval",
        action="store_true",
        help="Use training data for both training and evaluation (overfitting test)",
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="Save model checkpoint after every epoch (default: save top 3 only)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data loaders
    train_loader, val_loader, num_filenames = prepare_unet_data(
        args.train_data,
        args.eval_data,
        args.batch_size,
        args.num_workers,
        use_train_as_eval=args.use_train_as_eval,
    )

    # Initialize model
    model = SimpleUNetColorMappingModel(
        feature_dim=147,
        pos_embed_dim=args.pos_embed_dim,
        filename_embed_dim=args.filename_embed_dim,
        base_channels=args.base_channels,
        num_classes=10,
        num_filenames=num_filenames,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="unet_color_mapping-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        save_top_k=-1 if args.save_every_epoch else 3,  # Save all or top 3
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=args.save_every_epoch,  # Save after every training epoch if requested
        verbose=(
            True if args.save_every_epoch else False
        ),  # Print when saving if saving every epoch
    )

    # Adjust early stopping for overfitting test
    if args.use_train_as_eval:
        # When overfitting, we expect validation loss to match training loss
        # Use more patience since we want to see if it can memorize perfectly
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=20,  # More patience for overfitting
            mode="min",
            verbose=True,
        )
        print("Note: Using extended patience for overfitting test")
    else:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        )

    # Setup logger
    logger = None  # Use default logger

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=1.0,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Train model
    print("Starting training...")
    if args.use_train_as_eval:
        print("OVERFITTING TEST: Training and validation sets are identical")
        print("Expecting validation accuracy to reach ~100% if model can memorize")

    trainer.fit(model, train_loader, val_loader)

    print(f"Training completed! Outputs saved to: {args.output_dir}")

    if args.use_train_as_eval:
        print("\nOVERFITTING TEST RESULTS:")
        print("Check if validation accuracy reached close to 100%")
        print("This tests the model's capacity to memorize color mappings")


if __name__ == "__main__":
    main()
