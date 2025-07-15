import argparse
import os

# Import from parent module
import sys
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcagi.data_loader import load_npz_data, one_hot_to_categorical


class ColorMappingModel(pl.LightningModule):
    """
    PyTorch Lightning model that predicts input colors from features and position embeddings.

    Args:
        feature_dim: Number of input features (147)
        pos_embed_dim: Dimension of position embeddings for x and y coordinates
        hidden_dim: Hidden dimension for the MLP
        num_classes: Number of color classes (10: colors 0-9)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    """

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        hidden_dim: int = 256,
        num_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.pos_embed_dim = pos_embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        # Position embeddings for x and y coordinates (0-29)
        self.x_embedding = nn.Embedding(30, pos_embed_dim)
        self.y_embedding = nn.Embedding(30, pos_embed_dim)

        # Input dimension: features + 2 * pos_embed_dim
        input_dim = feature_dim + 2 * pos_embed_dim

        # Deeper MLP for color prediction to handle complex feature interactions
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
        # Based on analysis: color 0 is most common (~9.5%), others are much rarer (0.2-1.1%)
        # Weight inversely proportional to frequency
        class_weights = torch.tensor([
            1.0,    # color 0 (most common)
            8.6,    # color 1 
            11.2,   # color 2
            12.1,   # color 3
            16.5,   # color 4
            12.6,   # color 5
            29.7,   # color 6
            47.2,   # color 7
            9.1,    # color 8
            55.2,   # color 9 (rarest)
        ])
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)  # Ignore mask values (-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            features: Input features of shape (batch_size, 30, 30, feature_dim)

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

        # Get position embeddings
        x_embeds = self.x_embedding(x_pos)  # (30, 30, pos_embed_dim)
        y_embeds = self.y_embedding(y_pos)  # (30, 30, pos_embed_dim)

        # Expand to batch size
        x_embeds = x_embeds.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch_size, 30, 30, pos_embed_dim)
        y_embeds = y_embeds.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch_size, 30, 30, pos_embed_dim)

        # Concatenate features with position embeddings
        combined_features = torch.cat(
            [features, x_embeds, y_embeds], dim=-1
        )  # (batch_size, 30, 30, feature_dim + 2*pos_embed_dim)

        # Apply MLP
        logits = self.mlp(combined_features)  # (batch_size, 30, 30, num_classes)

        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors = batch

        # Forward pass
        logits = self(features)  # (batch_size, 30, 30, num_classes)

        # Reshape for loss computation
        logits_flat = logits.view(
            -1, self.num_classes
        )  # (batch_size * 30 * 30, num_classes)
        colors_flat = colors.view(-1)  # (batch_size * 30 * 30,)

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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors = batch

        # Forward pass
        logits = self(features)  # (batch_size, 30, 30, num_classes)

        # Reshape for loss computation
        logits_flat = logits.view(
            -1, self.num_classes
        )  # (batch_size * 30 * 30, num_classes)
        colors_flat = colors.view(-1)  # (batch_size * 30 * 30,)

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
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def prepare_color_mapping_data(
    train_npz_path: str,
    eval_npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders for color mapping task.

    Args:
        train_npz_path: Path to training NPZ file with features
        eval_npz_path: Path to evaluation NPZ file with features
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Load training data
    print("Loading training data...")
    train_filenames, train_indices, train_inputs, train_outputs, train_features, _ = (
        load_npz_data(train_npz_path, use_features=True)
    )

    # Load validation data
    print("Loading validation data...")
    val_filenames, val_indices, val_inputs, val_outputs, val_features, _ = (
        load_npz_data(eval_npz_path, use_features=True)
    )

    # Convert one-hot encoded inputs back to categorical colors
    # Note: channel 10 represents mask (-1), but we want to convert it back to -1
    train_colors = one_hot_to_categorical(train_inputs, last_value=10)  # Get 10 for mask
    val_colors = one_hot_to_categorical(val_inputs, last_value=10)
    
    # Convert 10 back to -1 for mask
    train_colors[train_colors == 10] = -1
    val_colors[val_colors == 10] = -1

    # Check that features were loaded
    if train_features is None or val_features is None:
        raise ValueError(
            "Features not found in NPZ files. Make sure to use files with features."
        )

    # Convert to proper data types
    train_features = train_features.float()
    val_features = val_features.float()
    train_colors = train_colors.long()
    val_colors = val_colors.long()

    print(f"Training data: {train_features.shape}, colors: {train_colors.shape}")
    print(f"Validation data: {val_features.shape}, colors: {val_colors.shape}")

    # Create datasets
    train_dataset = TensorDataset(train_features, train_colors)
    val_dataset = TensorDataset(val_features, val_colors)

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

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train color mapping model")
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
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="color_mapping_outputs",
        help="Output directory for logs and checkpoints",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data loaders
    train_loader, val_loader = prepare_color_mapping_data(
        args.train_data,
        args.eval_data,
        args.batch_size,
        args.num_workers,
    )

    # Initialize model
    model = ColorMappingModel(
        feature_dim=147,
        pos_embed_dim=args.pos_embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="color_mapping-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

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
    trainer.fit(model, train_loader, val_loader)

    print(f"Training completed! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
