import argparse
import os

# Import from parent module
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcagi.data_loader import load_npz_data, one_hot_to_categorical


class FilenameAwareColorMappingModel(pl.LightningModule):
    """
    PyTorch Lightning model that predicts input colors from features, position embeddings,
    and filename-specific latent variables for memorization.

    Args:
        feature_dim: Number of input features (147)
        pos_embed_dim: Dimension of position embeddings for x and y coordinates
        filename_embed_dim: Dimension of filename-specific latent embeddings
        hidden_dim: Hidden dimension for the MLP
        num_classes: Number of color classes (10: colors 0-9)
        num_filenames: Number of unique filenames
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    """

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_filenames: int = 400,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.pos_embed_dim = pos_embed_dim
        self.filename_embed_dim = filename_embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_filenames = num_filenames
        self.lr = lr
        self.weight_decay = weight_decay

        # Sinusoidal positional embeddings for x and y coordinates (0-29)
        self.pos_embed_dim = pos_embed_dim
        self.register_buffer(
            "x_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )
        self.register_buffer(
            "y_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )

        # Filename-specific latent variables for memorization
        self.filename_embedding = nn.Embedding(num_filenames, filename_embed_dim)

        # Input dimension: features + 2 * pos_embed_dim + filename_embed_dim
        input_dim = feature_dim + 2 * pos_embed_dim + filename_embed_dim

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

    def forward(
        self, features: torch.Tensor, filename_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

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

        # Apply MLP
        logits = self.mlp(combined_features)  # (batch_size, 30, 30, num_classes)

        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors, filename_ids = batch

        # Forward pass
        logits = self(features, filename_ids)  # (batch_size, 30, 30, num_classes)

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
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors, filename_ids = batch

        # Forward pass
        logits = self(features, filename_ids)  # (batch_size, 30, 30, num_classes)

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


def create_filename_mapping(filenames: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """Create mapping from filename to integer ID."""
    unique_filenames = sorted(list(set(filenames)))
    filename_to_id = {fname: i for i, fname in enumerate(unique_filenames)}
    return filename_to_id, unique_filenames


def prepare_filename_aware_data(
    train_npz_path: str,
    eval_npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_train_as_eval: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare data loaders for filename-aware color mapping task.

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
        description="Train filename-aware color mapping model"
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
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="color_mapping_outputs_v2",
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
    train_loader, val_loader, num_filenames = prepare_filename_aware_data(
        args.train_data,
        args.eval_data,
        args.batch_size,
        args.num_workers,
        use_train_as_eval=args.use_train_as_eval,
    )

    # Initialize model
    model = FilenameAwareColorMappingModel(
        feature_dim=147,
        pos_embed_dim=args.pos_embed_dim,
        filename_embed_dim=args.filename_embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        num_filenames=num_filenames,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="filename_aware_color_mapping-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
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
