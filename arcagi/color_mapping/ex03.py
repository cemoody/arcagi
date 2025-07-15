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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcagi.data_loader import load_npz_data, one_hot_to_categorical


class AttentionBasedColorMappingModel(pl.LightningModule):
    """
    Attention-based model that treats color mapping as a graph coloring problem.
    Uses PyTorch's built-in transformer encoder layers for spatial coordination.

    Args:
        feature_dim: Number of input features (147)
        pos_embed_dim: Dimension of position embeddings for x and y coordinates
        filename_embed_dim: Dimension of filename-specific latent embeddings
        embed_dim: Embedding dimension for attention layers
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        num_classes: Number of color classes (10: colors 0-9)
        num_filenames: Number of unique filenames
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        dropout: Dropout rate for transformer layers
    """

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 32,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
        num_filenames: int = 800,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.pos_embed_dim = pos_embed_dim
        self.filename_embed_dim = filename_embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_filenames = num_filenames
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

        # Input projection to embedding dimension
        input_dim = feature_dim + 2 * pos_embed_dim + filename_embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # PyTorch transformer encoder layers for global coordination
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # Important: batch dimension first
            norm_first=False,  # Layer norm after residual connection (standard)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),  # Final layer norm
        )

        # Output projection to color classes
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
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
        Forward pass treating the spatial grid as a sequence for attention.

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

        # Project to embedding dimension
        x = self.input_projection(combined_features)  # (batch_size, 30, 30, embed_dim)

        # Reshape to sequence format for attention: (batch_size, seq_len, embed_dim)
        x = x.view(
            batch_size, height * width, self.embed_dim
        )  # (batch_size, 900, embed_dim)

        # Apply transformer encoder for global coordination
        # PyTorch transformer expects (batch_size, seq_len, embed_dim) when batch_first=True
        x = self.transformer_encoder(x)  # (batch_size, 900, embed_dim)

        # Project to output classes
        logits = self.output_projection(x)  # (batch_size, 900, num_classes)

        # Reshape back to spatial format
        logits = logits.view(
            batch_size, height, width, self.num_classes
        )  # (batch_size, 30, 30, num_classes)

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


# Reuse data preparation functions from ex02.py
def create_filename_mapping(filenames: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """Create mapping from filename to integer ID."""
    unique_filenames = sorted(list(set(filenames)))
    filename_to_id = {fname: i for i, fname in enumerate(unique_filenames)}
    return filename_to_id, unique_filenames


def prepare_attention_data(
    train_npz_path: str,
    eval_npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Prepare data loaders for attention-based color mapping task.
    """
    from torch.utils.data import DataLoader, TensorDataset

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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, num_filenames


def main():
    parser = argparse.ArgumentParser(
        description="Train attention-based color mapping model"
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
        help="Batch size (smaller due to attention)",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate (smaller for attention)"
    )
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
        "--embed_dim", type=int, default=256, help="Attention embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate for transformer layers"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="color_mapping_outputs_v3",
        help="Output directory for logs and checkpoints",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data loaders
    train_loader, val_loader, num_filenames = prepare_attention_data(
        args.train_data,
        args.eval_data,
        args.batch_size,
        args.num_workers,
    )

    # Initialize model
    model = AttentionBasedColorMappingModel(
        feature_dim=147,
        pos_embed_dim=args.pos_embed_dim,
        filename_embed_dim=args.filename_embed_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=10,
        num_filenames=num_filenames,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="attention_color_mapping-{epoch:02d}-{val_loss:.4f}",
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
        accumulate_grad_batches=2,  # Effective batch size = 16 * 2 = 32
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"Training completed! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
