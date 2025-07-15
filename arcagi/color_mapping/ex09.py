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


class ImprovedSingleFileModel(pl.LightningModule):
    """
    Improved deep MLP model for perfect memorization of color mappings for a single filename.

    Key improvements:
    1. Residual connections for better gradient flow
    2. Position-aware features with sinusoidal embeddings
    3. Color-specific sub-networks for better specialization
    4. Auxiliary losses for better training
    5. Temperature scaling for sharper predictions
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_classes: int = 10,
        available_colors: Optional[torch.Tensor] = None,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        use_residual: bool = True,
        use_position_encoding: bool = True,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_position_encoding = use_position_encoding
        self.temperature = temperature

        # Store available colors as a buffer
        if available_colors is not None:
            self.register_buffer("available_colors", available_colors.float())
        else:
            self.register_buffer("available_colors", torch.ones(num_classes))

        # Position encoding dimension
        self.pos_dim = 64 if use_position_encoding else 0

        # Create sinusoidal position encodings if enabled
        if self.use_position_encoding:
            self.register_buffer("pos_encoding", self._create_position_encoding())

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim + self.pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Build residual blocks for input network
        self.input_blocks = nn.ModuleList()
        for i in range(num_layers):
            block = ResidualBlock(hidden_dim, dropout)
            self.input_blocks.append(block)

        # Build residual blocks for output network
        self.output_blocks = nn.ModuleList()
        for i in range(num_layers):
            block = ResidualBlock(hidden_dim, dropout)
            self.output_blocks.append(block)

        # Color-specific prediction heads
        self.color_heads = nn.ModuleList()
        for i in range(num_classes):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.color_heads.append(head)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Auxiliary losses
        self.mse_criterion = nn.MSELoss()

    def _create_position_encoding(self) -> torch.Tensor:
        """Create sinusoidal position encodings."""
        max_len = 30
        d_model = self.pos_dim

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * -(np.log(10000.0) / (d_model // 2))
        )

        pe = torch.zeros(max_len, max_len, d_model)
        for i in range(max_len):
            for j in range(max_len):
                # X position encoding (first half of dimensions)
                pe[i, j, 0:d_model//2:2] = torch.sin(position[i] * div_term)
                pe[i, j, 1:d_model//2:2] = torch.cos(position[i] * div_term)
                # Y position encoding (second half of dimensions)
                pe[i, j, d_model//2::2] = torch.sin(position[j] * div_term)
                pe[i, j, d_model//2+1::2] = torch.cos(position[j] * div_term)

        return pe

    def forward(self, features: torch.Tensor, is_output: bool = False) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            features: Input features of shape (batch_size, 30, 30, feature_dim)
            is_output: Whether these are output features (use output MLP)

        Returns:
            logits: Output logits of shape (batch_size, 30, 30, num_classes)
        """
        batch_size = features.shape[0]

        # Add position encodings if enabled
        if self.use_position_encoding:
            pos_enc = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)
            features = torch.cat([features, pos_enc], dim=-1)

        # Project to hidden dimension
        x = self.input_projection(features)

        # Apply residual blocks
        blocks = self.output_blocks if is_output else self.input_blocks
        for block in blocks:
            x = block(x)

        # Apply color-specific heads
        logits_list = []
        for i, head in enumerate(self.color_heads):
            logit = head(x)  # (batch_size, 30, 30, 1)
            logits_list.append(logit)

        logits = torch.cat(logits_list, dim=-1)  # (batch_size, 30, 30, num_classes)

        # Apply temperature scaling
        logits = logits / self.temperature

        # Apply color constraints
        logits = self.apply_color_constraints(logits)

        return logits

    def apply_color_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hard constraints to eliminate unavailable colors."""
        available_colors: torch.Tensor = self.available_colors  # type: ignore
        unavailable_mask = 1 - available_colors

        # Expand mask to match logits shape
        batch_size = logits.shape[0]
        unavailable_mask_expanded = (
            unavailable_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        )
        unavailable_mask_expanded = unavailable_mask_expanded.expand(
            batch_size, 30, 30, -1
        )

        # Set logits for unavailable colors to very negative value
        constrained_logits = logits.clone()
        constrained_logits[unavailable_mask_expanded.bool()] = -1e10

        return constrained_logits

    def compute_auxiliary_loss(
        self, logits: torch.Tensor, colors: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary losses to help training."""
        # Entropy regularization to encourage confident predictions
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Only compute on valid pixels
        valid_mask = colors != -1
        if valid_mask.any():
            entropy_loss = entropy[valid_mask].mean()
        else:
            entropy_loss = torch.tensor(0.0)

        return 0.1 * entropy_loss  # Small weight for regularization

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        # Unpack batch
        if len(batch) == 4:
            input_features, input_colors, output_features, output_colors = batch
        else:
            input_features, input_colors, output_features, output_colors = batch[:4]

        total_loss = 0.0
        total_correct = 0
        total_pixels = 0

        # Predict input colors from input features
        input_logits = self(input_features, is_output=False)
        input_logits_flat = input_logits.view(-1, self.num_classes)
        input_colors_flat = input_colors.view(-1)

        # Mask out invalid positions
        valid_mask = input_colors_flat != -1
        if valid_mask.any():
            input_loss = self.criterion(
                input_logits_flat[valid_mask], input_colors_flat[valid_mask]
            )
            aux_loss = self.compute_auxiliary_loss(input_logits, input_colors)
            total_loss += input_loss + aux_loss

            # Calculate accuracy
            input_preds = torch.argmax(input_logits_flat[valid_mask], dim=-1)
            input_correct = (input_preds == input_colors_flat[valid_mask]).sum()
            total_correct += input_correct
            total_pixels += valid_mask.sum()

        # Predict output colors from output features
        if output_features is not None and output_colors is not None:
            output_logits = self(output_features, is_output=True)
            output_logits_flat = output_logits.view(-1, self.num_classes)
            output_colors_flat = output_colors.view(-1)

            # Mask out invalid positions
            valid_mask = output_colors_flat != -1
            if valid_mask.any():
                output_loss = self.criterion(
                    output_logits_flat[valid_mask], output_colors_flat[valid_mask]
                )
                aux_loss = self.compute_auxiliary_loss(output_logits, output_colors)
                total_loss += output_loss + aux_loss

                # Calculate accuracy
                output_preds = torch.argmax(output_logits_flat[valid_mask], dim=-1)
                output_correct = (output_preds == output_colors_flat[valid_mask]).sum()
                total_correct += output_correct
                total_pixels += valid_mask.sum()

        # Calculate overall accuracy
        accuracy = total_correct.float() / total_pixels if total_pixels > 0 else 0.0

        # Log metrics
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        # Same as training step but without auxiliary losses
        if len(batch) == 4:
            input_features, input_colors, output_features, output_colors = batch
        else:
            input_features, input_colors, output_features, output_colors = batch[:4]

        total_loss = 0.0
        total_correct = 0
        total_pixels = 0

        # Predict input colors from input features
        input_logits = self(input_features, is_output=False)
        input_logits_flat = input_logits.view(-1, self.num_classes)
        input_colors_flat = input_colors.view(-1)

        # Mask out invalid positions
        valid_mask = input_colors_flat != -1
        if valid_mask.any():
            input_loss = self.criterion(
                input_logits_flat[valid_mask], input_colors_flat[valid_mask]
            )
            total_loss += input_loss

            # Calculate accuracy
            input_preds = torch.argmax(input_logits_flat[valid_mask], dim=-1)
            input_correct = (input_preds == input_colors_flat[valid_mask]).sum()
            total_correct += input_correct
            total_pixels += valid_mask.sum()

        # Predict output colors from output features
        if output_features is not None and output_colors is not None:
            output_logits = self(output_features, is_output=True)
            output_logits_flat = output_logits.view(-1, self.num_classes)
            output_colors_flat = output_colors.view(-1)

            # Mask out invalid positions
            valid_mask = output_colors_flat != -1
            if valid_mask.any():
                output_loss = self.criterion(
                    output_logits_flat[valid_mask], output_colors_flat[valid_mask]
                )
                total_loss += output_loss

                # Calculate accuracy
                output_preds = torch.argmax(output_logits_flat[valid_mask], dim=-1)
                output_correct = (output_preds == output_colors_flat[valid_mask]).sum()
                total_correct += output_correct
                total_pixels += valid_mask.sum()

        # Calculate overall accuracy
        accuracy = total_correct.float() / total_pixels if total_pixels > 0 else 0.0

        # Log metrics
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Visualize predictions periodically
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self.visualize_predictions(
                input_features[0], input_colors[0], input_logits[0], "input"
            )
            if output_features is not None:
                self.visualize_predictions(
                    output_features[0], output_colors[0], output_logits[0], "output"
                )

        return total_loss

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
            correct = (predictions[valid_mask] == ground_truth[valid_mask]).sum()
            total = valid_mask.sum()
            acc = correct.float() / total
            print(f"\nAccuracy: {acc:.2%} ({correct}/{total} pixels)")

        # Show which colors are being used
        unique_gt = torch.unique(ground_truth[ground_truth != -1])
        unique_pred = torch.unique(predictions[ground_truth != -1])
        print(f"Ground truth colors: {sorted(unique_gt.tolist())}")
        print(f"Predicted colors: {sorted(unique_pred.tolist())}")
        print(f"Available colors: {torch.where(self.available_colors)[0].tolist()}")

    def configure_optimizers(self):
        # Use AdamW with cosine annealing
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and GELU activation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ff2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection
        residual = x
        x = self.norm1(x)
        x = self.ff1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)
        return x


def prepare_single_file_data(
    train_path: str,
    eval_path: str,
    target_filename: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Dict[str, Any]]:
    """
    Prepare data loaders for a single filename, combining train and eval data.
    """
    # Load training data
    train_data = np.load(train_path)
    train_filenames = train_data["filenames"]
    train_inputs = train_data["inputs"]
    train_outputs = train_data["outputs"]
    train_input_features = train_data["inputs_features"]
    train_output_features = train_data["outputs_features"]

    # Load evaluation data
    eval_data = np.load(eval_path)
    eval_filenames = eval_data["filenames"]
    eval_inputs = eval_data["inputs"]
    eval_outputs = eval_data["outputs"]
    eval_input_features = eval_data["inputs_features"]
    eval_output_features = eval_data["outputs_features"]

    # Filter for target filename in both datasets
    train_mask = train_filenames == target_filename
    eval_mask = eval_filenames == target_filename

    # Get examples for this filename
    train_inputs_filtered = train_inputs[train_mask]
    train_outputs_filtered = train_outputs[train_mask]
    train_input_features_filtered = train_input_features[train_mask]
    train_output_features_filtered = train_output_features[train_mask]

    eval_inputs_filtered = eval_inputs[eval_mask]
    eval_outputs_filtered = eval_outputs[eval_mask]
    eval_input_features_filtered = eval_input_features[eval_mask]
    eval_output_features_filtered = eval_output_features[eval_mask]

    # Combine all data (we don't care about generalization for single file)
    all_inputs = np.concatenate([train_inputs_filtered, eval_inputs_filtered], axis=0)
    all_outputs = np.concatenate(
        [train_outputs_filtered, eval_outputs_filtered], axis=0
    )
    all_input_features = np.concatenate(
        [train_input_features_filtered, eval_input_features_filtered], axis=0
    )
    all_output_features = np.concatenate(
        [train_output_features_filtered, eval_output_features_filtered], axis=0
    )

    print(f"\nTarget filename: {target_filename}")
    print(f"Found {len(train_inputs_filtered)} examples in training data")
    print(f"Found {len(eval_inputs_filtered)} examples in evaluation data")
    print(f"\nTotal examples: {len(all_inputs)}")
    print(f"Input shape: {all_inputs.shape}")
    print(f"Output shape: {all_outputs.shape}")
    print(f"Input features shape: {all_input_features.shape}")
    print(f"Output features shape: {all_output_features.shape}")

    # Determine which colors are used by this filename
    all_colors = set()
    for inp in all_inputs:
        colors = set(inp.flatten())
        colors.discard(-1)  # Remove mask
        all_colors.update(colors)
    for out in all_outputs:
        colors = set(out.flatten())
        colors.discard(-1)  # Remove mask
        all_colors.update(colors)

    # Create available colors tensor
    available_colors = torch.zeros(10)
    for color in all_colors:
        if 0 <= color <= 9:
            available_colors[color] = 1

    print(f"\nColors used by {target_filename}: {sorted(all_colors)}")
    print(f"Number of distinct colors: {len(all_colors)}")

    # Convert to tensors
    inputs_tensor = torch.from_numpy(all_inputs).long()
    outputs_tensor = torch.from_numpy(all_outputs).long()
    input_features_tensor = torch.from_numpy(all_input_features).float()
    output_features_tensor = torch.from_numpy(all_output_features).float()

    # Create dataset
    dataset = TensorDataset(
        input_features_tensor,
        inputs_tensor,
        output_features_tensor,
        outputs_tensor,
    )

    # Split into train/val (use all data for both since we want to memorize)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Collect statistics
    stats = {
        "total_examples": len(all_inputs),
        "train_examples": len(train_inputs_filtered),
        "eval_examples": len(eval_inputs_filtered),
        "colors_used": sorted(all_colors),
        "num_colors": len(all_colors),
    }

    return train_loader, val_loader, available_colors, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train improved deep model on single filename for perfect memorization"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="processed_data/train_all.npz",
        help="Path to training NPZ file with all features",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="processed_data/eval_all.npz",
        help="Path to evaluation NPZ file with all features",
    )
    parser.add_argument(
        "--target_filename",
        type=str,
        default="3345333e.json",
        help="Target filename to train on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 2 for single file)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="Hidden dimension for MLP",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of residual blocks",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for logit scaling",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="single_file_improved_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        default=True,
        help="Use residual connections",
    )
    parser.add_argument(
        "--use_position_encoding",
        action="store_true",
        default=True,
        help="Use sinusoidal position encodings",
    )

    args = parser.parse_args()

    # Prepare data
    train_loader, val_loader, available_colors, stats = prepare_single_file_data(
        args.train_path,
        args.eval_path,
        args.target_filename,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")

    # Create model
    model = ImprovedSingleFileModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=10,
        available_colors=available_colors,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_residual=args.use_residual,
        use_position_encoding=args.use_position_encoding,
        temperature=args.temperature,
    )

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/checkpoints",
        filename="improved-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc", mode="max", patience=50, verbose=True
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print(f"\nTraining on filename: {args.target_filename}")
    print(f"Goal: Achieve perfect memorization (100% accuracy)")

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Load best model and evaluate
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nLoading best model from {best_model_path}")
        model = ImprovedSingleFileModel.load_from_checkpoint(
            best_model_path, available_colors=available_colors
        )
        trainer.validate(model, val_loader)
