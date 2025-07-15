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


class SingleFileColorMappingModel(pl.LightningModule):
    """
    Deep MLP model for perfect memorization of color mappings for a single filename.

    This model:
    1. Uses both input and output features to predict colors
    2. Applies strong color constraints based on available colors for the filename
    3. Uses a deep MLP architecture to enable perfect memorization
    4. Combines predictions from input->input and output->output mappings

    Note: In the ARC-AGI dataset, train and eval sets have completely different filenames,
    so each filename appears in only one of the two sets.

    Args:
        feature_dim: Number of input features (147)
        hidden_dim: Hidden dimension for the MLP
        num_layers: Number of hidden layers in the MLP
        num_classes: Number of color classes (10: colors 0-9)
        available_colors: Binary mask indicating which colors are available
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_classes: int = 10,
        available_colors: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        # Store available colors as a buffer
        if available_colors is not None:
            self.register_buffer("available_colors", available_colors.float())
        else:
            # If not provided, assume all colors are available
            self.register_buffer("available_colors", torch.ones(num_classes))

        # Build deep MLP for input features -> input colors
        input_layers: List[nn.Module] = []
        in_dim = feature_dim
        for i in range(num_layers):
            input_layers.append(nn.Linear(in_dim, hidden_dim))
            input_layers.append(nn.LayerNorm(hidden_dim))
            input_layers.append(nn.ReLU())
            input_layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        input_layers.append(nn.Linear(hidden_dim, num_classes))
        self.input_mlp = nn.Sequential(*input_layers)

        # Build deep MLP for output features -> output colors
        output_layers: List[nn.Module] = []
        in_dim = feature_dim
        for i in range(num_layers):
            output_layers.append(nn.Linear(in_dim, hidden_dim))
            output_layers.append(nn.LayerNorm(hidden_dim))
            output_layers.append(nn.ReLU())
            output_layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        output_layers.append(nn.Linear(hidden_dim, num_classes))
        self.output_mlp = nn.Sequential(*output_layers)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, features: torch.Tensor, is_output: bool = False) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            features: Input features of shape (batch_size, 30, 30, feature_dim)
            is_output: Whether these are output features (use output MLP)

        Returns:
            logits: Output logits of shape (batch_size, 30, 30, num_classes)
        """
        # Use appropriate MLP based on whether these are input or output features
        if is_output:
            logits = self.output_mlp(features)
        else:
            logits = self.input_mlp(features)

        # Apply color constraints
        logits = self.apply_color_constraints(logits)

        return logits

    def apply_color_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hard constraints to eliminate unavailable colors."""
        # Create mask for unavailable colors (1 for unavailable, 0 for available)
        # Type assertion to help the type checker understand this is always a Tensor
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

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        # Unpack batch - can have 4 or 6 elements depending on whether we have both input and output data
        if len(batch) == 4:
            # Only input features and colors
            input_features, input_colors, output_features, output_colors = batch
            has_output = False
        else:
            # Both input and output features and colors
            input_features, input_colors, output_features, output_colors = batch[:4]
            has_output = True

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
        if total_pixels > 0:
            accuracy = total_correct.float() / total_pixels.float()
        else:
            accuracy = torch.tensor(0.0)

        # Log metrics
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        # Same as training step but without gradients
        if len(batch) == 4:
            input_features, input_colors, output_features, output_colors = batch
            has_output = False
        else:
            input_features, input_colors, output_features, output_colors = batch[:4]
            has_output = True

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

        # Visualize first example on first validation batch
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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def prepare_single_file_data(
    train_npz_path: str,
    eval_npz_path: str,
    target_filename: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Dict[str, Any]]:
    """
    Prepare data loaders for a single filename, combining train and eval data.

    Returns:
        train_loader: Training data loader (all examples for the filename)
        val_loader: Validation data loader (same as train for memorization)
        available_colors: Binary mask of which colors are used by this filename
        stats: Dictionary with data statistics
    """
    # Load training data
    print(f"Loading training data from {train_npz_path}...")
    train_data = np.load(train_npz_path)

    train_filenames = train_data["filenames"]
    train_inputs = train_data["inputs"]  # Shape: (N, 30, 30)
    train_outputs = train_data["outputs"]  # Shape: (N, 30, 30)
    train_inputs_features = train_data["inputs_features"]  # Shape: (N, 30, 30, 147)
    train_outputs_features = train_data["outputs_features"]  # Shape: (N, 30, 30, 147)

    # Load eval data
    print(f"Loading evaluation data from {eval_npz_path}...")
    eval_data = np.load(eval_npz_path)

    eval_filenames = eval_data["filenames"]
    eval_inputs = eval_data["inputs"]
    eval_outputs = eval_data["outputs"]
    eval_inputs_features = eval_data["inputs_features"]
    eval_outputs_features = eval_data["outputs_features"]

    # Find indices for target filename in both datasets
    train_mask = train_filenames == target_filename
    eval_mask = eval_filenames == target_filename

    train_indices = np.where(train_mask)[0]
    eval_indices = np.where(eval_mask)[0]

    print(f"\nTarget filename: {target_filename}")
    print(f"Found {len(train_indices)} examples in training data")
    print(f"Found {len(eval_indices)} examples in evaluation data")

    if len(train_indices) == 0 and len(eval_indices) == 0:
        raise ValueError(f"No examples found for filename {target_filename}")

    # Extract data for target filename
    if len(train_indices) > 0:
        selected_train_inputs = train_inputs[train_indices]
        selected_train_outputs = train_outputs[train_indices]
        selected_train_inputs_features = train_inputs_features[train_indices]
        selected_train_outputs_features = train_outputs_features[train_indices]
    else:
        selected_train_inputs = np.empty((0, 30, 30), dtype=np.int32)
        selected_train_outputs = np.empty((0, 30, 30), dtype=np.int32)
        selected_train_inputs_features = np.empty((0, 30, 30, 147), dtype=np.uint8)
        selected_train_outputs_features = np.empty((0, 30, 30, 147), dtype=np.uint8)

    if len(eval_indices) > 0:
        selected_eval_inputs = eval_inputs[eval_indices]
        selected_eval_outputs = eval_outputs[eval_indices]
        selected_eval_inputs_features = eval_inputs_features[eval_indices]
        selected_eval_outputs_features = eval_outputs_features[eval_indices]
    else:
        selected_eval_inputs = np.empty((0, 30, 30), dtype=np.int32)
        selected_eval_outputs = np.empty((0, 30, 30), dtype=np.int32)
        selected_eval_inputs_features = np.empty((0, 30, 30, 147), dtype=np.uint8)
        selected_eval_outputs_features = np.empty((0, 30, 30, 147), dtype=np.uint8)

    # Combine train and eval data
    all_inputs = np.concatenate([selected_train_inputs, selected_eval_inputs], axis=0)
    all_outputs = np.concatenate(
        [selected_train_outputs, selected_eval_outputs], axis=0
    )
    all_inputs_features = np.concatenate(
        [selected_train_inputs_features, selected_eval_inputs_features], axis=0
    )
    all_outputs_features = np.concatenate(
        [selected_train_outputs_features, selected_eval_outputs_features], axis=0
    )

    print(f"\nTotal examples: {len(all_inputs)}")
    print(f"Input shape: {all_inputs.shape}")
    print(f"Output shape: {all_outputs.shape}")
    print(f"Input features shape: {all_inputs_features.shape}")
    print(f"Output features shape: {all_outputs_features.shape}")

    # Find which colors are used by this filename
    all_colors_used = set()
    for grid in all_inputs:
        colors = set(grid.flatten())
        colors.discard(-1)  # Remove mask
        all_colors_used.update(colors)
    for grid in all_outputs:
        colors = set(grid.flatten())
        colors.discard(-1)  # Remove mask
        all_colors_used.update(colors)

    # Create available colors mask
    available_colors = torch.zeros(10)
    for color in all_colors_used:
        if 0 <= color <= 9:
            available_colors[color] = 1

    print(f"\nColors used by {target_filename}: {sorted(all_colors_used)}")
    print(f"Number of distinct colors: {len(all_colors_used)}")

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(all_inputs).long()
    outputs_tensor = torch.from_numpy(all_outputs).long()
    inputs_features_tensor = torch.from_numpy(all_inputs_features).float()
    outputs_features_tensor = torch.from_numpy(all_outputs_features).float()

    # Create dataset
    dataset = TensorDataset(
        inputs_features_tensor, inputs_tensor, outputs_features_tensor, outputs_tensor
    )

    # For memorization, use the same data for train and validation
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Collect statistics
    stats = {
        "total_examples": len(all_inputs),
        "train_examples": len(train_indices),
        "eval_examples": len(eval_indices),
        "colors_used": sorted(all_colors_used),
        "num_colors": len(all_colors_used),
    }

    return train_loader, val_loader, available_colors, stats


def main():
    parser = argparse.ArgumentParser(
        description="Train a deep MLP to memorize color mappings for a single filename"
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
        default=8,
        help="Batch size (default: 8 for single file)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of hidden layers in MLP",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
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
        default="single_file_checkpoints",
        help="Directory to save checkpoints",
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
    model = SingleFileColorMappingModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=10,
        available_colors=available_colors,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/checkpoints",
        filename="single_file-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc", mode="max", patience=30, verbose=True
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
        enable_model_summary=True,
    )

    # Train the model
    print(f"\nTraining on filename: {args.target_filename}")
    print(f"Goal: Achieve perfect memorization (100% accuracy)")
    trainer.fit(model, train_loader, val_loader)

    # Load best model and evaluate
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nLoading best model from {best_model_path}")
        model = SingleFileColorMappingModel.load_from_checkpoint(
            best_model_path,
            feature_dim=147,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=10,
            available_colors=available_colors,
        )

        # Final evaluation
        trainer.validate(model, val_loader)


if __name__ == "__main__":
    main()
