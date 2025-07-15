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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LookupMemorizationModel(pl.LightningModule):
    """
    Extremely simple lookup-based model for perfect memorization.

    Since we only have 2 examples, we can use a simple approach:
    1. Learn embeddings for each example
    2. Use those embeddings to directly predict the output
    3. Add strong regularization to force exact memorization
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 2048,
        num_classes: int = 10,
        lr: float = 1e-2,
        available_colors: Optional[List[int]] = None,
        num_examples: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.num_examples = num_examples

        # Register available colors as buffer
        if available_colors is not None:
            colors_tensor = torch.zeros(num_classes)
            colors_tensor[available_colors] = 1.0
            self.register_buffer("available_colors", colors_tensor)
        else:
            self.register_buffer("available_colors", torch.ones(num_classes))

        # Simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learn direct mappings for each position in each example
        # This is essentially a lookup table
        self.example_embeddings = nn.Parameter(
            torch.randn(num_examples, 30, 30, hidden_dim) * 0.01
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_classes)

        # Example classifier to identify which example we're looking at
        self.example_classifier = nn.Sequential(
            nn.Linear(feature_dim * 900, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_examples),
        )

        # Store ground truth for direct comparison
        self.register_buffer("stored_inputs", torch.zeros(num_examples, 30, 30))
        self.register_buffer("stored_outputs", torch.zeros(num_examples, 30, 30))
        self.register_buffer("examples_seen", torch.zeros(num_examples))

    def forward(
        self,
        features: torch.Tensor,
        store_mode: bool = False,
        example_idx: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = features.shape[0]

        # Extract features
        extracted_features = self.feature_extractor(features)  # [B, 30, 30, hidden_dim]

        # Identify which example this is
        flat_features = features.reshape(batch_size, -1)
        example_logits = self.example_classifier(flat_features)  # [B, num_examples]
        example_probs = F.softmax(example_logits, dim=-1)

        # Get the most likely example index
        example_indices = torch.argmax(example_probs, dim=-1)  # [B]

        # Blend embeddings based on example probabilities
        blended_embeddings = torch.zeros_like(extracted_features)
        for b in range(batch_size):
            for e in range(self.num_examples):
                weight = example_probs[b, e]
                blended_embeddings[b] += weight * self.example_embeddings[e]

        # Combine with extracted features
        combined = extracted_features + blended_embeddings

        # Project to logits
        logits = self.output_projection(combined)  # [B, 30, 30, num_classes]

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
        constrained_logits[unavailable_mask_expanded.bool()] = -1e10

        return constrained_logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with heavy weighting on errors."""
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.num_classes)
        targets_flat = targets.reshape(-1)

        # Mask for valid pixels
        valid_mask = targets_flat != -1

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits_flat[valid_mask],
            targets_flat[valid_mask],
        )

        # Add per-pixel penalty for wrong predictions
        predictions = torch.argmax(logits, dim=-1)
        wrong_mask = (predictions != targets) & (targets != -1)
        penalty = wrong_mask.float().mean() * 10.0

        return ce_loss + penalty

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
        input_logits = self(input_features)
        input_loss = self.compute_loss(input_logits, input_colors)

        # Process outputs
        output_logits = self(output_features)
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

        # Visualize every 20 epochs
        if self.current_epoch % 20 == 0 and batch_idx == 0:
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
        input_logits = self(input_features)
        input_loss = self.compute_loss(input_logits, input_colors)

        # Process outputs
        output_logits = self(output_features)
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
        # Use high learning rate with Adam
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        # Aggressive learning rate schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-5
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
        imshow(ground_truth[0].cpu())

        print(f"\nPredicted {prefix} colors:")
        imshow(predictions[0].cpu())

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

        # Show confusion matrix for errors
        errors = predictions[valid_mask] != ground_truth[valid_mask]
        if errors.any():
            print(f"\nErrors: {errors.sum().item()} pixels")
            error_pred = predictions[valid_mask][errors]
            error_true = ground_truth[valid_mask][errors]
            for true_color in torch.unique(error_true):
                mask = error_true == true_color
                if mask.any():
                    pred_colors = error_pred[mask]
                    unique_pred, counts = torch.unique(pred_colors, return_counts=True)
                    print(
                        f"  True {true_color.item()} → Predicted: {dict(zip(unique_pred.tolist(), counts.tolist()))}"
                    )


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
        available_colors = [
            i for i, available in enumerate(color_mask) if available == 1
        ]
    else:
        # If not in train, check eval
        filename_idx = np.where(eval_data["filename_colors_keys"] == filename)[0]
        if len(filename_idx) > 0:
            color_mask = eval_data["filename_colors_values"][filename_idx[0]]
            # Convert binary mask to list of available color indices
            available_colors = [
                i for i, available in enumerate(color_mask) if available == 1
            ]
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
        description="Train lookup-based memorization model on single filename"
    )
    parser.add_argument(
        "--target_filename",
        type=str,
        default="3345333e.json",
        help="Target filename to memorize",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500, help="Maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="lookup_memorization_checkpoints",
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
    model = LookupMemorizationModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        lr=args.lr,
        available_colors=available_colors,
        num_examples=num_examples,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/checkpoints",
        filename="lookup-{epoch:02d}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}",
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
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed for {args.target_filename}")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
