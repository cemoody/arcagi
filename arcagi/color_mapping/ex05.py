import argparse
import os

# Import from parent module
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcagi.data_loader import load_npz_data, one_hot_to_categorical


def analyze_color_usage(
    filenames: List[str], inputs_array: torch.Tensor
) -> Dict[str, Set[int]]:
    """Analyze which colors are used by each filename across all its examples."""
    filename_to_colors: Dict[str, Set[int]] = defaultdict(set)

    # Convert to numpy for easier manipulation
    inputs_np = (
        inputs_array.numpy() if isinstance(inputs_array, torch.Tensor) else inputs_array
    )

    for fname, input_grid in zip(filenames, inputs_np):
        colors_used = set(int(c) for c in input_grid.flatten() if c != -1)
        filename_to_colors[fname].update(colors_used)

    # Print analysis
    print("\n=== Color Usage Analysis ===")
    ambiguous_files = []
    color_counts = defaultdict(int)

    for fname, colors in filename_to_colors.items():
        if len(colors) < 10:  # Not using all colors
            ambiguous_files.append((fname, colors))
        for color in colors:
            color_counts[color] += 1

    print(f"Total unique filenames: {len(filename_to_colors)}")
    print(f"Files using < 10 colors: {len(ambiguous_files)}")

    if ambiguous_files:
        print("\nFiles with limited color usage (potential ambiguity):")
        for fname, colors in sorted(ambiguous_files[:10]):  # Show first 10
            print(f"  {fname}: Uses only {sorted(colors)} ({len(colors)} colors)")

    print("\nColor frequency across files:")
    for color in range(10):
        print(
            f"  Color {color}: Used in {color_counts[color]}/{len(filename_to_colors)} files"
        )

    # Count distribution of number of colors per file
    color_count_distribution = defaultdict(int)
    for fname, colors in filename_to_colors.items():
        num_colors = len(colors)
        color_count_distribution[num_colors] += 1

    print("\nDistribution of distinct colors per file:")
    for num_colors in range(1, 11):
        count = color_count_distribution.get(num_colors, 0)
        print(f"  {num_colors} colors: {count} files")

    return dict(filename_to_colors)


def check_color_consistency(
    filenames: List[str], inputs_array: torch.Tensor
) -> Dict[str, List[np.ndarray]]:
    """Check if examples within a filename have consistent color mappings."""
    filename_groups = defaultdict(list)

    # Convert to numpy
    if isinstance(inputs_array, torch.Tensor):
        inputs_array = inputs_array.numpy()

    for fname, input_grid in zip(filenames, inputs_array):
        filename_groups[fname].append(input_grid)

    print("\n=== Color Consistency Analysis ===")
    print(
        f"Files with multiple examples: {sum(1 for grids in filename_groups.values() if len(grids) > 1)}"
    )

    # Analyze example distribution
    example_counts = defaultdict(int)
    for grids in filename_groups.values():
        example_counts[len(grids)] += 1

    print("\nExamples per file distribution:")
    for count, num_files in sorted(example_counts.items()):
        print(f"  {count} examples: {num_files} files")

    return dict(filename_groups)


def analyze_error_patterns(
    predictions: torch.Tensor, targets: torch.Tensor, filenames: List[str]
) -> Dict:
    """Analyze where the errors are concentrated."""
    errors_by_color = defaultdict(lambda: {"correct": 0, "total": 0})
    errors_by_file = defaultdict(lambda: {"correct": 0, "total": 0})

    # Flatten predictions and targets
    preds_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    # Expand filenames to match flattened size
    expanded_filenames = []
    for fname in filenames:
        expanded_filenames.extend([fname] * (30 * 30))

    # Analyze errors
    mask = targets_flat != -1  # Ignore mask values

    for pred, target, fname in zip(
        preds_flat[mask],
        targets_flat[mask],
        [expanded_filenames[i] for i in range(len(expanded_filenames)) if mask[i]],
    ):
        color = target.item()
        is_correct = (pred == target).item()

        errors_by_color[color]["total"] += 1
        errors_by_color[color]["correct"] += int(is_correct)

        errors_by_file[fname]["total"] += 1
        errors_by_file[fname]["correct"] += int(is_correct)

    return {"by_color": dict(errors_by_color), "by_file": dict(errors_by_file)}


class GlobalConsistencyColorMappingModel(pl.LightningModule):
    """
    Enhanced model that uses stronger filename influence and global consistency.

    Key improvements:
    1. Filename embeddings gate/modulate the features
    2. Multi-head self-attention for global consistency
    3. Iterative refinement of predictions
    4. Additional consistency loss
    """

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 64,  # Larger filename embedding
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_filenames: int = 400,
        num_refinement_steps: int = 3,
        consistency_weight: float = 0.1,
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
        self.num_refinement_steps = num_refinement_steps
        self.consistency_weight = consistency_weight
        self.lr = lr
        self.weight_decay = weight_decay

        # Sinusoidal positional embeddings
        self.register_buffer(
            "x_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )
        self.register_buffer(
            "y_pos_embed", self._create_sinusoidal_embeddings(30, pos_embed_dim)
        )

        # Filename embedding with larger dimension
        self.filename_embedding = nn.Embedding(num_filenames, filename_embed_dim)

        # Filename-based feature gating
        self.filename_gate_mlp = nn.Sequential(
            nn.Linear(filename_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        # Initial feature processing
        input_dim = feature_dim + 2 * pos_embed_dim + filename_embed_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Self-attention for global consistency
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Iterative refinement layers
        self.refinement_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + num_classes, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes),
                )
                for _ in range(num_refinement_steps)
            ]
        )

        # Initial prediction layer
        self.initial_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Loss function with class weights
        class_weights = torch.tensor(
            [1.0, 8.6, 11.2, 12.1, 16.5, 12.6, 29.7, 47.2, 9.1, 55.2]
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

    def compute_consistency_loss(
        self, logits: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss based on edge features.
        Penalizes predictions where adjacent pixels with similar features have different colors.
        """
        batch_size, height, width, num_classes = logits.shape

        # Get predicted probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute pairwise similarity based on features (simplified)
        # In practice, you'd use the actual edge features from preprocessing
        consistency_loss = 0.0

        # Horizontal consistency
        if width > 1:
            h_diff = F.mse_loss(probs[:, :, :-1], probs[:, :, 1:], reduction="none")
            # Weight by feature similarity (placeholder - you'd use actual edge features)
            consistency_loss += h_diff.mean()

        # Vertical consistency
        if height > 1:
            v_diff = F.mse_loss(probs[:, :-1, :], probs[:, 1:, :], reduction="none")
            consistency_loss += v_diff.mean()

        return consistency_loss

    def forward(
        self, features: torch.Tensor, filename_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with feature gating and iterative refinement."""
        batch_size, height, width, _ = features.shape

        # Position embeddings
        x_pos = (
            torch.arange(width, device=features.device).unsqueeze(0).expand(height, -1)
        )
        y_pos = (
            torch.arange(height, device=features.device).unsqueeze(1).expand(-1, width)
        )

        x_embeds = self.x_pos_embed[x_pos].unsqueeze(0).expand(batch_size, -1, -1, -1)
        y_embeds = self.y_pos_embed[y_pos].unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Filename embeddings
        filename_embeds = self.filename_embedding(filename_ids)

        # Gate features using filename embedding
        filename_gate = self.filename_gate_mlp(
            filename_embeds
        )  # (batch_size, feature_dim)
        filename_gate = (
            filename_gate.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, -1)
        )
        gated_features = features * filename_gate  # Modulate features by filename

        # Expand filename embeddings spatially
        filename_embeds_spatial = (
            filename_embeds.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, -1)
        )

        # Combine all inputs
        combined_features = torch.cat(
            [gated_features, x_embeds, y_embeds, filename_embeds_spatial], dim=-1
        )

        # Encode features
        encoded = self.feature_encoder(
            combined_features
        )  # (batch_size, 30, 30, hidden_dim)

        # Reshape for self-attention
        seq_len = height * width
        encoded_seq = encoded.view(batch_size, seq_len, self.hidden_dim)

        # Apply self-attention for global consistency
        attended, _ = self.self_attention(encoded_seq, encoded_seq, encoded_seq)
        attended = attended.view(batch_size, height, width, self.hidden_dim)

        # Initial prediction
        logits = self.initial_predictor(attended)

        # Iterative refinement
        for refinement_layer in self.refinement_layers:
            # Concatenate current predictions with features
            refined_input = torch.cat([attended, logits], dim=-1)
            # Refine predictions
            logits = logits + refinement_layer(refined_input)  # Residual connection

        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, colors, filename_ids = batch

        # Forward pass
        logits = self(features, filename_ids)

        # Reshape for loss
        logits_flat = logits.reshape(-1, self.num_classes)
        colors_flat = colors.reshape(-1)

        # Classification loss
        cls_loss = self.criterion(logits_flat, colors_flat)

        # Consistency loss
        consistency_loss = self.compute_consistency_loss(logits, features)

        # Total loss
        loss = cls_loss + self.consistency_weight * consistency_loss

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=1)
            mask = colors_flat != -1
            if mask.sum() > 0:
                acc = (preds[mask] == colors_flat[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log(
            "train_consistency_loss", consistency_loss, on_step=False, on_epoch=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        features, colors, filename_ids = batch

        # Forward pass
        logits = self(features, filename_ids)

        # Reshape for loss
        logits_flat = logits.reshape(-1, self.num_classes)
        colors_flat = colors.reshape(-1)

        # Classification loss
        cls_loss = self.criterion(logits_flat, colors_flat)

        # Consistency loss
        consistency_loss = self.compute_consistency_loss(logits, features)

        # Total loss
        loss = cls_loss + self.consistency_weight * consistency_loss

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=1)
            mask = colors_flat != -1
            if mask.sum() > 0:
                acc = (preds[mask] == colors_flat[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log("val_consistency_loss", consistency_loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Return predictions for error analysis
        return {
            "loss": loss,
            "predictions": preds.view(colors.shape),
            "targets": colors,
            "filename_ids": filename_ids,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# Data preparation functions
def create_filename_mapping(filenames: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """Create mapping from filename to integer ID."""
    unique_filenames = sorted(list(set(filenames)))
    filename_to_id = {fname: i for i, fname in enumerate(unique_filenames)}
    return filename_to_id, unique_filenames


def prepare_data_with_analysis(
    train_npz_path: str,
    eval_npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    run_analysis: bool = True,
) -> Tuple[DataLoader, DataLoader, int, Dict[str, int]]:
    """Prepare data loaders with optional diagnostic analysis."""

    # Load training data
    print("Loading training data...")
    train_filenames, _, train_inputs, _, train_features, _ = load_npz_data(
        train_npz_path, use_features=True
    )

    # Load validation data
    print("Loading validation data...")
    val_filenames, _, val_inputs, _, val_features, _ = load_npz_data(
        eval_npz_path, use_features=True
    )

    # Convert to categorical
    train_colors = one_hot_to_categorical(train_inputs, last_value=10)
    val_colors = one_hot_to_categorical(val_inputs, last_value=10)
    train_colors[train_colors == 10] = -1
    val_colors[val_colors == 10] = -1

    # Run diagnostic analysis
    if run_analysis:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC ANALYSIS")
        print("=" * 60)

        # Analyze color usage
        train_color_usage = analyze_color_usage(train_filenames, train_colors)

        # Check consistency
        train_filename_groups = check_color_consistency(train_filenames, train_colors)

        print("\n" + "=" * 60 + "\n")

    # Create filename mappings
    all_filenames = train_filenames + val_filenames
    filename_to_id, unique_filenames = create_filename_mapping(all_filenames)
    num_filenames = len(unique_filenames)

    # Convert filenames to IDs
    train_filename_ids = torch.tensor(
        [filename_to_id[fname] for fname in train_filenames], dtype=torch.long
    )
    val_filename_ids = torch.tensor(
        [filename_to_id[fname] for fname in val_filenames], dtype=torch.long
    )

    # Convert to tensors
    train_features = train_features.float()
    val_features = val_features.float()
    train_colors = train_colors.long()
    val_colors = val_colors.long()

    # Create datasets
    train_dataset = TensorDataset(train_features, train_colors, train_filename_ids)
    val_dataset = TensorDataset(val_features, val_colors, val_filename_ids)

    # Create loaders
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

    return train_loader, val_loader, num_filenames, filename_to_id


def main():
    parser = argparse.ArgumentParser(
        description="Train global consistency color mapping model with diagnostics"
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
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--filename_embed_dim",
        type=int,
        default=64,
        help="Filename embedding dimension",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num_refinement_steps", type=int, default=3, help="Number of refinement steps"
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.1,
        help="Weight for consistency loss",
    )
    parser.add_argument(
        "--skip_analysis", action="store_true", help="Skip diagnostic analysis"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="color_mapping_outputs_v5",
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data with analysis
    train_loader, val_loader, num_filenames, filename_to_id = (
        prepare_data_with_analysis(
            args.train_data,
            args.eval_data,
            args.batch_size,
            num_workers=4,
            run_analysis=not args.skip_analysis,
        )
    )

    # Initialize model
    model = GlobalConsistencyColorMappingModel(
        feature_dim=147,
        pos_embed_dim=16,
        filename_embed_dim=args.filename_embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        num_filenames=num_filenames,
        num_refinement_steps=args.num_refinement_steps,
        consistency_weight=args.consistency_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="global_consistency-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
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

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Train
    print("\nStarting training with global consistency model...")
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
