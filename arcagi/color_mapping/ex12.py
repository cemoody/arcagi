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


class MessagePassingColorModel(pl.LightningModule):
    """
    Message-passing neural network for color prediction with spatial consistency.

    Key features:
    1. Initial feature extraction with position embeddings
    2. Multiple rounds of message passing between spatial neighbors
    3. Gated message aggregation for selective information flow
    4. Residual connections to preserve local information
    5. Color constraints based on available colors
    """

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 512,
        num_message_rounds: int = 6,
        num_classes: int = 10,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        temperature: float = 0.5,
        available_colors: Optional[List[int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_message_rounds = num_message_rounds
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature

        # Register available colors as buffer
        if available_colors is not None:
            colors_tensor = torch.zeros(num_classes)
            colors_tensor[available_colors] = 1.0
            self.register_buffer("available_colors", colors_tensor)
        else:
            self.register_buffer("available_colors", torch.ones(num_classes))

        # Initial feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Position embeddings
        self.pos_embedding = self._create_sinusoidal_embeddings()

        # Message passing layers
        self.message_layers = nn.ModuleList(
            [
                MessagePassingLayer(hidden_dim, dropout)
                for _ in range(num_message_rounds)
            ]
        )

        # Global context attention
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concatenated global context
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Temperature parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def _create_sinusoidal_embeddings(self) -> nn.Parameter:
        """Create sinusoidal position embeddings."""
        max_len = 30
        d_model = self.hidden_dim

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, max_len, d_model)
        for i in range(max_len):
            for j in range(max_len):
                # Encode row position
                pe[i, j, 0::4] = torch.sin(position[i] * div_term[0::2])
                pe[i, j, 1::4] = torch.cos(position[i] * div_term[0::2])
                # Encode column position
                pe[i, j, 2::4] = torch.sin(position[j] * div_term[1::2])
                pe[i, j, 3::4] = torch.cos(position[j] * div_term[1::2])

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]

        # Initial feature extraction
        h = self.feature_extractor(features)  # [B, 30, 30, hidden_dim]

        # Add position embeddings
        h = h + self.pos_embedding.unsqueeze(0)

        # Store initial features for residual connections
        h_init = h

        # Message passing rounds
        for i, message_layer in enumerate(self.message_layers):
            h = message_layer(h)
            # Add residual connection every 2 layers
            if (i + 1) % 2 == 0:
                h = h + h_init

        # Global context via self-attention
        # Reshape for attention: [B, 900, hidden_dim]
        h_flat = h.reshape(batch_size, -1, self.hidden_dim)
        h_global, _ = self.global_attention(h_flat, h_flat, h_flat)

        # Pool global features
        h_global_pooled = h_global.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        h_global_pooled = h_global_pooled.expand(-1, 900, -1)  # [B, 900, hidden_dim]

        # Concatenate local and global features
        h_combined = torch.cat(
            [h_flat, h_global_pooled], dim=-1
        )  # [B, 900, hidden_dim*2]

        # Reshape back to spatial
        h_combined = h_combined.reshape(batch_size, 30, 30, -1)

        # Final prediction
        logits = self.prediction_head(h_combined)  # [B, 30, 30, num_classes]

        # Apply temperature scaling
        temperature = torch.exp(self.log_temperature)
        logits = logits / temperature

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
        """Compute cross-entropy loss with spatial consistency regularization."""
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.num_classes)
        targets_flat = targets.reshape(-1)

        # Mask for valid pixels
        valid_mask = targets_flat != -1

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits_flat[valid_mask],
            targets_flat[valid_mask],
        )

        # Spatial consistency loss - penalize differences between neighboring predictions
        predictions = torch.argmax(logits, dim=-1)  # [B, 30, 30]

        # Compute differences with neighbors
        diff_right = torch.abs(
            predictions[:, :, :-1].float() - predictions[:, :, 1:].float()
        )
        diff_down = torch.abs(
            predictions[:, :-1, :].float() - predictions[:, 1:, :].float()
        )

        # Only consider differences where both pixels are valid
        valid_2d = targets != -1
        valid_right = valid_2d[:, :, :-1] & valid_2d[:, :, 1:]
        valid_down = valid_2d[:, :-1, :] & valid_2d[:, 1:, :]

        # Mask differences
        diff_right = diff_right * valid_right.float()
        diff_down = diff_down * valid_down.float()

        # Compute mean spatial consistency loss
        spatial_loss = (diff_right.sum() + diff_down.sum()) / (
            valid_right.sum() + valid_down.sum() + 1e-6
        )

        # Combined loss
        total_loss = ce_loss + 0.1 * spatial_loss

        return total_loss

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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
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


class MessagePassingLayer(nn.Module):
    """
    A single message passing layer that aggregates information from spatial neighbors.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gating mechanism for messages
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features of shape [B, 30, 30, hidden_dim]

        Returns:
            Updated node features of shape [B, 30, 30, hidden_dim]
        """
        batch_size = h.shape[0]

        # Pad the input for easier neighbor access
        h_padded = F.pad(h, (0, 0, 1, 1, 1, 1), mode="constant", value=0)

        # Collect messages from 8 neighbors
        messages = []
        gates = []

        # Define neighbor offsets (8-connected)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for di, dj in offsets:
            # Extract neighbor features
            neighbor = h_padded[:, 1 + di : 31 + di, 1 + dj : 31 + dj, :]

            # Concatenate center and neighbor features
            h_concat = torch.cat([h, neighbor], dim=-1)

            # Compute message
            message = self.message_mlp(h_concat)
            messages.append(message)

            # Compute gate
            gate = self.gate_mlp(h_concat)
            gates.append(gate)

        # Stack messages and gates
        messages = torch.stack(messages, dim=0)  # [8, B, 30, 30, hidden_dim]
        gates = torch.stack(gates, dim=0)  # [8, B, 30, 30, hidden_dim]

        # Apply gates to messages
        gated_messages = messages * gates

        # Aggregate messages (mean)
        aggregated = gated_messages.mean(dim=0)  # [B, 30, 30, hidden_dim]

        # Update node features
        h_updated = self.update_mlp(torch.cat([h, aggregated], dim=-1))

        # Residual connection
        h_updated = h + h_updated

        return h_updated


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
        description="Train message-passing model on single filename"
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
        "--max_epochs", type=int, default=300, help="Maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num_message_rounds",
        type=int,
        default=6,
        help="Number of message passing rounds",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for scaling"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="message_passing_checkpoints",
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
    model = MessagePassingColorModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_message_rounds=args.num_message_rounds,
        num_classes=10,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        available_colors=available_colors,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/checkpoints",
        filename="message_passing-{epoch:02d}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=50,
        verbose=True,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
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
