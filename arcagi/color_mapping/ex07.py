#!/usr/bin/env python3
"""
Neural CSP (Constraint Satisfaction Problem) for Color Assignment

This approach treats color prediction as a constraint satisfaction problem:
1. Extract constraints from edge features (same/different color relationships)
2. Initialize color beliefs using a neural network
3. Iteratively refine beliefs through message passing
4. Enforce consistency within connected components
5. Use filename embeddings to break symmetries

Key innovations:
- Explicit constraint graph construction from edge features
- Differentiable message passing for constraint propagation
- Soft-to-hard assignment with Gumbel-softmax
- Multi-stage training with teacher forcing
"""

import argparse
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


def create_position_encoding_sin_cos(max_len: int, embed_dim: int) -> torch.Tensor:
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


class EdgeFeatureEncoder(nn.Module):
    """Encodes edge features to extract pairwise constraints."""

    def __init__(self, edge_feature_dim: int = 147, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(
                64, 3
            ),  # Output: [same_color_logit, diff_color_logit, no_constraint_logit]
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: (B, H, W, 8, F) - features for 8-connected neighbors
        Returns:
            constraints: (B, H, W, 8, 3) - constraint logits per edge
        """
        B, H, W, N, F = edge_features.shape
        # Reshape to (B*H*W*N, F) for processing
        edge_features_flat = (
            edge_features.permute(0, 1, 2, 3, 4).contiguous().view(-1, F)
        )
        constraints_flat = self.encoder(edge_features_flat)
        # Reshape back to (B, H, W, N, 3)
        constraints = constraints_flat.view(B, H, W, N, 3)
        return constraints


class InitialBeliefNetwork(nn.Module):
    """Generates initial color beliefs from features and filename."""

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_colors: int = 10,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.pos_embed_dim = pos_embed_dim
        self.filename_embed_dim = filename_embed_dim

        # Position encodings
        self.pos_x_encoding = create_position_encoding_sin_cos(30, pos_embed_dim)
        self.pos_y_encoding = create_position_encoding_sin_cos(30, pos_embed_dim)

        # Process local features with convolutions
        input_dim = feature_dim + 2 * pos_embed_dim + filename_embed_dim
        self.local_encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 128, kernel_size=1),
            nn.ReLU(),
        )

        # Output color beliefs
        self.belief_head = nn.Conv2d(128, num_colors, kernel_size=1)

    def forward(
        self, features: torch.Tensor, filename_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, H, W, F)
            filename_embed: (B, D)
        Returns:
            beliefs: (B, H, W, C) - initial color beliefs
        """
        B, H, W, feat_dim = features.shape
        device = features.device

        # Add position encodings
        pos_x = self.pos_x_encoding[:W].unsqueeze(0).to(device)  # (1, W, D)
        pos_y = self.pos_y_encoding[:H].unsqueeze(1).to(device)  # (H, 1, D)
        pos_x = pos_x.expand(H, -1, -1)  # (H, W, D)
        pos_y = pos_y.expand(-1, W, -1)  # (H, W, D)

        # Expand filename embedding
        filename_expand = filename_embed.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)

        # Concatenate all inputs
        combined = torch.cat(
            [
                features,
                pos_x.unsqueeze(0).expand(B, -1, -1, -1),
                pos_y.unsqueeze(0).expand(B, -1, -1, -1),
                filename_expand,
            ],
            dim=-1,
        )  # (B, H, W, input_dim)

        # Convert to BCHW for convolutions
        combined = combined.permute(0, 3, 1, 2)

        # Process with convolutions
        encoded = self.local_encoder(combined)
        beliefs = self.belief_head(encoded)

        # Convert back to BHWC
        beliefs = beliefs.permute(0, 2, 3, 1)

        return beliefs


class ConstraintMessagePassing(nn.Module):
    """Iterative message passing to enforce constraints."""

    def __init__(
        self, num_colors: int = 10, hidden_dim: int = 64, filename_embed_dim: int = 32
    ):
        super().__init__()

        self.num_colors = num_colors

        # Message computation
        self.message_net = nn.Sequential(
            nn.Linear(num_colors * 2 + 3 + filename_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )

        # Belief update with residual connection
        self.update_net = nn.Sequential(
            nn.Linear(num_colors * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )

        # Layer norm for stability
        self.belief_norm = nn.LayerNorm(num_colors)

    def compute_messages(
        self,
        beliefs: torch.Tensor,
        constraints: torch.Tensor,
        filename_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages from neighbors based on constraints."""
        B, H, W, C = beliefs.shape
        device = beliefs.device

        # Pad beliefs for neighbor access
        beliefs_pad = F.pad(beliefs.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
        beliefs_pad = beliefs_pad.permute(0, 2, 3, 1)  # Back to BHWC

        # 8-connected neighbor offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        messages = torch.zeros(B, H, W, 8, C, device=device)

        for idx, (dy, dx) in enumerate(offsets):
            # Get neighbor beliefs
            neighbor_beliefs = beliefs_pad[
                :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W, :
            ]

            # Get constraint for this edge
            edge_constraint = constraints[:, :, :, idx, :]  # (B, H, W, 3)

            # Expand filename embedding
            filename_expand = (
                filename_embed.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
            )

            # Compute message
            message_input = torch.cat(
                [
                    beliefs.contiguous().reshape(B * H * W, C),
                    neighbor_beliefs.contiguous().reshape(B * H * W, C),
                    edge_constraint.contiguous().reshape(B * H * W, 3),
                    filename_expand.contiguous().reshape(B * H * W, -1),
                ],
                dim=-1,
            )

            message = self.message_net(message_input)
            messages[:, :, :, idx, :] = message.contiguous().reshape(B, H, W, C)

        return messages

    def forward(
        self,
        beliefs: torch.Tensor,
        constraints: torch.Tensor,
        filename_embed: torch.Tensor,
    ) -> torch.Tensor:
        """One iteration of message passing."""
        B, H, W, C = beliefs.shape

        # Compute messages from all neighbors
        messages = self.compute_messages(beliefs, constraints, filename_embed)

        # Aggregate messages
        aggregated = messages.mean(dim=3)  # (B, H, W, C)

        # Update beliefs with residual connection
        update_input = torch.cat([beliefs, aggregated], dim=-1)
        update_input_flat = update_input.contiguous().reshape(B * H * W, -1)

        belief_update = self.update_net(update_input_flat)
        belief_update = belief_update.contiguous().reshape(B, H, W, C)

        # Residual connection and normalization
        new_beliefs = beliefs + 0.5 * belief_update
        new_beliefs = self.belief_norm(new_beliefs)

        return new_beliefs


class GlobalConsistencyModule(nn.Module):
    """Enforces consistency within connected components."""

    def __init__(self, num_colors: int = 10, filename_embed_dim: int = 32):
        super().__init__()

        self.num_colors = num_colors

        # Network to select canonical color for component
        self.canonical_selector = nn.Sequential(
            nn.Linear(num_colors + filename_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_colors),
        )

    def find_connected_components(self, constraints: torch.Tensor) -> torch.Tensor:
        """Find connected components based on same-color constraints."""
        # Simplified: Return pixel indices as their own component
        # In full implementation, would do proper connected component analysis
        B, H, W, _, _ = constraints.shape
        components = torch.arange(H * W, device=constraints.device)
        components = components.reshape(1, H, W).expand(B, -1, -1)
        return components

    def forward(
        self,
        beliefs: torch.Tensor,
        constraints: torch.Tensor,
        filename_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce consistency within components."""
        B, H, W, C = beliefs.shape

        # Find connected components
        components = self.find_connected_components(constraints)

        # For now, just return beliefs with slight smoothing
        # In full implementation, would aggregate beliefs per component
        # and enforce consistency
        kernel = torch.ones(1, 1, 3, 3, device=beliefs.device) / 9.0
        beliefs_smooth = F.conv2d(
            beliefs.permute(0, 3, 1, 2),
            kernel.expand(C, -1, -1, -1),
            padding=1,
            groups=C,
        )
        beliefs_smooth = beliefs_smooth.permute(0, 2, 3, 1)

        return 0.9 * beliefs + 0.1 * beliefs_smooth


class NeuralCSPColorModel(pl.LightningModule):
    """Main model combining all components."""

    def __init__(
        self,
        feature_dim: int = 147,
        pos_embed_dim: int = 16,
        filename_embed_dim: int = 32,
        num_filenames: int = 800,
        num_iterations: int = 5,
        lr: float = 1e-3,
        constraint_weight: float = 0.1,
        consistency_weight: float = 0.05,
        teacher_forcing_prob: float = 0.3,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Model components
        self.edge_encoder = EdgeFeatureEncoder(feature_dim)
        self.filename_embedding = nn.Embedding(num_filenames, filename_embed_dim)
        self.initial_beliefs = InitialBeliefNetwork(
            feature_dim, pos_embed_dim, filename_embed_dim
        )
        self.message_passing = ConstraintMessagePassing(
            num_colors=10, filename_embed_dim=filename_embed_dim
        )
        self.global_consistency = GlobalConsistencyModule(
            num_colors=10, filename_embed_dim=filename_embed_dim
        )

        # Training parameters
        self.num_iterations = num_iterations
        self.lr = lr
        self.constraint_weight = constraint_weight
        self.consistency_weight = consistency_weight
        self.teacher_forcing_prob = teacher_forcing_prob

        # Loss weights (inverse frequency)
        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, 11.6, 15.7, 11.3, 23.7, 13.4, 55.2, 45.5, 21.0, 39.6]),
        )

    def extract_edge_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract edge features for 8-connected neighbors."""
        B, H, W, feat_dim = features.shape
        device = features.device

        # Pad features
        features_pad = F.pad(
            features.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate"
        )
        features_pad = features_pad.permute(0, 2, 3, 1)

        # Extract neighbor features
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        edge_features = torch.zeros(B, H, W, 8, feat_dim, device=device)

        for idx, (dy, dx) in enumerate(offsets):
            # For now, use difference between features as edge features
            # In full implementation, might use more sophisticated edge features
            neighbor_feat = features_pad[:, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W, :]
            edge_features[:, :, :, idx, :] = torch.abs(features - neighbor_feat)

        return edge_features

    def forward(
        self,
        features: torch.Tensor,
        filename_ids: torch.Tensor,
        teacher_colors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional teacher forcing."""
        B, H, W, feat_dim = features.shape

        # Get filename embeddings
        filename_embed = self.filename_embedding(filename_ids)

        # Extract edge features and constraints
        # TEMPORARILY SKIP edge features to debug
        # edge_features = self.extract_edge_features(features)
        # constraints = self.edge_encoder(edge_features)
        # constraints_prob = F.softmax(constraints, dim=-1)

        # Initialize beliefs
        beliefs = self.initial_beliefs(features, filename_embed)

        # Store intermediate beliefs for loss computation
        all_beliefs = [beliefs]

        # TEMPORARILY SKIP message passing to debug
        # Iterative message passing
        # for t in range(self.num_iterations):
        #     beliefs = self.message_passing(beliefs, constraints_prob, filename_embed)
        #     all_beliefs.append(beliefs)

        # Create dummy constraints for now
        constraints_prob = torch.zeros(B, H, W, 8, 3, device=features.device)
        constraints_prob[:, :, :, :, 2] = 1.0  # All "no constraint"

        return {
            "beliefs": beliefs,
            "all_beliefs": all_beliefs,
            "constraints": constraints_prob,
        }

    def compute_constraint_violation_loss(
        self, beliefs: torch.Tensor, constraints: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for constraint violations."""
        B, H, W, C = beliefs.shape
        device = beliefs.device

        # Pad beliefs
        beliefs_pad = F.pad(beliefs.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
        beliefs_pad = beliefs_pad.permute(0, 2, 3, 1)

        # Compute violation for each edge
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        total_violation = torch.tensor(0.0, device=device)

        for idx, (dy, dx) in enumerate(offsets):
            neighbor_beliefs = beliefs_pad[
                :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W, :
            ]

            # Get constraint probabilities
            same_prob = constraints[:, :, :, idx, 0]
            diff_prob = constraints[:, :, :, idx, 1]

            # Compute agreement (dot product of belief distributions)
            agreement = (beliefs * neighbor_beliefs).sum(dim=-1)

            # Violation: same constraint wants high agreement, diff wants low
            violation = same_prob * (1 - agreement) + diff_prob * agreement
            total_violation = total_violation + violation.mean()

        return total_violation / len(offsets)

    def compute_consistency_loss(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Compute loss for belief consistency."""
        # Compute entropy of beliefs (lower entropy = more confident)
        entropy = -(beliefs * (beliefs + 1e-10).log()).sum(dim=-1)
        return entropy.mean()

    def training_step(self, batch, batch_idx):
        features, filename_ids, colors = batch

        # Forward pass with teacher forcing
        output = self(features, filename_ids, teacher_colors=colors)
        beliefs = output["beliefs"]
        all_beliefs = output["all_beliefs"]
        constraints = output["constraints"]

        # Color prediction loss (on final beliefs)
        color_logits = beliefs
        color_loss = F.cross_entropy(
            color_logits.contiguous().reshape(-1, 10),
            colors.contiguous().reshape(-1),
            weight=self.class_weights,
            ignore_index=-1,
        )

        # Constraint violation loss
        constraint_loss = self.compute_constraint_violation_loss(beliefs, constraints)

        # Consistency loss
        consistency_loss = self.compute_consistency_loss(beliefs)

        # Total loss
        loss = (
            color_loss
            + self.constraint_weight * constraint_loss
            + self.consistency_weight * consistency_loss
        )

        # Compute accuracy
        predictions = beliefs.argmax(dim=-1)
        mask = colors != -1
        correct = (predictions == colors) & mask
        accuracy = correct.float().sum() / mask.float().sum()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_color_loss", color_loss)
        self.log("train_constraint_loss", constraint_loss)
        self.log("train_consistency_loss", consistency_loss)
        self.log("train_acc", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, filename_ids, colors = batch

        # Forward pass without teacher forcing
        output = self(features, filename_ids, teacher_colors=None)
        beliefs = output["beliefs"]
        constraints = output["constraints"]

        # Losses
        color_logits = beliefs
        color_loss = F.cross_entropy(
            color_logits.contiguous().reshape(-1, 10),
            colors.contiguous().reshape(-1),
            weight=self.class_weights,
            ignore_index=-1,
        )

        constraint_loss = self.compute_constraint_violation_loss(beliefs, constraints)
        consistency_loss = self.compute_consistency_loss(beliefs)

        loss = (
            color_loss
            + self.constraint_weight * constraint_loss
            + self.consistency_weight * consistency_loss
        )

        # Accuracy
        predictions = beliefs.argmax(dim=-1)
        mask = colors != -1
        correct = (predictions == colors) & mask
        accuracy = correct.float().sum() / mask.float().sum()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_color_loss", color_loss)
        self.log("val_constraint_loss", constraint_loss)
        self.log("val_consistency_loss", consistency_loss)
        self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def load_data(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed data from NPZ file."""
    print(f"Loading data from {data_path}")
    data = np.load(data_path)

    # Load features and colors
    features = data["inputs_features"]  # Shape: (N, 30, 30, 147)
    colors = data["inputs"]  # Shape: (N, 30, 30)
    filenames = data["filenames"]

    # Create filename to index mapping
    unique_filenames = np.unique(filenames)
    filename_to_idx = {fn: idx for idx, fn in enumerate(unique_filenames)}

    # Convert filenames to indices
    filename_indices = np.array([filename_to_idx[f] for f in filenames])

    print(f"Loaded {len(features)} examples from {len(unique_filenames)} unique files")
    print(f"Feature shape: {features.shape}, Colors shape: {colors.shape}")

    return features, colors, filename_indices, unique_filenames, filename_to_idx


def main():
    parser = argparse.ArgumentParser(description="Neural CSP for color mapping")
    parser.add_argument(
        "--train_path",
        type=str,
        default="../../processed_data/train_all.npz",
        help="Path to training data NPZ file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="../../processed_data/val_all.npz",
        help="Path to validation data NPZ file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of message passing iterations",
    )
    parser.add_argument(
        "--constraint_weight",
        type=float,
        default=0.1,
        help="Weight for constraint violation loss",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.05,
        help="Weight for consistency loss",
    )
    parser.add_argument(
        "--teacher_forcing_prob",
        type=float,
        default=0.3,
        help="Probability of teacher forcing during training",
    )
    parser.add_argument(
        "--use_train_as_eval",
        action="store_true",
        help="Use training data for validation (overfitting test)",
    )

    args = parser.parse_args()

    # Load data
    train_features, train_colors, train_filename_ids, _, _ = load_data(args.train_path)

    if args.use_train_as_eval:
        val_features, val_colors, val_filename_ids = (
            train_features,
            train_colors,
            train_filename_ids,
        )
        print("Using training data for validation (overfitting test)")
    else:
        val_features, val_colors, val_filename_ids, _, _ = load_data(args.val_path)

    # Get number of unique filenames across train and val
    all_filename_ids = np.concatenate([train_filename_ids, val_filename_ids])
    num_filenames = len(np.unique(all_filename_ids))
    print(f"Total unique filenames: {num_filenames}")

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_filename_ids).long(),
        torch.from_numpy(train_colors).long(),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_features).float(),
        torch.from_numpy(val_filename_ids).long(),
        torch.from_numpy(val_colors).long(),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    model = NeuralCSPColorModel(
        num_filenames=num_filenames,
        num_iterations=args.num_iterations,
        lr=args.lr,
        constraint_weight=args.constraint_weight,
        consistency_weight=args.consistency_weight,
        teacher_forcing_prob=args.teacher_forcing_prob,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="neural_csp_checkpoints/checkpoints",
        filename="neural_csp-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test on validation set
    print("\nFinal validation results:")
    trainer.validate(model, val_loader)


if __name__ == "__main__":
    main()
