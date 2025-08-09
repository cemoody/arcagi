import os

# Import from parent module
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pydanclick import from_pydantic
from pydantic import BaseModel
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
# Import data loading functions
from data_loader import create_dataloader, prepare_dataset
from models import Batch, BatchData
from utils.metrics import training_index_metrics
from utils.terminal_imshow import imshow

# Import Order2Features from lib
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lib.order2 import Order2Features


class TrainingConfig(BaseModel):
    """Configuration for training the order2-based neural cellular automata model."""

    # Data and checkpoint paths
    data_dir: str = "/tmp/arc_data"
    checkpoint_dir: str = "order2_checkpoints"

    # Training parameters
    max_epochs: int = 2000
    lr: float = 1e-2
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    num_message_rounds: int = 24

    # Self-healing noise parameters
    enable_self_healing: bool = True
    death_prob: float = 0.02
    gaussian_std: float = 0.05
    salt_pepper_prob: float = 0.01
    spatial_corruption_prob: float = 0.01

    # Model parameters
    dropout: float = 0.1
    temperature: float = 1.0
    filename: str = "3345333e"

    # Early stopping parameters
    patience: int = 100
    min_epochs: int = 500

    # Training mode
    single_file_mode: bool = True

    # Noise parameters
    noise_prob: float = 0.10
    num_final_steps: int = 6


def apply_binary_noise(
    features: torch.Tensor, noise_prob: float, training: bool
) -> torch.Tensor:
    """Apply binary noise to features during training only."""
    if training and noise_prob > 0:
        # For binary features (0/1), flip them with noise_prob
        noise_mask = torch.rand_like(features) < noise_prob
        # Flip binary features: 0->1, 1->0
        features = torch.where(noise_mask, 1 - features, features)

    return features


def apply_color_constraints(
    logits: torch.Tensor,
    available_colors: torch.Tensor | List[int] | None = None,
    num_classes: int = 10,
) -> torch.Tensor:
    """Apply hard constraints to ensure only available colors are predicted."""
    if available_colors is not None:
        # Convert list to tensor if needed
        if isinstance(available_colors, list):
            available_colors = torch.tensor(available_colors, device=logits.device)

        # Assert that available_colors contains integer types
        assert available_colors.dtype in [
            torch.int,
            torch.long,
            torch.int32,
            torch.int64,
        ], f"available_colors must have integer dtype, got {available_colors.dtype}"
        # Check max color is within valid range
        max_color = available_colors.max() if len(available_colors) > 0 else -1
        assert (
            max_color < num_classes
        ), f"Maximum color index {max_color} must be less than num_classes {num_classes}"
        mask = torch.ones(num_classes, device=logits.device) * -1e10
        mask[available_colors] = 0
        logits = logits + mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    return logits


def batch_to_dataclass(
    batch: Tuple[torch.Tensor, ...],
) -> BatchData:
    inputs_one_hot, outputs_one_hot, input_features, output_features, indices = batch

    # Ensure features are float tensors for MPS compatibility
    input_features = input_features.float()
    output_features = output_features.float()

    # Extract colors and masks from the grid data
    input_colors = inputs_one_hot.argmax(dim=-1).long()
    output_colors = outputs_one_hot.argmax(dim=-1).long()

    # Change any colors >= 10 to -1 (mask)
    input_colors = torch.where(input_colors >= 10, -1, input_colors)
    output_colors = torch.where(output_colors >= 10, -1, output_colors)

    # Create masks (True where valid colors exist)
    input_masks = input_colors >= 0
    output_masks = output_colors >= 0

    # Get example indices for this batch
    input_colors_flat = input_colors.reshape(-1)
    output_colors_flat = output_colors.reshape(-1)
    input_masks_flat = input_masks.reshape(-1)
    output_masks_flat = output_masks.reshape(-1)

    inp = Batch(
        one=inputs_one_hot,
        fea=input_features,
        col=input_colors,
        msk=input_masks,
        colf=input_colors_flat,
        mskf=input_masks_flat.float(),
        idx=indices,
    )

    out = Batch(
        one=outputs_one_hot,
        fea=output_features,
        col=output_colors,
        msk=output_masks,
        colf=output_colors_flat,
        mskf=output_masks_flat.float(),
        idx=indices,
    )

    return BatchData(inp=inp, out=out)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm for recurrent architectures like message-passing networks.
    Used in modern LLMs like Llama for better performance.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization: x / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class Order2ToColorDecoder(nn.Module):
    """
    Decoder network that converts order2 features back to color predictions.

    This network learns to invert the order2 feature extraction process.
    """

    def __init__(
        self, hidden_dim: int = 256, num_classes: int = 10, dropout: float = 0.1
    ):
        super(Order2ToColorDecoder, self).__init__()

        # Input: 44 order2 features
        # Hidden layers to process and decode
        self.decoder = nn.Sequential(
            nn.Linear(44, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            RMSNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for colors and mask
        self.color_head = nn.Linear(hidden_dim // 2, num_classes)
        self.mask_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self, order2_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            order2_features: [B, 30, 30, 44] binary features

        Returns:
            color_logits: [B, 30, 30, num_classes]
            mask_logits: [B, 30, 30, 1]
        """
        # Process through decoder
        h = self.decoder(order2_features)

        # Get color and mask predictions
        color_logits = self.color_head(h)
        mask_logits = self.mask_head(h)

        return color_logits, mask_logits


class Order2MemorizationModel(pl.LightningModule):
    """
    Order2-based model for perfect memorization.

    This model processes colors through order2 features:
    1. Convert colors to order2 features (44 binary features)
    2. Process order2 features through Neural Cellular Automata
    3. Decode order2 features back to colors and masks

    Key differences from ex29:
    - Uses order2 features as intermediate representation
    - NCA operates on 44-channel binary features
    - Includes a learned decoder from order2 back to colors
    """

    training_index_metrics: Dict[int, Dict[str, int]] = {}
    validation_index_metrics: Dict[int, Dict[str, int]] = {}
    force_visualize: bool = False

    def __init__(
        self,
        feature_dim: int = 147,  # Not used in this version
        hidden_dim: int = 64,
        num_message_rounds: int = 6,
        num_classes: int = 10,
        dropout: float = 0.0,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        temperature: float = 0.01,
        filename: str = "3345333e",
        num_train_examples: int = 2,
        dim_nca: int = 64,  # Increased for 44-channel input
        decoder_hidden_dim: int = 64,
        # Self-healing noise parameters
        enable_self_healing: bool = True,
        death_prob: float = 0.02,
        gaussian_std: float = 0.05,
        salt_pepper_prob: float = 0.01,
        spatial_corruption_prob: float = 0.01,
        num_final_steps: int = 12,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_message_rounds = num_message_rounds
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.filename = filename
        self.num_train_examples = num_train_examples
        self.decoder_hidden_dim = decoder_hidden_dim

        # Track validation metrics across entire epoch
        self.validation_outputs = []

        # Track per-index metrics
        self.index_metrics = {}
        self.training_index_metrics = {}

        # Order2 feature extractor
        self.order2_encoder = Order2Features()

        # Feature processing layers
        # Convert 44 binary features to hidden dimension
        self.feature_processor = nn.Sequential(
            nn.Linear(44, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            # Removed redundant second normalization layer
        )

        # Binary noise layer for training-time regularization
        self.noise_prob = 0.05  # 5% probability of flipping each feature

        # Project to NCA dimension
        self.linear_1 = nn.Linear(hidden_dim, dim_nca)
        self.linear_2 = nn.Linear(dim_nca, hidden_dim)

        # Neural Cellular Automata for spatial refinement
        self.nca = NeuralCellularAutomata(
            dim_nca,
            dropout=dropout,
            enable_self_healing=enable_self_healing,
            death_prob=death_prob,
            gaussian_std=gaussian_std,
            salt_pepper_prob=salt_pepper_prob,
            spatial_corruption_prob=spatial_corruption_prob,
            num_final_steps=num_final_steps,
        )

        # Spatial message passing
        self.message_passing = SpatialMessagePassing(hidden_dim, dropout=dropout)

        # Order2 to color decoder
        self.order2_decoder = Order2ToColorDecoder(
            hidden_dim=decoder_hidden_dim, num_classes=num_classes, dropout=dropout
        )

        # Projection layer to reconstruct order2 features
        self.order2_projection = nn.Linear(hidden_dim, 44)

        # Store available colors
        self.available_colors: Optional[List[int]] = None

        self.drop = nn.Dropout(dropout)

        # Better initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        colors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass processing colors through order2 features.

        Args:
            colors: [B, 30, 30] integer color values (-1 for mask)

        Returns:
            color_logits: [B, 30, 30, num_classes]
            mask_logits: [B, 30, 30, 1]
        """
        # Step 1: Convert colors to order2 features
        with torch.no_grad():
            order2_features = self.order2_encoder(colors)  # [B, 30, 30, 44]

            # Apply binary noise during training
            order2_features = apply_binary_noise(
                order2_features, noise_prob=self.noise_prob, training=self.training
            )

        # Step 2: Process order2 features
        h = self.feature_processor(order2_features)  # [B, 30, 30, hidden_dim]

        # Step 3: Hybrid NCA and Message Passing processing
        noisy_steps = self.num_message_rounds - self.nca.num_final_steps
        for i in range(max(0, noisy_steps + self.nca.num_final_steps)):
            apply_noise = i < noisy_steps
            p = self.dropout if apply_noise else 0
            # Project to NCA dimension (but keep residual)
            x_nca = self.linear_1(h)
            if apply_noise:
                x_nca = self.drop(x_nca)

            # Neural Cellular Automata with residual
            x_nca = x_nca + self.nca(x_nca, apply_noise=apply_noise)

            # Project back and add as residual to h
            h_update = self.linear_2(x_nca)
            if apply_noise:
                h_update = self.drop(h_update)
            h = h + h_update  # Small weight on NCA update

            # Spatial message passing on current h
            h_messages = self.message_passing(h)
            h = h + h_messages  # Residual connection for messages

        # Step 4: Reconstruct order2 features from processed hidden state
        signal = self.order2_projection(h)
        color_logits, mask_logits = self.order2_decoder(signal)
        color_logits = color_logits / self.temperature
        color_logits = apply_color_constraints(color_logits, self.available_colors)
        return color_logits, mask_logits

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="val")

    def step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        step_type: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        i = batch_to_dataclass(batch)

        # Compute losses for color predictions
        losses: List[torch.Tensor] = []
        metrics: Dict[int, Dict[str, float]] = {}
        for prefix, ex in [("input", i.inp), ("output", i.out)]:
            # Use color grids instead of features
            col_log, msk_log = self(ex.col)
            col_logf = col_log.reshape(-1, col_log.size(-1))

            # Filter out -1 targets (invalid/masked pixels)
            ex_mask = ex.colf != -1
            col_loss = F.cross_entropy(col_logf[ex_mask], ex.colf[ex_mask])

            msk_logf = msk_log.reshape(-1)
            # Also filter mask loss for consistency
            msk_loss = F.binary_cross_entropy_with_logits(
                msk_logf[ex_mask], ex.mskf[ex_mask]
            )
            losses.append(col_loss + msk_loss)
            # Cast prefix to Literal type for type checker
            if prefix == "input":
                training_index_metrics(i, col_log, msk_log, "input", metrics)
            else:  # prefix == "output"
                training_index_metrics(i, col_log, msk_log, "output", metrics)

            # Visualize every 10 epochs
            if self.force_visualize or (
                self.current_epoch % 10 == 0 and batch_idx == 0
            ):
                self.visualize_predictions(
                    i,
                    ex.col,
                    col_log,
                    f"{prefix}-{step_type}",
                )
        # Convert float metrics to int for compatibility
        int_metrics: Dict[int, Dict[str, int]] = {}
        for idx, m in metrics.items():
            int_metrics[idx] = {k: int(v) for k, v in m.items()}

        if step_type == "train":
            self.training_index_metrics = int_metrics
        else:
            self.validation_index_metrics = int_metrics

        # Combined loss
        total_loss = losses[0] + losses[1]

        # Only log metrics during training or validation, not during test evaluation
        try:
            self.log_metrics(int_metrics, step_type)
            self.log(f"{step_type}_total_loss", total_loss)  # type: ignore
        except Exception:
            # Silent fail during test evaluation when logging is not supported
            pass

        return total_loss

    def log_metrics(
        self, metrics: Dict[int, Dict[str, int]], prefix: str = "train"
    ) -> None:
        for idx in metrics.keys():
            for key, value in metrics[idx].items():
                self.log(f"{prefix}_{key}", value)  # type: ignore

    def on_train_epoch_end(self) -> None:
        """Print training per-index accuracy at the end of each epoch."""
        self.epoch_end("train")

    def on_train_epoch_start(self) -> None:
        # logger.info(f"Training epoch start: {self.current_epoch}")
        pass

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at the start of each epoch."""
        self.validation_index_metrics = {}

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level validation metrics."""
        self.epoch_end("val")

        # Compute epoch-level metrics from validation step outputs
        val_loss = self.trainer.callback_metrics.get("val_total_loss", 0.0)

        # Calculate color and mask accuracy from validation metrics
        val_input_pixels_incorrect = self.trainer.callback_metrics.get(
            "val_input_pixels_incorrect", 0.0
        )
        val_output_pixels_incorrect = self.trainer.callback_metrics.get(
            "val_output_pixels_incorrect", 0.0
        )
        val_input_pixels_per_index = self.trainer.callback_metrics.get(
            "val_input_pixels_per_index", 1.0
        )
        val_output_pixels_per_index = self.trainer.callback_metrics.get(
            "val_output_pixels_per_index", 1.0
        )

        # Calculate accuracy as (1 - error_rate)
        # Handle both tensor and float cases
        if isinstance(val_input_pixels_per_index, torch.Tensor):
            val_input_pixels_per_index = val_input_pixels_per_index.item()
        if isinstance(val_output_pixels_per_index, torch.Tensor):
            val_output_pixels_per_index = val_output_pixels_per_index.item()
        if isinstance(val_input_pixels_incorrect, torch.Tensor):
            val_input_pixels_incorrect = val_input_pixels_incorrect.item()
        if isinstance(val_output_pixels_incorrect, torch.Tensor):
            val_output_pixels_incorrect = val_output_pixels_incorrect.item()

        val_input_color_acc = 1.0 - (
            val_input_pixels_incorrect / max(val_input_pixels_per_index, 1.0)
        )
        val_output_color_acc = 1.0 - (
            val_output_pixels_incorrect / max(val_output_pixels_per_index, 1.0)
        )

        # For now, use color accuracy for mask accuracy
        val_input_mask_acc = val_input_color_acc
        val_output_mask_acc = val_output_color_acc

        # Log epoch-level metrics that ModelCheckpoint expects
        self.log("val_epoch_loss", val_loss, prog_bar=True)
        self.log(
            "val_epoch_color_acc", val_output_color_acc, prog_bar=True
        )  # Main metric for checkpointing
        self.log("val_epoch_mask_acc", val_output_mask_acc, prog_bar=True)
        self.log("val_input_color_acc", val_input_color_acc)
        self.log("val_output_color_acc", val_output_color_acc)
        self.log("val_input_mask_acc", val_input_mask_acc)
        self.log("val_output_mask_acc", val_output_mask_acc)

    def epoch_end(self, step_type: Literal["train", "val"]) -> None:
        if step_type == "train":
            metrics = self.training_index_metrics
        else:
            metrics = self.validation_index_metrics

        if metrics:
            print(
                f"\n{step_type.upper()} Per-Index Accuracy (Epoch {self.current_epoch}):"
            )
            print(
                f"{'Index':<8} {'Output Pixels':<12} {'Output Mask':<12} {'All Perfect':<12}"
            )
            print("-" * 50)

            for idx in sorted(self.training_index_metrics.keys()):
                metrics = self.training_index_metrics[idx]
                out_pix_incor = metrics["output_n_incorrect_num_color"]
                out_msk_incor = metrics["output_n_incorrect_num_mask"]

                all_perfect = "✓" if out_pix_incor == 0 else "✗"
                print(
                    f"{idx:<8} {out_pix_incor:<12} {out_msk_incor:<12} {all_perfect:<12}"
                )

        # Clear training metrics for next epoch
        self.training_index_metrics = {}

    def configure_optimizers(self) -> Dict[str, Any]:
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate schedule
        def lr_lambda(epoch: int) -> float:
            if epoch < 2:
                return (epoch + 1) / 2  # Quick warmup
            elif epoch < 30:
                return 1.0  # Full LR
            elif epoch < 50:
                return 0.5  # Half LR
            elif epoch < 500:
                return 0.1  # Half LR
            elif epoch < 1000:
                return 0.01  # Half LR
            else:
                return 0.001  # Low LR for fine-tuning

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def visualize_predictions(
        self,
        i: BatchData,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        prefix: str,
    ) -> None:
        """Visualize predictions vs ground truth for training."""
        # Get the first unique index from the batch
        _, inverse_indices = torch.unique(i.inp.idx, return_inverse=True)  # type: ignore

        for idx in inverse_indices:
            pred_colors = predictions[idx].argmax(dim=-1).cpu()
            true_colors = targets[idx].cpu()

            # Create visualization
            print(f"\n{'='*60}")
            print(f"{prefix.upper()} - Epoch {self.current_epoch}")
            print(f"{'='*60}")

            # Show ground truth
            print("\nGround Truth colors:")
            imshow(true_colors, title=None, show_legend=True)

            # Show predictions
            print("\nPredicted colors:")
            correct = (true_colors == pred_colors) | (true_colors == -1)
            imshow(pred_colors, title=None, show_legend=True, correct=correct)

            # Calculate accuracy for this example
            valid_mask = true_colors != -1
            if valid_mask.any():
                accuracy = (
                    (pred_colors[valid_mask] == true_colors[valid_mask]).float().mean()
                )

                # Also show mask predictions vs ground truth
                print("\nMask predictions (True=active pixel, False=inactive):")
                print(f"\nColor Accuracy: {accuracy:.2%}")


class SpatialMessagePassing(nn.Module):
    """Efficient spatial message passing for local consistency."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super(SpatialMessagePassing, self).__init__()
        self.hidden_dim = hidden_dim

        # Simple 3x3 convolution for message passing
        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim // 16,  # Group convolution for efficiency
        )

        self.norm = RMSNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, 30, 30, hidden_dim]
        # Convert to conv format
        h_conv = h.permute(0, 3, 1, 2)  # [B, hidden_dim, 30, 30]

        # Message passing
        messages = self.conv(h_conv)

        messages = messages.permute(0, 2, 3, 1)  # [B, 30, 30, hidden_dim]

        # Apply layer norm and activation
        messages = self.norm(messages)
        messages = self.activation(messages)
        messages = self.dropout(messages)

        return messages


class SelfHealingNoise(nn.Module):
    """
    Noise module for testing NCA self-healing capabilities.
    """

    def __init__(
        self,
        death_prob: float = 0.05,
        gaussian_std: float = 0.1,
        salt_pepper_prob: float = 0.02,
        spatial_corruption_prob: float = 0.03,
    ):
        super(SelfHealingNoise, self).__init__()
        self.death_prob = death_prob
        self.gaussian_std = gaussian_std
        self.salt_pepper_prob = salt_pepper_prob
        self.spatial_corruption_prob = spatial_corruption_prob

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply noise during training only.
        h: [B, 30, 30, hidden_dim]
        """
        if not self.training:
            return h

        B, H, W, _ = h.shape

        # 1. Cell Death Noise - randomly "kill" cells by zeroing them
        if self.death_prob > 0:
            death_mask = torch.rand(B, H, W, device=h.device) > self.death_prob
            h = h * death_mask.unsqueeze(-1)

        # 2. Gaussian Feature Noise - simulate metabolic fluctuations
        if self.gaussian_std > 0:
            noise = torch.randn_like(h) * self.gaussian_std
            h = h + noise

        # 3. Salt & Pepper Noise - random extreme values
        if self.salt_pepper_prob > 0:
            salt_pepper_mask = torch.rand_like(h) < self.salt_pepper_prob
            # Half salt (max value), half pepper (min value)
            salt_mask = torch.rand_like(h) > 0.5
            extreme_values = torch.where(salt_mask, 1.0, -1.0)
            h = torch.where(salt_pepper_mask, extreme_values, h)

        # 4. Spatial Corruption - damage random 3x3 regions
        if self.spatial_corruption_prob > 0:
            for b in range(B):
                if torch.rand(1).item() < self.spatial_corruption_prob:
                    # Random 3x3 corruption location
                    y = torch.randint(0, H - 2, (1,)).item()
                    x = torch.randint(0, W - 2, (1,)).item()
                    # Zero out the 3x3 region
                    h[b, y : y + 3, x : x + 3, :] = 0

        return h


class NeuralCellularAutomata(nn.Module):
    """
    Neural Cellular Automata for spatial pattern refinement on order2 features.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        enable_self_healing: bool = True,
        death_prob: float = 0.02,
        gaussian_std: float = 0.05,
        salt_pepper_prob: float = 0.01,
        spatial_corruption_prob: float = 0.01,
        num_final_steps: int = 12,
    ):
        super(NeuralCellularAutomata, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_final_steps = num_final_steps

        # Learnable perception filters
        self.perception_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim * 3,  # 3 filters per input channel
            kernel_size=3,
            padding=1,
            groups=hidden_dim,  # Each input channel gets its own set of 3 filters
        )

        # Self-healing noise for robustness testing
        self.self_healing_noise = (
            SelfHealingNoise(
                death_prob=death_prob,
                gaussian_std=gaussian_std,
                salt_pepper_prob=salt_pepper_prob,
                spatial_corruption_prob=spatial_corruption_prob,
            )
            if enable_self_healing
            else None
        )

        # Initialize with meaningful patterns
        with torch.no_grad():
            # Initialize first filter as identity-like (center detection)
            self.perception_conv.weight[0::3, :, 1, 1] = 1.0
            # Initialize second filter as horizontal edge-like
            self.perception_conv.weight[1::3, :, :, 0] = -0.5
            self.perception_conv.weight[1::3, :, :, 2] = 0.5
            # Initialize third filter as vertical edge-like
            self.perception_conv.weight[2::3, :, 0, :] = -0.5
            self.perception_conv.weight[2::3, :, 2, :] = 0.5

        # Update network: takes perceived features -> new state
        perception_dim = hidden_dim * 3  # 3 filters per channel
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_dim, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
        )

        # Stochastic update (only fraction of cells update each step)
        self.update_probability = 0.5

    def stochastic_update(self, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic updates - only some cells update each step.
        """
        if not self.training:
            # During inference, always update
            return x + dx

        # Create random mask for stochastic updates
        update_mask = (
            torch.rand_like(x[:, :1, :, :]) < self.update_probability
        ).float()
        return x + dx * update_mask

    def forward(self, h: torch.Tensor, apply_noise: bool = True) -> torch.Tensor:
        """
        Single NCA step.
        h: [B, 30, 30, hidden_dim] -> [B, 30, 30, hidden_dim]
        apply_noise: Whether to apply self-healing noise
        """
        # Apply self-healing noise at the beginning (tests robustness)
        if apply_noise and self.self_healing_noise is not None:
            h = self.self_healing_noise(h)

        # Convert to conv format: [B, C, H, W]
        x = h.permute(0, 3, 1, 2).contiguous()

        # Step 1: Perceive local neighborhood
        perceived = self.perception_conv(x)

        # Step 2: Compute state update
        dx = self.update_net(perceived)

        # Step 3: Stochastic update
        x_new = self.stochastic_update(x, dx)

        # Convert back to original format: [B, H, W, C]
        h_new = x_new.permute(0, 2, 3, 1).contiguous()

        return h_new


class PerfectAccuracyEarlyStopping(Callback):
    """Custom callback that stops training when both INPUT and OUTPUT achieve 100% accuracy for patience epochs."""

    def __init__(self, patience: int = 15, min_epochs: int = 500):
        super().__init__()
        self.patience = patience
        self.min_epochs = min_epochs
        self.perfect_epochs = 0
        self.best_epoch = -1

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # Get the current epoch metrics
        input_color_acc = trainer.callback_metrics.get("val_input_color_acc", 0.0)
        output_color_acc = trainer.callback_metrics.get("val_output_color_acc", 0.0)
        input_mask_acc = trainer.callback_metrics.get("val_input_mask_acc", 0.0)
        output_mask_acc = trainer.callback_metrics.get("val_output_mask_acc", 0.0)

        # Check if both colors and masks are perfect (>= 1.0)
        colors_perfect = (input_color_acc >= 1.0) and (output_color_acc >= 1.0)
        masks_perfect = (input_mask_acc >= 1.0) and (output_mask_acc >= 1.0)
        both_perfect = colors_perfect and masks_perfect

        if both_perfect:
            self.perfect_epochs += 1
            if self.best_epoch == -1:
                self.best_epoch = trainer.current_epoch

            print(
                f"\nBoth INPUT and OUTPUT achieved 100% accuracy for colors AND masks! ({self.perfect_epochs}/{self.patience} epochs)"
            )

            # Only stop if we've reached min_epochs and maintained perfect accuracy for patience epochs
            if (
                self.perfect_epochs >= self.patience
                and trainer.current_epoch >= self.min_epochs
            ):
                print(
                    f"Stopping training - perfect accuracy maintained for {self.patience} consecutive epochs (after {trainer.current_epoch} total epochs)"
                )
                trainer.should_stop = True
            elif trainer.current_epoch < self.min_epochs:
                print(
                    f"Perfect accuracy achieved but continuing to minimum {self.min_epochs} epochs (current: {trainer.current_epoch})"
                )
        else:
            # Reset counter if not perfect
            if self.perfect_epochs > 0:
                print(f"\nLost perfect accuracy after {self.perfect_epochs} epochs")
            self.perfect_epochs = 0
            self.best_epoch = -1


@click.command()
@from_pydantic(
    TrainingConfig,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfig.model_fields.keys()
        if "_" in field_name
    },
)
def main(training_config: TrainingConfig):

    # Target filename
    filename = training_config.filename

    print("\n=== Order2-based Model (ex30) ===")
    print(f"Training on 'train' subset of {filename}")
    print(f"Evaluating on 'test' subset of {filename}")
    print("Architecture: color -> order2 -> NCA -> order2 -> color")

    # Load train subset using data_loader.py
    (
        train_inputs,
        train_outputs,
        train_input_features,
        train_output_features,
        train_indices,
    ) = prepare_dataset(
        "processed_data/train_all_d4aug.npz",
        filter_filename=f"{filename}.json",
        use_features=True,
        dataset_name="train",
        use_train_subset=True,
    )

    # Load test subset using data_loader.py
    (
        test_inputs,
        test_outputs,
        test_input_features,
        test_output_features,
        test_indices,
    ) = prepare_dataset(
        "processed_data/train_all.npz",
        filter_filename=f"{filename}.json",
        use_features=True,
        dataset_name="test",
        use_train_subset=False,
    )

    # Get available colors from training data
    # Convert one-hot encoded tensors back to color indices
    train_input_colors = train_inputs.argmax(dim=-1)  # [B, 30, 30]
    train_output_colors = train_outputs.argmax(dim=-1)  # [B, 30, 30]

    # Handle mask: if max value is at index 10, it means masked (-1)
    train_input_colors = torch.where(train_input_colors == 10, -1, train_input_colors)
    train_output_colors = torch.where(
        train_output_colors == 10, -1, train_output_colors
    )

    all_colors = torch.cat(
        [train_input_colors.flatten(), train_output_colors.flatten()]
    )
    available_colors = torch.unique(all_colors[all_colors != -1])
    print(f"Available colors for {filename}: {available_colors}")

    # Create dataloaders using data_loader.py
    train_loader = create_dataloader(
        train_inputs,
        train_outputs,
        batch_size=len(train_inputs),
        shuffle=True,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        inputs_features=train_input_features,
        outputs_features=train_output_features,
        indices=train_indices,
    )
    val_loader = create_dataloader(
        test_inputs,
        test_outputs,
        batch_size=len(test_inputs),
        shuffle=False,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        inputs_features=test_input_features,
        outputs_features=test_output_features,
        indices=test_indices,
    )
    test_loader = create_dataloader(
        test_inputs,
        test_outputs,
        batch_size=len(test_inputs),
        shuffle=False,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        inputs_features=test_input_features,
        outputs_features=test_output_features,
        indices=test_indices,
    )

    # Create model
    model = Order2MemorizationModel(
        feature_dim=147,  # Not used in order2 version
        hidden_dim=training_config.hidden_dim,
        num_message_rounds=training_config.num_message_rounds,
        num_classes=10,
        dropout=training_config.dropout,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        temperature=training_config.temperature,
        filename=filename,
        num_train_examples=len(train_inputs),
        # Self-healing noise parameters
        enable_self_healing=training_config.enable_self_healing,
        death_prob=training_config.death_prob,
        gaussian_std=training_config.gaussian_std,
        salt_pepper_prob=training_config.salt_pepper_prob,
        spatial_corruption_prob=training_config.spatial_corruption_prob,
        num_final_steps=training_config.num_final_steps,
    )

    model.noise_prob = training_config.noise_prob
    model.available_colors = (
        available_colors.tolist()
        if hasattr(available_colors, "tolist")
        else list(available_colors)
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Create trainer with callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_color_acc",  # Monitor epoch-level color accuracy
        dirpath=os.path.join(training_config.checkpoint_dir, "checkpoints"),
        filename="order2-{epoch:02d}-{val_epoch_loss:.4f}-{val_epoch_color_acc:.4f}-{val_epoch_mask_acc:.4f}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    # Add custom callback for perfect accuracy
    perfect_accuracy_callback = PerfectAccuracyEarlyStopping(
        patience=training_config.patience, min_epochs=training_config.min_epochs
    )

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback, perfect_accuracy_callback],
        default_root_dir=training_config.checkpoint_dir,
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test set if in single file mode
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    # Run test evaluation
    model.eval()
    model.force_visualize = True
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model.validation_step(batch, batch_idx)

    # Save the final trained model with filename-specific name
    final_model_path = os.path.join(
        training_config.checkpoint_dir, f"order2_model_{filename}.pt"
    )

    # Save both model state dict and metadata
    model_save_data = {
        "model_state_dict": model.state_dict(),
        "available_colors": available_colors,
        "filename": filename,
        "hyperparameters": (
            dict(model.hparams) if hasattr(model.hparams, "__dict__") else model.hparams
        ),
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
