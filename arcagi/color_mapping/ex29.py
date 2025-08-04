import argparse
import os

# Import from parent module
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
# Import data loading functions
from data_loader import create_dataloader, prepare_dataset
from utils.terminal_imshow import imshow


def apply_binary_noise(
    features: torch.Tensor, noise_prob: float, training: bool
) -> torch.Tensor:
    """Apply binary noise to features during training only."""
    if training and noise_prob > 0:
        # Create noise mask with probability noise_prob
        noise_mask = torch.rand_like(features) < noise_prob

        # For binary features (0/1), flip them
        # For continuous features, add small random perturbation
        binary_features = (features == 0) | (features == 1)

        # Flip binary features
        features = torch.where(
            noise_mask & binary_features, 1 - features, features  # Flip 0->1, 1->0
        )

        # Add small noise to continuous features
        continuous_noise = torch.randn_like(features) * 0.01
        features = torch.where(
            noise_mask & ~binary_features, features + continuous_noise, features
        )

    return features


def apply_color_constraints(
    logits: torch.Tensor,
    available_colors: torch.Tensor | None = None,
    num_classes: int = 10,
) -> torch.Tensor:
    """Apply hard constraints to ensure only available colors are predicted."""
    if available_colors is not None:

        # Assert that available_colors contains integer types
        assert available_colors.dtype in [
            torch.int,
            torch.long,
            torch.int32,
            torch.int64,
        ], f"available_colors must have integer dtype, got {available_colors.dtype}"
        # Check max color is within valid range
        max_color = available_colors.max() if available_colors else -1
        assert (
            max_color < num_classes
        ), f"Maximum color index {max_color} must be less than num_classes {num_classes}"
        mask = torch.ones(num_classes, device=logits.device) * -1e10
        mask[available_colors] = 0
        logits = logits + mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    return logits


@dataclass
class Batch:
    """Dataclass to hold output batch data with proper typing."""

    one: torch.Tensor  # one hot [B, 30, 30, 10]
    fea: torch.Tensor  # order2 features [B, 30, 30, 147]
    col: torch.Tensor  # colors, int array [B, 30, 30]
    msk: torch.Tensor  # masks, bool array [B, 30, 30]
    colf: torch.Tensor  # colors, flattened [B, 30*30]
    mskf: torch.Tensor  # masks, flattened [B, 30*30]


@dataclass
class BatchData:
    """Dataclass to hold complete batch data with proper typing."""

    inp: Batch
    out: Batch


def batch_to_dataclass(
    batch: Tuple[torch.Tensor, ...],
) -> BatchData:
    inputs_one_hot, outputs_one_hot, input_features, output_features = batch

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
    )

    out = Batch(
        one=outputs_one_hot,
        fea=output_features,
        col=output_colors,
        msk=output_masks,
        colf=output_colors_flat,
        mskf=output_masks_flat.float(),
    )

    return BatchData(inp=inp, out=out)


def color_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions[valid_mask] == targets[valid_mask]).float().mean()


def mask_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = (logits > 0).float()
    return (predictions == targets.float()).float().mean()


def training_index_metrics(
    i: BatchData,
    col_log: torch.Tensor,
    msk_log: torch.Tensor,
    prefix: str,
    metrics_dict: Dict[int, Dict[str, int]],
) -> Dict[int, Dict[str, int]]:
    # For every index in the batch, compute the metrics:
    # Compute total number of actiev pixels not-masked
    # Compute number of correct color pixels
    # Compute number of correct mask pixels

    # Get predictions
    col_pred = col_log.argmax(dim=-1)  # [B, 30, 30]
    msk_pred = (msk_log > 0).squeeze(-1)  # [B, 30, 30]

    pixels_per_index = i.inp.mskf.sum(dim=-1)  # [B]
    pixels_incorrect = (col_pred != i.inp.col) | (msk_pred != i.inp.msk)
    pixels_incorrect_sum = pixels_incorrect.sum(dim=-1).sum(dim=-1)

    for idx in range(i.inp.col.shape[0]):
        metrics_dict[idx] = {
            f"{prefix}pixels_per_index": int(pixels_per_index[idx]),
            f"{prefix}pixels_incorrect": int(pixels_incorrect_sum[idx]),
        }
    breakpoint()

    return metrics_dict


def create_sinusoidal_embeddings(hidden_dim: int) -> torch.Tensor:
    """Create sinusoidal position embeddings for a 30x30 grid."""
    import math

    # Create position indices for x and y
    x_pos = torch.arange(30).float()
    y_pos = torch.arange(30).float()

    # Create 2D grid of positions
    grid_x, grid_y = torch.meshgrid(x_pos, y_pos, indexing="ij")

    # Initialize embedding tensor
    pos_embed = torch.zeros(30, 30, hidden_dim)

    # Half dimensions for x, half for y
    half_dim = hidden_dim // 2

    # Create div_term for frequency scaling
    div_term_x = torch.exp(
        torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim)
    )
    div_term_y = torch.exp(
        torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim)
    )

    # Apply sinusoidal functions to x coordinates
    pos_embed[:, :, 0:half_dim:2] = torch.sin(grid_x.unsqueeze(-1) * div_term_x)
    pos_embed[:, :, 1:half_dim:2] = torch.cos(grid_x.unsqueeze(-1) * div_term_x)

    # Apply sinusoidal functions to y coordinates
    pos_embed[:, :, half_dim::2] = torch.sin(grid_y.unsqueeze(-1) * div_term_y)
    pos_embed[:, :, half_dim + 1 :: 2] = torch.cos(grid_y.unsqueeze(-1) * div_term_y)

    return pos_embed


class OptimizedMemorizationModel(pl.LightningModule):
    """
    Optimized model for perfect memorization with minimal parameters.

    This version (ex23) predicts both colors and masks with separate accuracies and losses.

    Key optimizations:
    1. Example-aware feature transformation
    2. Neural Cellular Automata (NCA) with learnable perception filters
    3. Adaptive learning rate with strong warmup
    4. Focused architecture for 2 training examples
    5. Batch normalization for faster convergence
    6. Better weight initialization
    7. Sinusoidal position embeddings (parameter-free)
    8. Dual prediction heads for colors and masks
    9. Emergent spatial patterns through cellular automata
    """

    training_index_metrics: Dict[int, Dict[str, int]] = {}
    validation_index_metrics: Dict[int, Dict[str, int]] = {}

    def __init__(
        self,
        feature_dim: int = 147,
        hidden_dim: int = 512,
        num_message_rounds: int = 6,
        num_classes: int = 10,
        dropout: float = 0.0,
        lr: float = 0.1,  # Even higher learning rate
        weight_decay: float = 0.0,
        temperature: float = 0.01,  # Lower temperature for sharper predictions
        filename: str = "3345333e",
        num_train_examples: int = 2,
        dim_nca: int = 16,
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

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_message_rounds = num_message_rounds
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.filename = filename
        self.num_train_examples = num_train_examples

        # Track validation metrics across entire epoch
        self.validation_outputs = []

        # Track per-index metrics for validation
        self.index_metrics = {}

        # Track per-index metrics for training
        self.training_index_metrics = {}

        # Feature extraction with layer normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Binary noise layer for training-time regularization
        self.noise_prob = 0.05  # 5% probability of flipping each feature

        # Sinusoidal position embeddings - no parameters needed
        # Pre-compute the position embeddings for efficiency
        self.register_buffer("pos_embed", create_sinusoidal_embeddings(hidden_dim))

        self.linear_1 = nn.Linear(hidden_dim, dim_nca)
        self.linear_2 = nn.Linear(dim_nca, hidden_dim)

        # Neural Cellular Automata for spatial refinement with self-healing
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

        # Spatial message passing for enhanced local consistency
        self.message_passing = SpatialMessagePassing(hidden_dim, dropout=dropout)

        # Color-specific prediction heads for better specialization
        self.color_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_classes)]
        )

        # Mask prediction head
        self.mask_head = nn.Linear(hidden_dim, 1)

        # Global context aggregation removed to reduce parameters

        # Store available colors
        self.available_colors: Optional[List[int]] = None

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
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply binary noise during training
        features = apply_binary_noise(
            features, noise_prob=self.noise_prob, training=self.training
        )

        # Extract base features
        h = self.feature_extractor(features)  # [B, 30, 30, hidden_dim]

        # Add position embeddings
        pos_embed = self.pos_embed.unsqueeze(0)  # Add batch dimension
        h = h + pos_embed

        # Hybrid NCA and Message Passing processing
        x = self.linear_1(h)

        # Phase 1: Steps with self-healing noise (for robustness training)
        noisy_steps = self.num_message_rounds - self.nca.num_final_steps
        for i in range(max(0, noisy_steps)):
            # Step 1: Neural Cellular Automata
            x = x + self.nca(x, apply_noise=True)

            # Step 2: Convert back to full hidden_dim for message passing
            h_temp = h + self.linear_2(x)

            # Step 3: Spatial message passing
            h_messages = self.message_passing(h_temp)

            # Step 4: Residual connection with gradual increase
            # alpha = 0.3 + 0.1 * (i / max(1, noisy_steps))
            alpha = 0.3
            h = h + alpha * h_messages

            # Step 5: Update x for next NCA iteration
            x = self.linear_1(h)

        # Phase 2: Final steps without noise (for self-cleaning)
        for i in range(min(self.nca.num_final_steps, self.num_message_rounds)):
            # Step 1: Neural Cellular Automata (no noise)
            x = x + self.nca(x, apply_noise=False)

            # Step 2: Convert back to full hidden_dim for message passing
            h_temp = h + self.linear_2(x)

            # Step 3: Spatial message passing
            h_messages = self.message_passing(h_temp)

            # Step 4: Residual connection with gradual increase
            # alpha = 0.4 + 0.1 * (i / max(1, self.nca.num_final_steps))
            alpha = 0.3
            h = h + alpha * h_messages

            # Step 5: Update x for next NCA iteration (if not last step)
            if i < min(self.nca.num_final_steps, self.num_message_rounds) - 1:
                x = self.linear_1(h)

        # Final projection
        h = h + self.linear_2(x)

        # Global context computation removed to reduce parameters
        # Sequential NCA+MP processing should provide sufficient spatial context

        # Color-specific predictions
        logits_list: List[torch.Tensor] = []
        for color_head in self.color_heads:
            logits_list.append(color_head(h))  # [B, 30, 30, 1]

        color_logits = torch.cat(logits_list, dim=-1)  # [B, 30, 30, num_classes]

        # Temperature scaling for colors
        color_logits = color_logits / self.temperature

        # Apply color constraints
        color_logits = apply_color_constraints(color_logits, self.available_colors)

        # Mask prediction
        mask_logits = self.mask_head(h)  # [B, 30, 30, 1]

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

        # Compute losses for order2->color
        # for (input/output) x (colors/masks)
        losses: List[torch.Tensor] = []
        metrics: Dict[int, Dict[str, int]] = {}
        for prefix, ex in [("input", i.inp), ("output", i.out)]:
            col_log, msk_log = self(ex.fea)
            col_logf = col_log.reshape(-1, col_log.size(-1))
            col_loss = F.cross_entropy(col_logf, ex.colf)
            msk_logf = msk_log.reshape(-1)
            msk_loss = F.binary_cross_entropy_with_logits(msk_logf, ex.mskf)
            losses.append(col_loss + msk_loss)
            training_index_metrics(i, col_log, msk_log, prefix, metrics)
        if step_type == "train":
            self.training_index_metrics = metrics
        else:
            self.validation_index_metrics = metrics

        # Combined loss
        total_loss = losses[0] + losses[1]

        self.log_metrics(metrics, "train")
        self.log("train_total_loss", total_loss)  # type: ignore

        # Visualize every 10 epochs
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            for idx in range(len(i.out.fea)):
                self.visualize_predictions(
                    i.out.col[idx : idx + 1],
                    i.out.msk[idx : idx + 1],
                    "output",
                )
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

    def on_validation_epoch_start(self) -> None:
        """Reset validation outputs at the start of each epoch."""
        self.epoch_end("val")

    def epoch_end(self, step_type: Literal["train", "val"]) -> None:
        if step_type == "train":
            metrics = self.training_index_metrics
        else:
            metrics = self.validation_index_metrics

        if metrics:
            print(
                f"\n{step_type.upper()} Per-Index Accuracy (Epoch {self.current_epoch}):"
            )
            print(f"{'Index':<8} {'Output Pixels':<12} {'All Perfect':<12}")
            print("-" * 50)

            for idx in sorted(self.training_index_metrics.keys()):
                metrics = self.training_index_metrics[idx]
                out_pix_incor = metrics["output_pixels_incorrect"]

                all_perfect = "✓" if out_pix_incor == 0 else "✗"
                print(f"{idx:<8} {out_pix_incor:<12.2%} {all_perfect:<12}")

        # Clear training metrics for next epoch
        self.training_index_metrics = {}

    def configure_optimizers(self):
        # Use SGD with momentum for faster convergence on small dataset
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Very aggressive learning rate schedule
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
        targets: torch.Tensor,
        predictions: torch.Tensor,
        prefix: str,
    ) -> None:
        """Visualize predictions vs ground truth for training."""

        pred_colors = predictions[0].argmax(dim=-1).cpu()
        true_colors = targets[0].cpu()

        # Create visualization
        print(f"\n{'='*60}")
        print(f"{prefix.upper()} - Epoch {self.current_epoch}")
        print(f"{'='*60}")

        # Show ground truth
        print("\nGround Truth colors:")
        imshow(true_colors, title=None, show_legend=True)

        # Show predictions
        print("\nPredicted colors:")
        imshow(pred_colors, title=None, show_legend=True, correct=true_colors)

        # Calculate accuracy for this example
        valid_mask = true_colors != -1
        if valid_mask.any():
            accuracy = (
                (pred_colors[valid_mask] == true_colors[valid_mask]).float().mean()
            )
            print(f"\nAccuracy: {accuracy:.2%}")


class SpatialMessagePassing(nn.Module):
    """Efficient spatial message passing for local consistency."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple 3x3 convolution for message passing
        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim // 16,  # Group convolution for efficiency
        )

        self.norm = nn.LayerNorm(hidden_dim)
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

    Applies various types of biologically-inspired noise:
    - Cell death (zeroing random cells)
    - Gaussian feature noise (metabolic fluctuations)
    - Salt & pepper noise (random extreme values)
    - Spatial corruption (damaging regions)
    """

    def __init__(
        self,
        death_prob: float = 0.05,
        gaussian_std: float = 0.1,
        salt_pepper_prob: float = 0.02,
        spatial_corruption_prob: float = 0.03,
    ):
        super().__init__()
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

        B, H, W, C = h.shape

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
    Simplified Neural Cellular Automata for spatial pattern refinement.

    Inspired by "Growing Neural Cellular Automata" (Mordvintsev et al., 2020).
    Each cell updates its state based on its neighborhood using learned rules.
    Uses learnable perception filters instead of fixed Sobel filters.

    Simplified version without aliveness concept - all cells can update.
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
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_final_steps = num_final_steps

        # Learnable perception filters instead of fixed Sobel filters
        # Each channel gets 3 learnable 3x3 filters for different spatial patterns
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
        apply_noise: Whether to apply self-healing noise (for two-phase processing)
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

    def __init__(self, patience: int = 5, min_epochs: int = 200):
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


# Custom dataloader removed - now using data_loader.py functions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/tmp/arc_data", help="Data directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="optimized_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2000,
        help="Maximum number of epochs for Phase 1 (with noise)",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay"
    )  # Increased for regularization
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num_message_rounds",
        type=int,
        default=24,  # Default to 24 NCA evolution steps
        help="Number of NCA refinement rounds",
    )

    # Self-healing noise parameters
    parser.add_argument(
        "--enable_self_healing",
        action="store_true",
        default=True,
        help="Enable self-healing noise",
    )
    parser.add_argument(
        "--death_prob",
        type=float,
        default=0.02,
        help="Cell death probability (0.0-1.0)",
    )
    parser.add_argument(
        "--gaussian_std",
        type=float,
        default=0.05,
        help="Gaussian noise standard deviation",
    )
    parser.add_argument(
        "--salt_pepper_prob",
        type=float,
        default=0.01,
        help="Salt & pepper noise probability",
    )
    parser.add_argument(
        "--spatial_corruption_prob",
        type=float,
        default=0.01,
        help="Spatial corruption probability",
    )

    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )  # Increased from 0.02
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for scaling"
    )
    parser.add_argument(
        "--filename", type=str, default="3345333e", help="Filename to train on"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (epochs to wait after reaching 100% accuracy)",
    )
    parser.add_argument(
        "--single_file_mode",
        action="store_true",
        help="Train on 'train' subset and evaluate on 'test' subset from the same file",
        default=True,
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.10,
        help="Probability of applying binary noise to features during training (default: 0.05)",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=200,
        help="Minimum number of epochs to train before early stopping (default: 200)",
    )
    parser.add_argument(
        "--num_final_steps",
        type=int,
        default=6,
        help="Number of final NCA steps per forward pass without self-healing noise for self-cleaning (default: 6)",
    )

    args = parser.parse_args()

    # Target filename
    filename = args.filename

    if args.single_file_mode:
        print(f"\n=== Single File Mode ===")
        print(f"Training on 'train' subset of {filename}")
        print(f"Evaluating on 'test' subset of {filename}")

        # Load train subset using data_loader.py
        train_inputs, train_outputs, train_input_features, train_output_features = (
            prepare_dataset(
                "processed_data/train_all_d4aug.npz",
                filter_filename=f"{filename}.json",
                use_features=True,
                dataset_name="train",
                use_train_subset=True,
            )
        )

        # Load test subset using data_loader.py
        test_inputs, test_outputs, test_input_features, test_output_features = (
            prepare_dataset(
                "processed_data/train_all.npz",
                filter_filename=f"{filename}.json",
                use_features=True,
                dataset_name="test",
                use_train_subset=False,
            )
        )

        # Get available colors from training data
        # Convert one-hot encoded tensors back to color indices
        train_input_colors = train_inputs.argmax(dim=-1)  # [B, 30, 30]
        train_output_colors = train_outputs.argmax(dim=-1)  # [B, 30, 30]

        # Handle mask: if max value is at index 10, it means masked (-1)
        train_input_colors = torch.where(
            train_input_colors == 10, -1, train_input_colors
        )
        train_output_colors = torch.where(
            train_output_colors == 10, -1, train_output_colors
        )

        all_colors = torch.cat(
            [train_input_colors.flatten(), train_output_colors.flatten()]
        )
        available_colors = torch.unique(all_colors[all_colors != -1]).tolist()
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
        )
        val_loader = create_dataloader(
            test_inputs,
            test_outputs,
            batch_size=len(test_inputs),
            shuffle=False,
            num_workers=0,  # Eliminate multiprocessing overhead for small datasets
            inputs_features=test_input_features,
            outputs_features=test_output_features,
        )
        test_loader = create_dataloader(
            test_inputs,
            test_outputs,
            batch_size=len(test_inputs),
            shuffle=False,
            num_workers=0,  # Eliminate multiprocessing overhead for small datasets
            inputs_features=test_input_features,
            outputs_features=test_output_features,
        )
    else:
        # Load all data (no subset filtering) using data_loader.py
        all_inputs, all_outputs, all_input_features, all_output_features = (
            prepare_dataset(
                "processed_data/train_all.npz",
                filter_filename=f"{filename}.json",
                use_features=True,
                dataset_name="all",
                use_train_subset=None,
            )
        )

        # Get available colors
        # Convert one-hot encoded tensors back to color indices
        all_input_colors = all_inputs.argmax(dim=-1)  # [B, 30, 30]
        all_output_colors = all_outputs.argmax(dim=-1)  # [B, 30, 30]

        # Handle mask: if max value is at index 10, it means masked (-1)
        all_input_colors = torch.where(all_input_colors == 10, -1, all_input_colors)
        all_output_colors = torch.where(all_output_colors == 10, -1, all_output_colors)

        all_colors = torch.cat(
            [all_input_colors.flatten(), all_output_colors.flatten()]
        )
        available_colors = torch.unique(all_colors[all_colors != -1]).tolist()
        print(f"Available colors for {filename}: {available_colors}")

        # Create dataloaders using data_loader.py
        train_loader = create_dataloader(
            all_inputs,
            all_outputs,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # Eliminate multiprocessing overhead for small datasets
            inputs_features=all_input_features,
            outputs_features=all_output_features,
        )
        val_loader = create_dataloader(
            all_inputs,
            all_outputs,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Eliminate multiprocessing overhead for small datasets
            inputs_features=all_input_features,
            outputs_features=all_output_features,
        )
        test_loader = None

    # Create model
    model = OptimizedMemorizationModel(
        feature_dim=147,
        hidden_dim=args.hidden_dim,
        num_message_rounds=args.num_message_rounds,
        num_classes=10,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        filename=filename,
        num_train_examples=(
            len(train_inputs) if args.single_file_mode else len(all_inputs)
        ),
        # Self-healing noise parameters
        enable_self_healing=args.enable_self_healing,
        death_prob=args.death_prob,
        gaussian_std=args.gaussian_std,
        salt_pepper_prob=args.salt_pepper_prob,
        spatial_corruption_prob=args.spatial_corruption_prob,
        num_final_steps=args.num_final_steps,
    )

    # Set noise probability
    model.noise_prob = args.noise_prob

    # Set available colors
    model.available_colors = available_colors

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Create trainer with callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_color_acc",  # Monitor epoch-level color accuracy
        dirpath=os.path.join(args.checkpoint_dir, "checkpoints"),
        filename=f"optimized-{{epoch:02d}}-{{val_epoch_loss:.4f}}-{{val_epoch_color_acc:.4f}}-{{val_epoch_mask_acc:.4f}}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    # Add custom callback for perfect accuracy
    perfect_accuracy_callback = PerfectAccuracyEarlyStopping(
        patience=args.patience, min_epochs=args.min_epochs
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, perfect_accuracy_callback],
        default_root_dir=args.checkpoint_dir,
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )

    # Train with two-phase NCA processing
    print(f"\n=== TRAINING WITH TWO-PHASE NCA PROCESSING ===")
    print(f"Training for {args.max_epochs} epochs")
    print(f"NCA steps per forward pass: {args.num_message_rounds}")
    print(
        f"  - Steps with noise: {max(0, args.num_message_rounds - args.num_final_steps)}"
    )
    print(
        f"  - Steps without noise (self-cleaning): {min(args.num_final_steps, args.num_message_rounds)}"
    )
    print(f"Self-healing noise enabled: {args.enable_self_healing}")

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining completed for {filename}")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")

    # Evaluate on test set if in single file mode
    if args.single_file_mode and test_loader is not None:
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)

        # Run test evaluation
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                # Handle new data_loader.py format: (inputs, outputs, inputs_features, outputs_features)
                if len(batch) == 4:
                    inputs_one_hot, outputs_one_hot, input_features, output_features = (
                        batch
                    )
                    # Ensure features are float tensors for MPS compatibility
                    input_features = input_features.float()
                    output_features = output_features.float()
                    # Extract colors and masks from the grid data
                    input_colors = inputs_one_hot.argmax(dim=-1).long()
                    output_colors = outputs_one_hot.argmax(dim=-1).long()
                    # Change any colors >= 10 to -1 (mask)
                    input_colors = torch.where(input_colors >= 10, -1, input_colors)
                    output_colors = torch.where(output_colors >= 10, -1, output_colors)
                    input_masks = input_colors >= 0
                    output_masks = output_colors >= 0
                else:
                    # Legacy format for backward compatibility
                    (
                        input_features,
                        output_features,
                        input_colors,
                        output_colors,
                        input_masks,
                        output_masks,
                        _,
                        _,
                    ) = batch

                # Get predictions
                input_color_logits, input_mask_logits = model(input_features)
                output_color_logits, output_mask_logits = model(output_features)

                input_color_predictions = input_color_logits.argmax(dim=-1)
                output_color_predictions = output_color_logits.argmax(dim=-1)
                input_mask_predictions = (input_mask_logits > 0).squeeze(-1)
                output_mask_predictions = (output_mask_logits > 0).squeeze(-1)

                # Calculate color accuracies
                input_color_valid_mask = input_colors != -1
                output_color_valid_mask = output_colors != -1

                input_color_correct = 0
                input_color_total = 0
                output_color_correct = 0
                output_color_total = 0

                if input_color_valid_mask.any():
                    input_color_correct = (
                        (
                            input_color_predictions[input_color_valid_mask]
                            == input_colors[input_color_valid_mask]
                        )
                        .sum()
                        .item()
                    )
                    input_color_total = input_color_valid_mask.sum().item()

                if output_color_valid_mask.any():
                    output_color_correct = (
                        (
                            output_color_predictions[output_color_valid_mask]
                            == output_colors[output_color_valid_mask]
                        )
                        .sum()
                        .item()
                    )
                    output_color_total = output_color_valid_mask.sum().item()

                # Calculate mask accuracies
                input_mask_correct = (
                    (input_mask_predictions == input_masks).sum().item()
                )
                output_mask_correct = (
                    (output_mask_predictions == output_masks).sum().item()
                )
                input_mask_total = input_masks.numel()
                output_mask_total = output_masks.numel()

                # Print results
                print(f"\nTest Set Results:")
                print(
                    f"INPUT color accuracy:  {input_color_correct}/{input_color_total} = {input_color_correct/input_color_total*100:.1f}%"
                    if input_color_total > 0
                    else "No input color pixels"
                )
                print(
                    f"OUTPUT color accuracy: {output_color_correct}/{output_color_total} = {output_color_correct/output_color_total*100:.1f}%"
                    if output_color_total > 0
                    else "No output color pixels"
                )
                print(
                    f"INPUT mask accuracy:  {input_mask_correct}/{input_mask_total} = {input_mask_correct/input_mask_total*100:.1f}%"
                )
                print(
                    f"OUTPUT mask accuracy: {output_mask_correct}/{output_mask_total} = {output_mask_correct/output_mask_total*100:.1f}%"
                )

                total_color_correct = input_color_correct + output_color_correct
                total_color_pixels = input_color_total + output_color_total
                total_mask_correct = input_mask_correct + output_mask_correct
                total_mask_pixels = input_mask_total + output_mask_total

                if total_color_pixels > 0:
                    print(
                        f"OVERALL color accuracy: {total_color_correct}/{total_color_pixels} = {total_color_correct/total_color_pixels*100:.1f}%"
                    )
                print(
                    f"OVERALL mask accuracy: {total_mask_correct}/{total_mask_pixels} = {total_mask_correct/total_mask_pixels*100:.1f}%"
                )

                # Show visual comparison for first example
                if len(input_features) > 0:
                    print("\nVisual comparison of first test example:")
                    # model.visualize_predictions(
                    #     input_features[0:1],
                    #     input_colors[0:1],
                    #     input_color_logits[0:1],
                    #     "TEST INPUT",
                    # )
                    model.visualize_predictions(
                        output_features[0:1],
                        output_colors[0:1],
                        output_color_logits[0:1],
                        "TEST OUTPUT",
                    )

    # Save the final trained model with filename-specific name
    final_model_path = os.path.join(args.checkpoint_dir, f"color_model_{filename}.pt")

    # Save both model state dict and metadata
    model_save_data = {
        "model_state_dict": model.state_dict(),
        "available_colors": available_colors,
        "filename": filename,
        "hyperparameters": model.hparams,
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


def load_color_model_for_inference(
    filename: str, checkpoint_dir: str = "color_mapping_outputs"
) -> OptimizedMemorizationModel:
    """Load a trained color and mask model for inference."""
    model_path = os.path.join(checkpoint_dir, f"color_model_{filename}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for filename: {filename} at {model_path}"
        )

    # Load saved data (weights_only=False for our trusted models to handle custom classes)
    saved_data = torch.load(model_path, map_location="cpu", weights_only=False)

    # Create model with saved hyperparameters
    hparams = saved_data["hyperparameters"]
    model = OptimizedMemorizationModel(
        feature_dim=hparams.get("feature_dim", 147),
        hidden_dim=hparams.get("hidden_dim", 512),
        num_message_rounds=hparams.get("num_message_rounds", 6),
        num_classes=hparams.get("num_classes", 10),
        dropout=hparams.get("dropout", 0.0),
        lr=hparams.get("lr", 0.1),
        weight_decay=hparams.get("weight_decay", 0.0),
        temperature=hparams.get("temperature", 0.01),
        filename=saved_data["filename"],
        num_train_examples=hparams.get("num_train_examples", 2),
        # Self-healing noise parameters
        enable_self_healing=hparams.get("enable_self_healing", True),
        death_prob=hparams.get("death_prob", 0.02),
        gaussian_std=hparams.get("gaussian_std", 0.05),
        salt_pepper_prob=hparams.get("salt_pepper_prob", 0.01),
        spatial_corruption_prob=hparams.get("spatial_corruption_prob", 0.01),
        num_final_steps=hparams.get("num_final_steps", 12),
    )

    # Load model weights
    model.load_state_dict(saved_data["model_state_dict"])

    # Set available colors
    model.available_colors = saved_data["available_colors"]

    # Set to evaluation mode
    model.eval()

    print(
        f"Loaded color and mask model for {filename} with colors: {saved_data['available_colors']}"
    )

    return model


if __name__ == "__main__":
    main()
