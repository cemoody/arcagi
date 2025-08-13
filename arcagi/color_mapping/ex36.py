import hashlib
import json
import os
import random

# Import from parent module
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    TypeAlias,
    TypeVar,
    Optional,
)

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydanclick import from_pydantic
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from loguru import logger
import rich

FuncT = TypeVar("FuncT")

if TYPE_CHECKING:
    # During type-checking, we can reference callables with the right types
    def jaxtyped(*args: Any, **kwargs: Any) -> Callable[[FuncT], FuncT]:  # type: ignore[misc]
        ...

    def beartype(func: FuncT) -> FuncT:  # type: ignore[misc]
        ...

else:
    try:
        from beartype import beartype
        from jaxtyping import jaxtyped
    except Exception:  # pragma: no cover - fallback if not installed

        def beartype(func: FuncT) -> FuncT:  # type: ignore[misc]
            return func

        def jaxtyped(*args: Any, **kwargs: Any) -> Callable[[FuncT], FuncT]:  # type: ignore[misc]
            def decorator(func: FuncT) -> FuncT:
                return func

            return decorator


# Readable tensor aliases (shapes noted in comments)
ColorGrid: TypeAlias = "torch.Tensor"  # [B, 30, 30] int
OneHot11: TypeAlias = "torch.Tensor"  # [B, 30, 30, 11] float
Features45: TypeAlias = "torch.Tensor"  # [B, 30, 30, 45] float
HiddenGrid: TypeAlias = "torch.Tensor"  # [B, 30, 30, H] float
NCALatent: TypeAlias = "torch.Tensor"  # [B, D, 30, 30] float
D4TransformType: TypeAlias = "torch.Tensor"  # [B, 30, 30] int
try:
    from pytorch_lightning.loggers import WandbLogger  # type: ignore
except Exception:
    WandbLogger = None  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
# Import data loading functions
from data_loader import create_dataloader, prepare_dataset
from d4_augmentation_dataset import create_d4_augmented_dataloader
from models import Batch, BatchData
from utils.metrics import image_metrics
from utils.terminal_imshow import imshow
from utils.frame_capture import FrameCapture

# Import Order2Features from lib
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lib.order2 import Order2Features
from arcagi.adaptive_grad_scaler import AdaptiveGradScalerSkip

class TrainingConfig(BaseModel):
    """Configuration for training the order2-based neural cellular automata model."""

    # Checkpoint path
    checkpoint_dir: str = "order2_checkpoints"

    # Training parameters
    max_epochs: int = 2000
    lr: float = 0.005
    weight_decay: float = 1e-9
    feature_dim: int = 64
    num_message_rounds: int = 48
    num_final_steps: int = 12
    num_rounds_stop_prob: float = 0.10

    # Self-healing noise parameters
    enable_self_healing: bool = True
    death_prob: float = 0.30
    gaussian_std: float = 0.30
    spatial_corruption_prob: float = 0.30
    dropout: float = 0.15
    dropout_h: float = 0.15

    # Model parameters
    filename: str = "3345333e"

    # D4 augmentation parameters
    use_d4_augmentation: bool = True
    d4_deterministic: bool = True  # If True, cycles through all 8 transformations
    
    # Adaptive gradient scaler parameters
    enable_adaptive_grad_scaler: bool = True
    grad_scaler_beta: float = 0.98
    grad_scaler_warmup_steps: int = 500
    grad_scaler_mult: float = 4.0
    grad_scaler_z_thresh: float = 4.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self: "RMSNorm", dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @jaxtyped(typechecker=beartype)
    def forward(self, x: HiddenGrid) -> HiddenGrid:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps
        x_normed = x / rms
        return self.weight * x_normed


class ArsinhNorm(nn.Module):
    """Inverse Hyperbolic Sine Normalization - more stable than RMSNorm."""

    def __init__(self: "ArsinhNorm", dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    @jaxtyped(typechecker=beartype)
    def forward(self, x: HiddenGrid) -> HiddenGrid:
        # Apply asinh normalization: asinh(x/scale)
        # asinh is smooth, handles pos/neg values, and has bounded gradients
        x_scaled = x / self.scale
        x_normed = torch.asinh(x_scaled)
        return self.weight * x_normed + self.bias



# Try to import NATTEN
try:
    from natten import NeighborhoodAttention2D
    NATTEN_AVAILABLE = True
except ImportError:
    print("Warning: NATTEN not found. Please install NATTEN for optimal performance.")
    print("Install with: pip install natten")
    NATTEN_AVAILABLE = False
    NeighborhoodAttention2D = None  # type: ignore


class LocalAttentionBlock(nn.Module):
    """Local attention block using NATTEN for efficient neighborhood attention."""
    
    def __init__(
        self: "LocalAttentionBlock",
        hidden_dim: int,
        num_heads: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        
        # Input projection if context is provided
        input_dim = hidden_dim + (context_dim if context_dim else 0)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if context_dim else None
        
        # Pre-normalization
        self.norm1 = ArsinhNorm(hidden_dim)
        
        if NATTEN_AVAILABLE:
            # Use NATTEN for efficient local attention
            self.attention = NeighborhoodAttention2D(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                qkv_bias=True,
                qk_scale=None,
                proj_drop=dropout,
            )
        else:
            # Fallback to a simple local convolution if NATTEN not available
            print("Using fallback convolution instead of NATTEN")
            self.attention = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                         padding=kernel_size//2, groups=min(hidden_dim, 8)),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
            )
            self.use_conv_fallback = True
        
        # Post-attention normalization and feed-forward
        self.norm2 = ArsinhNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        
    @jaxtyped(typechecker=beartype)
    def forward(self, h: HiddenGrid, context: Optional[HiddenGrid] = None) -> HiddenGrid:
        # Handle context if provided
        if context is not None and self.context_dim is not None:
            h_with_context = torch.cat([h, context], dim=-1)
            h = self.input_proj(h_with_context)
        
        # Pre-norm and attention
        h_normed = self.norm1(h)
        
        if NATTEN_AVAILABLE and not hasattr(self, 'use_conv_fallback'):
            # NATTEN expects [B, H, W, C] format
            h_attn = self.attention(h_normed)
        else:
            # Fallback convolution
            h_conv = h_normed.permute(0, 3, 1, 2)  # [B, C, H, W]
            h_attn = self.attention(h_conv)
            h_attn = h_attn.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Residual connection
        h = h + h_attn
        
        # Feed-forward with residual
        h_normed = self.norm2(h)
        h_ff = self.mlp(h_normed)
        h = h + h_ff
        
        return h


class SelfHealingNoise(nn.Module):
    def __init__(
        self: "SelfHealingNoise",
        death_prob: float = 0.05,
        gaussian_std: float = 0.1,
        spatial_corruption_prob: float = 0.03,
    ) -> None:
        super().__init__()
        self.death_prob = death_prob
        self.gaussian_std = gaussian_std
        self.spatial_corruption_prob = spatial_corruption_prob

    @jaxtyped(typechecker=beartype)
    def forward(self, h: HiddenGrid) -> HiddenGrid:
        if not self.training:
            return h
        b, height, width, _ = h.shape
        if self.death_prob > 0:
            death_mask = torch.rand(b, height, width, device=h.device) > self.death_prob
            h = h * death_mask.unsqueeze(-1)
        if self.gaussian_std > 0:
            noise = torch.randn_like(h) * self.gaussian_std
            h = h + noise
        if self.spatial_corruption_prob > 0:
            for bi in range(b):
                if random.random() < self.spatial_corruption_prob:
                    grid_size = random.randint(1, 5)  # Random size 1-5
                    y = torch.randint(0, height - grid_size + 1, (1,)).item()
                    x = torch.randint(0, width - grid_size + 1, (1,)).item()
                    h[bi, y : y + grid_size, x : x + grid_size, :] = 0
        return h


class LocalAttentionStack(nn.Module):
    """Stack of local attention blocks for iterative refinement."""
    
    def __init__(
        self: "LocalAttentionStack",
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        kernel_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.05,
        context_dim: Optional[int] = None,
        enable_self_healing: bool = True,
        death_prob: float = 0.05,
        gaussian_std: float = 0.05,
        spatial_corruption_prob: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        
        # Default kernel sizes and dilations if not provided
        if kernel_sizes is None:
            kernel_sizes = [3] * num_layers  # All layers use 3x3 kernels
        if dilations is None:
            # Gradually increase dilation for larger receptive field
            dilations = [1, 1, 2, 2] if num_layers >= 4 else [1] * num_layers
            
        assert len(kernel_sizes) == num_layers
        assert len(dilations) == num_layers
        
        # Self-healing noise for robustness
        self.self_healing_noise = (
            SelfHealingNoise(
                death_prob=death_prob,
                gaussian_std=gaussian_std,
                spatial_corruption_prob=spatial_corruption_prob,
            )
            if enable_self_healing
            else None
        )
        
        # Stack of local attention blocks
        self.layers = nn.ModuleList([
            LocalAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_sizes[i],
                dilation=dilations[i],
                dropout=dropout,
                context_dim=context_dim if i == 0 else None,  # Only first layer gets context
            )
            for i in range(num_layers)
        ])
        
    @jaxtyped(typechecker=beartype)
    def forward(
        self, 
        h: HiddenGrid, 
        context: Optional[HiddenGrid] = None,
        apply_noise: bool = True,
        num_steps: Optional[int] = None,
    ) -> HiddenGrid:
        # Apply self-healing noise if enabled
        if apply_noise and self.self_healing_noise is not None:
            h = self.self_healing_noise(h)
            
        # Determine number of steps
        steps = num_steps if num_steps is not None else self.num_layers
        
        # Apply attention layers
        for i in range(min(steps, self.num_layers)):
            # Only pass context to the first layer
            layer_context = context if i == 0 else None
            h = self.layers[i](h, context=layer_context)
            
        return h



@jaxtyped(typechecker=beartype)
def batch_to_dataclass(
    batch: List[torch.Tensor],
) -> BatchData:
    # Check if we have transform indices (from D4 augmented dataloader)
    if len(batch) == 6:
        inputs_one_hot, outputs_one_hot, input_features, output_features, indices, transform_indices = batch
    else:
        inputs_one_hot, outputs_one_hot, input_features, output_features, indices = batch
        transform_indices = None

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
        transform_idx=transform_indices,
    )

    out = Batch(
        one=outputs_one_hot,
        fea=output_features,
        col=output_colors,
        msk=output_masks,
        colf=output_colors_flat,
        mskf=output_masks_flat.float(),
        idx=indices,
        transform_idx=transform_indices,
    )

    return BatchData(inp=inp, out=out)


@jaxtyped(typechecker=beartype)
def cross_entropy_shifted(
    logits: torch.Tensor,
    targets: torch.Tensor,
    start_index: int = -1,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Cross-entropy loss for class indices starting at `start_index` instead of 0.

    Args:
        logits: [N, C, ...] raw predictions (unnormalized)
        targets: [N, ...] integer class labels in range [start_index, start_index + C - 1]
        start_index: the smallest valid class index (default: -1)
        **kwargs: extra args for F.cross_entropy (e.g., reduction, ignore_index)
    """
    # Shift targets so that smallest index becomes 0
    shifted_targets = targets - start_index
    return F.cross_entropy(logits, shifted_targets, **kwargs)


class MainModel(pl.LightningModule):
    """Order2 + NCA model that outputs 11-class logits (-1 mask + 10 colors)."""

    training_index_metrics: Dict[int, Dict[str, int]] = {}
    validation_index_metrics: Dict[int, Dict[str, int]] = {}
    force_visualize: bool = False
    last_max_loss: Optional[float] = None
    last_logits_abs_mean: Optional[float] = None
    frame_capture: FrameCapture = FrameCapture()

    def __init__(
        self,
        feature_dim: int = 64,
        num_message_rounds: int = 6,
        dropout: float = 0.0,
        dropout_h: float = 0.05,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        enable_self_healing: bool = True,
        death_prob: float = 0.02,
        gaussian_std: float = 0.05,
        spatial_corruption_prob: float = 0.01,
        num_final_steps: int = 12,
        num_rounds_stop_prob: float = 0.50,
        dim_feat: int = 45,
        enable_adaptive_grad_scaler: bool = True,
        grad_scaler_beta: float = 0.98,
        grad_scaler_warmup_steps: int = 500,
        grad_scaler_mult: float = 4.0,
        grad_scaler_z_thresh: float = 4.0,
        dim_d4_embed: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim = feature_dim + dim_d4_embed
        self.num_message_rounds = num_message_rounds
        self.num_rounds_stop_prob = num_rounds_stop_prob
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        # Order2 feature extractor
        self.order2_encoder = Order2Features()
        self.num_final_steps = num_final_steps

        # Feature processing and projection to NCA space
        self.feature_processor = nn.Sequential(
            nn.Linear(dim_feat, feature_dim),
            RMSNorm(feature_dim),
            nn.GELU(),
        )
        self.linear_0 = nn.Linear(hidden_dim, 11)

        # Replace NCA and message passing with local attention stack
        # hidden_dim = 72 (64 + 8), so we need num_heads that divides evenly
        # and gives head_dim that's a multiple of 8
        # 72 / 9 = 8 (head_dim = 8, which is good for NATTEN)
        self.attention_stack = LocalAttentionStack(
            hidden_dim=hidden_dim,
            num_layers=4,  # 4 layers of local attention
            num_heads=9,  # 9 attention heads (72 / 9 = 8)
            kernel_sizes=[3, 3, 5, 5],  # Varying kernel sizes
            dilations=[1, 2, 2, 4],  # Increasing dilation for larger receptive field
            dropout=dropout,
            context_dim=hidden_dim,
            enable_self_healing=enable_self_healing,
            death_prob=death_prob,
            gaussian_std=gaussian_std,
            spatial_corruption_prob=spatial_corruption_prob,
        )
        self.drop1 = nn.Dropout(dropout_h)
        self.available_colors: Optional[List[int]] = None
        self.d4_embed = nn.Embedding(8, dim_d4_embed)
        self.color_embed = nn.Embedding(11, feature_dim)
        
        # Initialize adaptive gradient scaler
        self.enable_adaptive_grad_scaler = enable_adaptive_grad_scaler
        if self.enable_adaptive_grad_scaler:
            self.adaptive_grad_scaler = AdaptiveGradScalerSkip(
                beta=grad_scaler_beta,
                warmup_steps=grad_scaler_warmup_steps,
                mult=grad_scaler_mult,
                z_thresh=grad_scaler_z_thresh,
                eps=1e-12
            )

    @jaxtyped(typechecker=beartype)
    def forward(self, colors: ColorGrid, d4_transform_type: D4TransformType,
    step_type: Literal["train", "val"] = "train") -> OneHot11:
        """Return combined logits with channel 0 = mask (-1), channels 1..10 = colors 0..9.
        Shape: [B, 30, 30, 11]
        """
        with torch.no_grad():
            order2_raw = self.order2_encoder(colors)
        color_embed = self.color_embed(colors + 1) # shape [B, 30, 30, 64]
        o1 = self.feature_processor(order2_raw) + color_embed # shape [B, 30, 30, 64]
        d4_embed = self.d4_embed(d4_transform_type) # shape [B, 8]
        # Broadcast d4_embed to match spatial dimensions [B, 8] -> [B, 30, 30, 8]
        d4_embed = d4_embed.unsqueeze(1).unsqueeze(2).expand(-1, 30, 30, -1)
        o = torch.cat([o1, d4_embed], dim=-1) # [B, 30, 30, 64 + 8]
        h = o
        self.frame_capture.capture(self.linear_0(h), round_idx=-1)  # Capture initial state
        # Iterative refinement through local attention
        for t in range(self.num_message_rounds):
            apply_noise = t < max(0, self.num_message_rounds - self.num_final_steps)
            
            # Early stopping during training
            if self.training and (t > self.num_message_rounds - self.num_final_steps) and (random.random() < self.num_rounds_stop_prob):
                break
                
            if apply_noise:
                h = self.drop1(h)
                
            # Store previous state for residual
            h_prev = h
            
            # Apply one step of attention stack
            h = self.attention_stack(
                h, 
                context=o,  # Pass original features as context
                apply_noise=apply_noise,
                num_steps=1,  # Apply one layer per iteration
            )
            
            # Strong residual connection with small update
            h = h_prev + 0.1 * (h - h_prev)
            
            # Maintain connection to input
            h = h + 0.01 * o
            
            # Capture frame for visualization
            self.frame_capture.capture(self.linear_0(h), round_idx=t)
        combined_logits = self.linear_0(h)
        
        # Logits explosion detection
        current_logits_abs_mean = float(combined_logits.abs().mean().item())
        try:
            self.log(f"{step_type}_logits_abs_mean", current_logits_abs_mean)  # type: ignore
        except Exception:
            pass  # Logging not available outside trainer
        self.last_logits_abs_mean = current_logits_abs_mean
        
        self.frame_capture.capture(combined_logits, round_idx=self.num_message_rounds)  # Capture final state
        return combined_logits

    @jaxtyped(typechecker=beartype)
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="train")

    @jaxtyped(typechecker=beartype)
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        self.frame_capture.clear()
        self.frame_capture.enable()
        out = self.step(batch, batch_idx, step_type="val")
        self.frame_capture.disable()
        self.frame_capture.to_gif(f"gifs/message_rounds_{self.current_epoch}.gif", duration_ms=100)
        logger.info(f"Animation saved to message_rounds_{self.current_epoch}.gif")
        return out

    @jaxtyped(typechecker=beartype)
    def step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        step_type: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        i = batch_to_dataclass(batch)

        # Compute losses for color predictions (unchanged)
        metrics: Dict[int, Dict[str, float]] = {}
        logits = self(i.inp.col, i.inp.transform_idx, step_type=step_type)
        logits_perm = logits.permute(0, 3, 1, 2)  # [B, 11, 30, 30]
        loss = cross_entropy_shifted(logits_perm, i.out.col.long(), start_index=-1)
        
        # Loss explosion detection
        current_loss = float(loss.item())
        if self.last_max_loss is not None and current_loss > self.last_max_loss:
            print(f"\n!!! LOSS EXPLOSION DETECTED !!!")
            print(f"Previous loss: {self.last_max_loss:.6f}")
            print(f"Current loss: {current_loss:.6f}")
            print(f"Ratio: {current_loss / self.last_max_loss:.2f}x")
            print(f"Epoch: {self.current_epoch}, Batch: {batch_idx}")
            print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
        
            self.last_max_loss = current_loss
        
        image_metrics(i.out, logits, metrics, prefix=step_type)

        # Visualize every 10 epochs
        if self.force_visualize or (self.current_epoch % 10 == 0 and batch_idx == 0):
            self.visualize_predictions(
                i,
                i.out.col,
                logits,
                f"step-{step_type}",
            )
        # Convert float metrics to int for compatibility
        int_metrics: Dict[int, Dict[str, int]] = {}
        for idx, m in metrics.items():
            int_metrics[idx] = {k: int(v) for k, v in m.items()}

        if step_type == "train":
            self.training_index_metrics = int_metrics
        else:
            self.validation_index_metrics = int_metrics

        # Only log metrics during training or validation, not during manual test evaluation
        try:
            self.log_metrics(int_metrics, step_type)
            self.log(f"{step_type}_total_loss", loss)  # type: ignore
        except Exception:
            pass

        return loss

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        # First apply adaptive gradient scaling for spike detection if enabled
        if self.enable_adaptive_grad_scaler:
            scale, spiked = self.adaptive_grad_scaler.maybe_rescale_(self.parameters())
            
            # Log spike events and scale
            if spiked:
                self.log("grad_spike_detected", 1.0, on_step=True, prog_bar=False)
                self.log("grad_spike_scale", scale, on_step=True, prog_bar=False)
            else:
                self.log("grad_spike_detected", 0.0, on_step=True, prog_bar=False)
        
        # Then apply standard gradient clipping
        if gradient_clip_val is None or gradient_clip_val <= 0:
            return
        parameters = [p for p in self.parameters() if p.grad is not None]
        if not parameters:
            return
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
        )
        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=gradient_clip_val, norm_type=2.0
        )
        # Lightweight logging each step
        self.log("grad_total_norm", total_norm, on_step=True, prog_bar=False)

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
        self.log("val_epoch_loss", float(val_loss), prog_bar=True)  # type: ignore
        self.log("val_epoch_color_acc", float(val_output_color_acc), prog_bar=True)  # type: ignore
        self.log("val_epoch_mask_acc", float(val_output_mask_acc), prog_bar=True)  # type: ignore
        self.log("val_input_color_acc", float(val_input_color_acc))  # type: ignore
        self.log("val_output_color_acc", float(val_output_color_acc))  # type: ignore
        self.log("val_input_mask_acc", float(val_input_mask_acc))  # type: ignore
        self.log("val_output_mask_acc", float(val_output_mask_acc))  # type: ignore

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
                out_pix_incor = metrics["train_n_incorrect_num_color"]
                out_msk_incor = metrics["train_n_incorrect_num_mask"]

                all_perfect = "✓" if out_pix_incor == 0 else "✗"
                print(
                    f"{idx:<8} {out_pix_incor:<12} {out_msk_incor:<12} {all_perfect:<12}"
                )

        # Clear training metrics for next epoch
        self.training_index_metrics = {}

    def configure_optimizers(self):  # type: ignore[override]
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
                return 0.1
            elif epoch < 1000:
                return 0.01
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
        targets: ColorGrid,
        predictions: OneHot11,
        prefix: str,
        start_index: int = -1,
    ) -> None:
        """Visualize predictions vs ground truth for training."""
        # Only visualize the first example in the batch
        b = 0  # First example only
        pred_colors = predictions[b].argmax(dim=-1).cpu() + start_index
        true_colors = targets[b].cpu()

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

    print("\n=== Order2-based Model with Local Attention (ex36 with NATTEN) ===")
    print(f"Training on 'train' subset of {filename}")
    print(f"Evaluating on 'test' subset of {filename}")
    print("Architecture: color -> order2 -> local attention stack -> color")
    print(f"NATTEN available: {NATTEN_AVAILABLE}")
    print(f"D4 Augmentation: {'Enabled' if training_config.use_d4_augmentation else 'Disabled'}")
    if training_config.use_d4_augmentation:
        print(f"  Mode: {'Deterministic (8x data)' if training_config.d4_deterministic else 'Random'}")
    print(f"Adaptive Gradient Scaler: {'Enabled' if training_config.enable_adaptive_grad_scaler else 'Disabled'}")
    if training_config.enable_adaptive_grad_scaler:
        print(f"  Beta: {training_config.grad_scaler_beta}, Warmup: {training_config.grad_scaler_warmup_steps}, Mult: {training_config.grad_scaler_mult}, Z-thresh: {training_config.grad_scaler_z_thresh}")
    
    rich.print(training_config)

    # Load train subset using data_loader.py
    (
        train_inputs,
        train_outputs,
        train_input_features,
        train_output_features,
        train_indices,
    ) = prepare_dataset(
        "processed_data/train_all.npz",
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

    # Create dataloaders - use D4 augmentation for training if enabled
    if training_config.use_d4_augmentation:
        print(f"\nUsing D4 augmentation for training (deterministic={training_config.d4_deterministic})")
        train_loader = create_d4_augmented_dataloader(
            train_inputs,
            train_outputs,
            inputs_features=train_input_features,
            outputs_features=train_output_features,
            indices=train_indices,
            batch_size=len(train_inputs) * (8 if training_config.d4_deterministic else 1),  # All transforms in one batch if deterministic
            shuffle=True,
            num_workers=0,  # Eliminate multiprocessing overhead for small datasets
            augment=True,
            deterministic_augmentation=training_config.d4_deterministic,
            repeat_factor=5,
        )
    else:
        print("\nTraining without D4 augmentation")
        train_loader = create_dataloader(
            train_inputs,
            train_outputs,
            batch_size=len(train_inputs),
            shuffle=True,
            num_workers=0,  # Eliminate multiprocessing overhead for small datasets
            inputs_features=train_input_features,
            outputs_features=train_output_features,
            indices=train_indices,
            repeat_factor=10,
        )
    
    # Validation and test loaders don't use augmentation
    val_loader = create_d4_augmented_dataloader(
        test_inputs,
        test_outputs,
        inputs_features=test_input_features,
        outputs_features=test_output_features,
        indices=test_indices,
        batch_size=len(test_inputs),  # No augmentation for validation/test
        shuffle=False,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        augment=False,  # No augmentation for validation
    )
    test_loader = create_d4_augmented_dataloader(
        test_inputs,
        test_outputs,
        inputs_features=test_input_features,
        outputs_features=test_output_features,
        indices=test_indices,
        batch_size=len(test_inputs),  # No augmentation for validation/test
        shuffle=False,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        augment=False,  # No augmentation for test
    )

    # Create model with config hyperparameters
    model = MainModel(
        feature_dim=training_config.feature_dim,
        num_message_rounds=training_config.num_message_rounds,
        dropout=training_config.dropout,
        dropout_h=training_config.dropout_h,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        enable_self_healing=training_config.enable_self_healing,
        death_prob=training_config.death_prob,
        gaussian_std=training_config.gaussian_std,
        spatial_corruption_prob=training_config.spatial_corruption_prob,
        num_final_steps=training_config.num_final_steps,
        num_rounds_stop_prob=training_config.num_rounds_stop_prob,
        enable_adaptive_grad_scaler=training_config.enable_adaptive_grad_scaler,
        grad_scaler_beta=training_config.grad_scaler_beta,
        grad_scaler_warmup_steps=training_config.grad_scaler_warmup_steps,
        grad_scaler_mult=training_config.grad_scaler_mult,
        grad_scaler_z_thresh=training_config.grad_scaler_z_thresh,
    )

    # Compute available colors from training data and set constraints
    all_colors_non_mask: torch.Tensor = all_colors[all_colors != -1]
    available_colors: torch.Tensor = torch.unique(all_colors_non_mask)  # type: ignore[call-arg]
    print(f"Available colors for {filename}: {available_colors}")
    model.available_colors = available_colors.tolist()  # type: ignore[no-untyped-call]

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

    # Optional Weights & Biases logging (enabled if available)
    wandb_logger = None
    if WandbLogger is not None:
        # Generate hash of training config for unique identification
        config_dict = training_config.model_dump()
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create run name with config hash
        run_name = f"ex36-{filename}-{config_hash}"
        
        # Initialize wandb logger with all config parameters
        wandb_logger = WandbLogger(
            project="arcagi", 
            name=run_name,
            config=config_dict  # Pass all training config parameters as hyperparameters
        )

    if wandb_logger is not None:
        trainer = pl.Trainer(
            max_epochs=training_config.max_epochs,
            callbacks=[checkpoint_callback],
            default_root_dir=training_config.checkpoint_dir,
            gradient_clip_val=1.00,
            gradient_clip_algorithm="norm",
            logger=wandb_logger,  # type: ignore[arg-type]
            log_every_n_steps=1,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=training_config.max_epochs,
            callbacks=[checkpoint_callback],
            default_root_dir=training_config.checkpoint_dir,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            log_every_n_steps=1,
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
    # Save both model state dict and minimal metadata (avoid typing complaints)
    try:
        hyperparams = dict(model.hparams)  # type: ignore[arg-type]
    except Exception:
        hyperparams = {}
    model_save_data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "filename": filename,
        "hyperparameters": hyperparams,
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
