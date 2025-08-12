import math
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
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydanclick import from_pydantic
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint

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
BatchTuple: TypeAlias = (
    "tuple[OneHot11, OneHot11, Features45, Features45, torch.Tensor]"
)

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

# Import Order2Features from lib
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lib.order2 import Order2Features


@jaxtyped(typechecker=beartype)
def apply_binary_noise(
    features: Union[Features45, HiddenGrid], noise_prob: float, training: bool
) -> Union[Features45, HiddenGrid]:
    """Apply binary noise to features during training only."""
    if training and noise_prob > 0:
        noise_mask = torch.rand_like(features) < noise_prob
        features = torch.where(noise_mask, 1 - features, features)
    return features


@jaxtyped(typechecker=beartype)
def apply_color_constraints(
    logits: HiddenGrid,
    available_colors: torch.Tensor | list[int] | None = None,
    num_classes: int = 10,
) -> HiddenGrid:
    """Apply hard constraints to ensure only available colors are predicted.

    Expects color logits shaped [B, H, W, num_classes].
    """
    if available_colors is None:
        return logits

    if isinstance(available_colors, list):
        available_colors = torch.tensor(available_colors, device=logits.device)

    assert available_colors.dtype in [
        torch.int,
        torch.long,
        torch.int32,
        torch.int64,
    ]
    max_color = (
        available_colors.max() if available_colors.numel() > 0 else torch.tensor(-1)
    )
    assert int(max_color.item()) < num_classes

    mask = torch.ones(num_classes, device=logits.device) * -1e10
    mask[available_colors] = 0
    logits = logits + mask.view(1, 1, 1, -1)
    return logits


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


class Order2ToColorDecoder(nn.Module):
    """
    Decoder network that converts order2 features back to color predictions.
    Returns color and mask logits separately.
    """

    def __init__(
        self: "Order2ToColorDecoder",
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
        self.color_head = nn.Linear(hidden_dim // 2, num_classes)
        self.mask_head = nn.Linear(hidden_dim // 2, 1)

    @jaxtyped(typechecker=beartype)
    def forward(self, order2_features: Features45) -> Tuple[HiddenGrid, torch.Tensor]:
        h = self.decoder(order2_features)
        color_logits = self.color_head(h)
        mask_logits = self.mask_head(h)
        return color_logits, mask_logits


class SpatialMessagePassing(nn.Module):
    """Efficient spatial message passing for local consistency."""

    def __init__(
        self: "SpatialMessagePassing", hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Ensure groups divides hidden_dim evenly
        groups = hidden_dim // 16 if hidden_dim >= 16 else 1
        # Find the largest divisor of hidden_dim that's <= groups
        while groups > 1 and hidden_dim % groups != 0:
            groups -= 1
        
        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=groups,
        )
        self.norm = RMSNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, h: HiddenGrid) -> HiddenGrid:
        h_conv = h.permute(0, 3, 1, 2)
        messages = self.conv(h_conv)
        messages = messages.permute(0, 2, 3, 1)
        messages = self.norm(messages)
        messages = self.activation(messages)
        messages = self.dropout(messages)
        return messages


class SelfHealingNoise(nn.Module):
    def __init__(
        self: "SelfHealingNoise",
        death_prob: float = 0.05,
        gaussian_std: float = 0.1,
        salt_pepper_prob: float = 0.02,
        spatial_corruption_prob: float = 0.03,
    ) -> None:
        super().__init__()
        self.death_prob = death_prob
        self.gaussian_std = gaussian_std
        self.salt_pepper_prob = salt_pepper_prob
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
        if self.salt_pepper_prob > 0:
            salt_pepper_mask = torch.rand_like(h) < self.salt_pepper_prob
            salt_mask = torch.rand_like(h) > 0.5
            extreme_values = torch.where(salt_mask, 1.0, -1.0)
            h = torch.where(salt_pepper_mask, extreme_values, h)
        if self.spatial_corruption_prob > 0:
            for bi in range(b):
                if random.random() < self.spatial_corruption_prob:
                    grid_size = random.randint(1, 5)  # Random size 1-5
                    y = torch.randint(0, height - grid_size + 1, (1,)).item()
                    x = torch.randint(0, width - grid_size + 1, (1,)).item()
                    h[bi, y : y + grid_size, x : x + grid_size, :] = 0
        return h


class NeuralCellularAutomata2(nn.Module):
    def __init__(
        self: "NeuralCellularAutomata2",
        hidden_dim: int,
        dropout: float = 0.05,
        enable_self_healing: bool = True,
        death_prob: float = 0.05,
        gaussian_std: float = 0.05,
        salt_pepper_prob: float = 0.05,
        spatial_corruption_prob: float = 0.05,
        num_final_steps: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_final_steps = num_final_steps

        self.perception_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim * 3,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )

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

        perception_dim = hidden_dim * 3
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_dim, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
        )
        # Attention components (currently unused but available for future use)
        self.lin_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.update_probability = 0.5
        self.drop = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def stochastic_update(
        self,
        x: NCALatent,
        dx: NCALatent,
    ) -> NCALatent:
        if not self.training:
            return x + dx
        update_mask = (
            torch.rand_like(x[:, :1, :, :]) < self.update_probability
        ).float()
        return x + dx * update_mask

    @jaxtyped(typechecker=beartype)
    def apply_local_spatial_attention(self, h: HiddenGrid) -> HiddenGrid:
        """Apply local spatial attention within 3x3 neighborhoods - FAST VERSION."""
        B, H, W, C = h.shape

        # Generate Q, K, V
        qkv = self.lin_qkv(h)  # [B, H, W, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, H, W, C]

        # Convert to [B, C, H, W] for convolution operations
        q = q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        k = k.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        v = v.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # FAST METHOD 1: Depthwise Convolution-based Attention
        # This is much faster than unfold/fold operations

        # Pad the tensors for 3x3 neighborhoods
        pad = 1
        k_padded = F.pad(k, (pad, pad, pad, pad), mode="constant", value=0)
        v_padded = F.pad(v, (pad, pad, pad, pad), mode="constant", value=0)

        # Initialize attention output
        h_attn = torch.zeros_like(v)

        # Process each position in the 3x3 grid
        positions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # Collect attention scores for all positions
        scores_list = []
        for dy, dx in positions:
            # Extract shifted keys
            k_shift = k_padded[:, :, pad + dy : pad + dy + H, pad + dx : pad + dx + W]
            # Compute dot product attention scores
            scores = (q * k_shift).sum(dim=1, keepdim=True) / math.sqrt(
                C
            )  # [B, 1, H, W]
            scores_list.append(scores)

        # Stack and normalize scores
        scores = torch.cat(scores_list, dim=1)  # [B, 9, H, W]
        attn_weights = F.softmax(scores, dim=1)  # [B, 9, H, W]

        # Apply attention weights to values
        for i, (dy, dx) in enumerate(positions):
            v_shift = v_padded[:, :, pad + dy : pad + dy + H, pad + dx : pad + dx + W]
            weight = attn_weights[:, i : i + 1, :, :]  # [B, 1, H, W]
            h_attn = h_attn + weight * v_shift

        # Convert back to [B, H, W, C]
        h_attn = h_attn.permute(0, 2, 3, 1).contiguous()

        return h_attn

    @jaxtyped(typechecker=beartype)
    def apply_local_spatial_attention_depthwise(self, h: HiddenGrid) -> HiddenGrid:
        """Even faster version using depthwise separable convolutions."""
        B, H, W, C = h.shape

        # Generate Q, K, V
        qkv = self.lin_qkv(h)  # [B, H, W, 3*C]
        qkv = qkv.permute(0, 3, 1, 2)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B, C, H, W]

        # FAST METHOD 2: Single-pass depthwise convolution
        # Approximate local attention with learned 3x3 kernels

        # Instead of computing explicit attention, use depthwise conv
        # This is an approximation but much faster
        if not hasattr(self, "local_attn_conv"):
            # Create depthwise conv layers on first use
            self.local_attn_conv = nn.Conv2d(
                self.hidden_dim * 3,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                groups=self.hidden_dim,
            )
            # Initialize with small weights
            nn.init.xavier_uniform_(self.local_attn_conv.weight, gain=0.1)

        # Concatenate q, k, v and apply depthwise convolution
        qkv_concat = torch.cat([q, k, v], dim=1)  # [B, 3*C, H, W]
        h_attn = self.local_attn_conv(qkv_concat)  # [B, C, H, W]

        # Apply activation and convert back
        h_attn = F.gelu(h_attn)
        h_attn = h_attn.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        return h_attn

    @jaxtyped(typechecker=beartype)
    def apply_local_spatial_attention_im2col(self, h: HiddenGrid) -> HiddenGrid:
        """Fastest version using im2col-style operations with batch matrix multiplication."""
        B, H, W, C = h.shape

        # Generate Q, K, V
        qkv = self.lin_qkv(h)  # [B, H, W, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, H, W, C]

        # Reshape to [B*H*W, C] for efficient processing
        q_flat = q.reshape(B * H * W, C)  # [B*H*W, C]

        # Convert to [B, C, H, W] for unfold
        k = k.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        v = v.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Use unfold for k and v (this is optimized in PyTorch)
        k_unfold = F.unfold(k, kernel_size=3, padding=1)  # [B, C*9, H*W]
        v_unfold = F.unfold(v, kernel_size=3, padding=1)  # [B, C*9, H*W]

        # Reshape for batch matrix multiplication
        k_unfold = k_unfold.transpose(1, 2).reshape(B * H * W, C, 9)  # [B*H*W, C, 9]
        v_unfold = v_unfold.transpose(1, 2).reshape(B * H * W, C, 9)  # [B*H*W, C, 9]

        # Compute attention scores using batch matrix multiplication
        # [B*H*W, 1, C] @ [B*H*W, C, 9] -> [B*H*W, 1, 9]
        scores = torch.bmm(q_flat.unsqueeze(1), k_unfold) / math.sqrt(C)
        scores = scores.squeeze(1)  # [B*H*W, 9]

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B*H*W, 9]

        # Apply attention to values
        # [B*H*W, 1, 9] @ [B*H*W, 9, C] -> [B*H*W, 1, C]
        v_unfold = v_unfold.transpose(1, 2)  # [B*H*W, 9, C]
        h_attn = torch.bmm(attn_weights.unsqueeze(1), v_unfold)  # [B*H*W, 1, C]

        # Reshape back to spatial dimensions
        h_attn = h_attn.squeeze(1).reshape(B, H, W, C)  # [B, H, W, C]

        return h_attn

    @jaxtyped(typechecker=beartype)
    def forward(self, h: HiddenGrid, apply_noise: bool = True) -> HiddenGrid:
        if apply_noise and self.self_healing_noise is not None:
            h = self.self_healing_noise(h)
        x = h.permute(0, 3, 1, 2).contiguous()
        perceived = self.perception_conv(x)
        dx = self.update_net(perceived)
        x_new = self.stochastic_update(x, dx)
        h_new = x_new.permute(0, 2, 3, 1).contiguous()

        # Apply local spatial attention (using fastest method)
        h_attn = self.apply_local_spatial_attention_im2col(h_new)

        # Residual connection with attention
        h_new = h_new + self.drop(h_attn)

        return h_new


class TrainingConfig(BaseModel):
    """Configuration for training the order2-based neural cellular automata model."""

    # Data and checkpoint paths
    data_dir: str = "/tmp/arc_data"
    checkpoint_dir: str = "order2_checkpoints"

    # Training parameters
    max_epochs: int = 2000
    lr: float = 0.005
    weight_decay: float = 1e-8
    hidden_dim: int = 64
    num_message_rounds: int = 32

    # Self-healing noise parameters
    enable_self_healing: bool = True
    death_prob: float = 0.05
    gaussian_std: float = 0.05
    salt_pepper_prob: float = 0.05
    spatial_corruption_prob: float = 0.05

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
    
    # D4 augmentation parameters
    use_d4_augmentation: bool = False
    d4_deterministic: bool = True  # If True, cycles through all 8 transformations


@jaxtyped(typechecker=beartype)
def batch_to_dataclass(
    batch: List[torch.Tensor],
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
    last_max_loss: float | None = None
    last_logits_abs_mean: float | None = None

    def __init__(
        self,
        hidden_dim: int = 64,
        num_message_rounds: int = 6,
        num_classes: int = 10,
        dropout: float = 0.0,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        temperature: float = 1.0,
        enable_self_healing: bool = True,
        death_prob: float = 0.02,
        gaussian_std: float = 0.05,
        salt_pepper_prob: float = 0.01,
        spatial_corruption_prob: float = 0.01,
        num_final_steps: int = 12,
        dim_nca: int = 64,
        decoder_hidden_dim: int = 64,
        dim_feat: int = 45,
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

        # Order2 feature extractor
        self.order2_encoder = Order2Features()

        self.rms_norm = RMSNorm(hidden_dim)

        # Feature processing and projection to NCA space
        self.feature_processor = nn.Sequential(
            nn.Linear(dim_feat, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
        )
        self.noise_prob = 0.05
        self.linear_0 = nn.Linear(hidden_dim, 11)

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(dim_nca, hidden_dim)

        self.nca = NeuralCellularAutomata2(
            hidden_dim,
            dropout=dropout,
            enable_self_healing=enable_self_healing,
            death_prob=death_prob,
            gaussian_std=gaussian_std,
            salt_pepper_prob=salt_pepper_prob,
            spatial_corruption_prob=spatial_corruption_prob,
            num_final_steps=num_final_steps,
        )
        self.message_passing = SpatialMessagePassing(hidden_dim, dropout=dropout)
        self.order2_projection = nn.Linear(hidden_dim, dim_feat)
        self.order2_decoder = Order2ToColorDecoder(
            hidden_dim=decoder_hidden_dim, num_classes=num_classes, dropout=dropout
        )
        self.drop = nn.Dropout(dropout)
        self.available_colors: list[int] | None = None
        self.arsinh_norm1 = ArsinhNorm(hidden_dim)
        self.arsinh_norm2 = ArsinhNorm(hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, colors: ColorGrid) -> OneHot11:
        """Return combined logits with channel 0 = mask (-1), channels 1..10 = colors 0..9.
        Shape: [B, 30, 30, 11]
        """
        with torch.no_grad():
            order2_raw = self.order2_encoder(colors)
            # order2_noise = apply_binary_noise(
            #     order2_raw, noise_prob=self.noise_prob, training=self.training
            # )

        o = self.feature_processor(order2_raw)  # shape [B, 30, 30, 64]
        h = o
        # signal = self.order2_projection(h)  # shape [B, 30, 30, 45]
        # combined_logits = self.linear_0(signal)
        # combined_logits = self.linear_0(h)
        # assert combined_logits.shape == (colors.shape[0], 30, 30, 11)
        # noisy_steps = self.num_message_rounds - self.nca.num_final_steps
        for t in range(self.num_message_rounds):
            apply_noise = t < max(
                0, self.num_message_rounds - self.nca.num_final_steps
            )
            # x_nca = self.linear_1(h)
            # if apply_noise:
            #     x_nca = self.drop(x_nca)
            h = h + 0.1 * self.nca(h, apply_noise=apply_noise)
            m = self.linear_1(self.message_passing(h))
            if apply_noise:
                m = self.drop(m)
            m = self.arsinh_norm2(m)
            h = h + 0.1 * m
            # if t % 2 == 0:  # More frequent normalization
            #     h = self.rms_norm(h)
            h = self.arsinh_norm1(h)
            # signal = self.order2_projection(h)
        #     h_update = self.linear_2(x_nca)
        #     if apply_noise:
        #         h_update = self.drop(h_update)
        #     h = h + h_update
        #     h = h + self.message_passing(h)

        # signal = self.order2_projection(h)
        # color_logits, mask_logits = self.order2_decoder(signal)
        # color_logits = color_logits / max(self.temperature, 1e-6)
        # if self.available_colors is not None:
        #     color_logits = apply_color_constraints(color_logits, self.available_colors)
        # combined_logits = torch.cat([mask_logits, color_logits], dim=-1)
        combined_logits = self.linear_0(h)
        
        # Logits explosion detection
        current_logits_abs_mean = float(combined_logits.abs().mean().item())
        self.log(f"logits_abs_mean", current_logits_abs_mean)  # type: ignore
        # print(f"Current logits abs mean: {current_logits_abs_mean:.6f}")
        if self.last_logits_abs_mean is not None and current_logits_abs_mean > 2 * self.last_logits_abs_mean:
            print(f"\n!!! LOGITS EXPLOSION DETECTED IN FORWARD !!!")
            print(f"Previous logits abs mean: {self.last_logits_abs_mean:.6f}")
            print(f"Current logits abs mean: {current_logits_abs_mean:.6f}")
            print(f"Ratio: {current_logits_abs_mean / self.last_logits_abs_mean:.2f}x")
            print(f"Epoch: {self.current_epoch}")
            print(f"Logits stats: min={combined_logits.min().item():.6f}, max={combined_logits.max().item():.6f}")
            print(f"Hidden h stats: min={h.min().item():.6f}, max={h.max().item():.6f}, mean={h.mean().item():.6f}")
        
        self.last_logits_abs_mean = current_logits_abs_mean
        
        return combined_logits

    @jaxtyped(typechecker=beartype)
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="train")

    @jaxtyped(typechecker=beartype)
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="val")

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
        logits = self(i.inp.col)
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
        gradient_clip_val: float,
        gradient_clip_algorithm: str,
    ) -> None:
        # Log and clip by global norm for stability
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

    @jaxtyped(typechecker=beartype)
    def visualize_predictions(
        self,
        i: BatchData,
        targets: ColorGrid,
        predictions: OneHot11,
        prefix: str,
        start_index: int = -1,
    ) -> None:
        """Visualize predictions vs ground truth for training."""
        # Visualize first N items in the batch for simplicity
        batch_size = int(targets.size(0))
        for b in range(batch_size):
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

    print("\n=== Order2-based Model (ex33 with D4 Augmentation) ===")
    print(f"Training on 'train' subset of {filename}")
    print(f"Evaluating on 'test' subset of {filename}")
    print("Architecture: color -> order2 -> NCA -> order2 -> color")
    print(f"D4 Augmentation: {'Enabled' if training_config.use_d4_augmentation else 'Disabled'}")
    if training_config.use_d4_augmentation:
        print(f"  Mode: {'Deterministic (8x data)' if training_config.d4_deterministic else 'Random'}")

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
            repeat_factor=10,
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
        hidden_dim=training_config.hidden_dim,
        num_message_rounds=training_config.num_message_rounds,
        dropout=training_config.dropout,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        temperature=training_config.temperature,
        enable_self_healing=training_config.enable_self_healing,
        death_prob=training_config.death_prob,
        gaussian_std=training_config.gaussian_std,
        salt_pepper_prob=training_config.salt_pepper_prob,
        spatial_corruption_prob=training_config.spatial_corruption_prob,
        num_final_steps=training_config.num_final_steps,
    )

    # Compute available colors from training data and set constraints
    all_colors_non_mask: torch.Tensor = all_colors[all_colors != -1]
    available_colors: torch.Tensor = torch.unique(all_colors_non_mask)  # type: ignore[call-arg]
    print(f"Available colors for {filename}: {available_colors}")
    model.available_colors = available_colors.tolist()  # type: ignore[no-untyped-call]
    model.noise_prob = training_config.noise_prob

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
        try:
            wandb_logger = WandbLogger(project="arcagi", name=f"ex32-{filename}")
        except Exception:
            wandb_logger = None

    if wandb_logger is not None:
        trainer = pl.Trainer(
            max_epochs=training_config.max_epochs,
            callbacks=[checkpoint_callback],
            default_root_dir=training_config.checkpoint_dir,
            gradient_clip_val=1.0,
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
