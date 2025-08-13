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


class FileContextEmbedding(nn.Module):
    """Maps filenames to context embeddings for hypernetwork modulation."""

    def __init__(self, context_dim: int = 32):
        super().__init__()
        self.context_dim = context_dim
        self.filename_to_id: Dict[str, int] = {}
        self.embedding = nn.Embedding(100, context_dim)  # Support up to 100 files

        # Initialize embeddings to small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def register_filename(self, filename: str) -> int:
        """Register a filename and return its ID."""
        if filename not in self.filename_to_id:
            new_id = len(self.filename_to_id)
            self.filename_to_id[filename] = new_id
        return self.filename_to_id[filename]

    def forward(self, file_ids: torch.Tensor) -> torch.Tensor:
        """Get context embeddings for a batch of file IDs.

        Args:
            file_ids: [B] tensor of file IDs

        Returns:
            Context embeddings [B, context_dim]
        """
        return self.embedding(file_ids)


class BilinearLoRA(nn.Module):
    """Efficient bilinear LoRA for computing ΔW(c) = ((xA) ⊙ (Bc)) V^T."""

    def __init__(self, in_dim: int, out_dim: int, context_dim: int, rank: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        # LoRA parameters
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(context_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute bilinear LoRA: ((xA) ⊙ (Bc)) V^T.

        Args:
            x: Input tensor [..., in_dim]
            c: Context tensor [B, context_dim] or [context_dim]

        Returns:
            Output tensor [..., out_dim]
        """
        # Project input: [..., in_dim] -> [..., rank]
        x_proj = x @ self.A

        # Project context: [B, context_dim] -> [B, rank] or [context_dim] -> [rank]
        c_proj = c @ self.B

        # Handle broadcasting for context
        if c_proj.dim() == 1:
            c_proj = c_proj.unsqueeze(0)  # [1, rank]

        # Element-wise multiplication with broadcasting
        if x_proj.dim() > c_proj.dim():
            # Add dimensions to c_proj to match x_proj
            for _ in range(x_proj.dim() - c_proj.dim()):
                c_proj = c_proj.unsqueeze(1)

        bilinear = x_proj * c_proj  # [..., rank]

        # Up-project: [..., rank] -> [..., out_dim]
        output = bilinear @ self.V.T

        return output


class ContextModulatedLinear(nn.Module):
    """Linear layer with context-dependent modulation via bilinear LoRA."""

    def __init__(self, in_dim: int, out_dim: int, context_dim: int, rank: int = 8):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.bilinear_lora = BilinearLoRA(in_dim, out_dim, context_dim, rank)

        # Gating function α(c)
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Initialize gate to output near-zero values
        nn.init.constant_(self.gate[2].weight, -5.0)  # Large negative bias for sigmoid
        nn.init.constant_(self.gate[2].bias, -5.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply context-modulated linear transformation.

        Args:
            x: Input tensor [..., in_dim]
            c: Context tensor [B, context_dim]

        Returns:
            Output tensor [..., out_dim]
        """
        # Base transformation
        base_output = self.base_linear(x)

        # Context-dependent delta
        delta = self.bilinear_lora(x, c)

        # Gating
        alpha = self.gate(c)  # [B, 1]
        if alpha.dim() == 2 and x.dim() > 2:
            # Expand alpha to match x dimensions
            for _ in range(x.dim() - 2):
                alpha = alpha.unsqueeze(1)

        # Apply gated modulation
        output = base_output + alpha * delta

        return output


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

    # Multi-file training parameters
    experiment_type: str = "multi_file"  # "single_file" or "multi_file"
    filenames: List[str] = ["28e73c20.json", "00d62c1b.json"]  # For multi_file mode

    # Context-aware attention parameters
    context_dim: int = 32
    lora_rank: int = 8

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


class ContextModulatedNeighborhoodAttention(nn.Module):
    """Context-modulated neighborhood attention using bilinear LoRA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int,
        context_dim: int,
        lora_rank: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Context-modulated QKV projections using bilinear LoRA
        self.qkv_modulation = ContextModulatedLinear(
            embed_dim, embed_dim * 3, context_dim, lora_rank
        )

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, H, W, C]
            context: Context tensor [B, context_dim]
        Returns:
            Output tensor [B, H, W, C]
        """
        B, H, W, C = x.shape

        # Get context-modulated QKV
        qkv = self.qkv_modulation(x, context)  # [B, H, W, 3C]

        # Reshape for attention computation
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # [3, B, num_heads, H, W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, H, W, head_dim]

        # Apply local attention with unfold operation
        # Unfold creates windows of size kernel_size around each position
        pad = self.dilation * (self.kernel_size - 1) // 2
        
        # Pad k and v for neighborhood extraction
        if pad > 0:
            k_padded = F.pad(k, (0, 0, pad, pad, pad, pad), mode='constant', value=0)
            v_padded = F.pad(v, (0, 0, pad, pad, pad, pad), mode='constant', value=0)
        else:
            k_padded = k
            v_padded = v

        # Extract neighborhoods using unfold
        # k_padded: [B, num_heads, H+2*pad, W+2*pad, head_dim]
        k_unfolded = F.unfold(
            k_padded.reshape(B * self.num_heads, H + 2*pad, W + 2*pad, self.head_dim).permute(0, 3, 1, 2),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=1
        )  # [B*num_heads, head_dim*kernel_size^2, H*W]
        
        v_unfolded = F.unfold(
            v_padded.reshape(B * self.num_heads, H + 2*pad, W + 2*pad, self.head_dim).permute(0, 3, 1, 2),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=1
        )  # [B*num_heads, head_dim*kernel_size^2, H*W]

        # Reshape unfolded tensors
        k_unfolded = k_unfolded.reshape(B, self.num_heads, self.head_dim, self.kernel_size**2, H*W)
        k_unfolded = k_unfolded.permute(0, 1, 4, 3, 2)  # [B, num_heads, H*W, kernel_size^2, head_dim]
        
        v_unfolded = v_unfolded.reshape(B, self.num_heads, self.head_dim, self.kernel_size**2, H*W)
        v_unfolded = v_unfolded.permute(0, 1, 4, 3, 2)  # [B, num_heads, H*W, kernel_size^2, head_dim]

        # Reshape q for attention
        q = q.reshape(B, self.num_heads, H*W, self.head_dim)

        # Compute attention scores
        attn = torch.matmul(q.unsqueeze(3), k_unfolded.transpose(-2, -1))  # [B, num_heads, H*W, 1, kernel_size^2]
        attn = attn.squeeze(3) * self.scale  # [B, num_heads, H*W, kernel_size^2]

        # Apply softmax
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn.unsqueeze(3), v_unfolded).squeeze(3)  # [B, num_heads, H*W, head_dim]

        # Reshape back
        out = out.reshape(B, self.num_heads, H, W, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LocalAttentionBlock(nn.Module):
    """Local attention block with optional context modulation."""

    def __init__(
        self: "LocalAttentionBlock",
        hidden_dim: int,
        num_heads: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        file_context_dim: Optional[int] = None,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.file_context_dim = file_context_dim
        self.num_heads = num_heads

        # Input projection if context is provided
        input_dim = hidden_dim + (context_dim if context_dim else 0)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if context_dim else None

        # Pre-normalization
        self.norm1 = ArsinhNorm(hidden_dim)

        # Use context-modulated attention if file context is provided
        if file_context_dim is not None:
            self.attention = ContextModulatedNeighborhoodAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                context_dim=file_context_dim,
                lora_rank=lora_rank,
                qkv_bias=True,
                qk_scale=None,
                proj_drop=dropout,
            )
            self.use_context_modulation = True
        elif NATTEN_AVAILABLE:
            # Use standard NATTEN without context modulation
            self.attention = NeighborhoodAttention2D(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                qkv_bias=True,
                qk_scale=None,
                proj_drop=dropout,
            )
            self.use_context_modulation = False
        else:
            # Fallback to a simple local convolution if NATTEN not available
            print("Using fallback convolution instead of NATTEN")
            self.attention = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=1,  # Use single group for compatibility with small hidden_dim
                ),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
            )
            self.use_conv_fallback = True
            self.use_context_modulation = False

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
    def forward(
        self,
        h: HiddenGrid,
        context: Optional[HiddenGrid] = None,
        file_context: Optional[torch.Tensor] = None,
    ) -> HiddenGrid:
        # Handle spatial context if provided
        if context is not None and self.context_dim is not None:
            h_with_context = torch.cat([h, context], dim=-1)
            h = self.input_proj(h_with_context)

        # Pre-norm and attention
        h_normed = self.norm1(h)

        if self.use_context_modulation and file_context is not None:
            # Use context-modulated attention
            h_attn = self.attention(h_normed, file_context)
        elif hasattr(self, "use_conv_fallback") and self.use_conv_fallback:
            # Fallback convolution
            h_conv = h_normed.permute(0, 3, 1, 2)  # [B, C, H, W]
            h_attn = self.attention(h_conv)
            h_attn = h_attn.permute(0, 2, 3, 1)  # [B, H, W, C]
        else:
            # Standard NATTEN
            h_attn = self.attention(h_normed)

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
        file_context_dim: Optional[int] = None,
        lora_rank: int = 8,
        enable_self_healing: bool = True,
        death_prob: float = 0.05,
        gaussian_std: float = 0.05,
        spatial_corruption_prob: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.file_context_dim = file_context_dim
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
        self.layers = nn.ModuleList(
            [
                LocalAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    dropout=dropout,
                    context_dim=(
                        context_dim if i == 0 else None
                    ),  # Only first layer gets spatial context
                    file_context_dim=file_context_dim,  # All layers get file context
                    lora_rank=lora_rank,
                )
                for i in range(num_layers)
            ]
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: HiddenGrid,
        context: Optional[HiddenGrid] = None,
        file_context: Optional[torch.Tensor] = None,
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
            # Only pass spatial context to the first layer
            layer_context = context if i == 0 else None
            h = self.layers[i](h, context=layer_context, file_context=file_context)

        return h


@jaxtyped(typechecker=beartype)
def batch_to_dataclass(
    batch: List[torch.Tensor],
) -> BatchData:
    # Check batch length to determine what data we have
    if len(batch) == 7:
        # Multi-file training with file IDs and transform indices
        (
            inputs_one_hot,
            outputs_one_hot,
            input_features,
            output_features,
            indices,
            transform_indices,
            file_ids,
        ) = batch
    elif len(batch) == 6:
        # D4 augmented dataloader without file IDs
        (
            inputs_one_hot,
            outputs_one_hot,
            input_features,
            output_features,
            indices,
            transform_indices,
        ) = batch
        file_ids = None
    else:
        # Standard dataloader
        inputs_one_hot, outputs_one_hot, input_features, output_features, indices = (
            batch
        )
        transform_indices = None
        file_ids = None

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

    # Add file_ids to the batch data if available
    if file_ids is not None:
        inp.file_ids = file_ids  # type: ignore[attr-defined]
        out.file_ids = file_ids  # type: ignore[attr-defined]

    return BatchData(inp=inp, out=out)


def create_multi_file_dataloader(
    filenames: List[str],
    file_context_embedding: FileContextEmbedding,
    training_config: TrainingConfig,
    dataset_name: str = "train",
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a dataloader that loads examples from multiple files with file ID tracking."""

    all_inputs = []
    all_outputs = []
    all_input_features = []
    all_output_features = []
    all_indices = []
    all_file_ids = []

    # Load data from each file
    for file_idx, filename in enumerate(filenames):
        # Register filename and get ID
        file_id = file_context_embedding.register_filename(filename)

        print(f"\nLoading {dataset_name} data from {filename} (file_id={file_id})")

        # Load data for this file
        (
            inputs,
            outputs,
            input_features,
            output_features,
            indices,
        ) = prepare_dataset(
            "processed_data/train_all.npz",
            filter_filename=filename,
            use_features=True,
            dataset_name=dataset_name,
            use_train_subset=(dataset_name == "train"),
        )

        # Create file ID tensor for all examples from this file
        file_ids = torch.full((len(inputs),), file_id, dtype=torch.long)

        # Append to lists
        all_inputs.append(inputs)
        all_outputs.append(outputs)
        all_input_features.append(input_features)
        all_output_features.append(output_features)
        all_indices.append(indices)
        all_file_ids.append(file_ids)

        print(f"  Loaded {len(inputs)} examples from {filename}")

    # Concatenate all data
    combined_inputs = torch.cat(all_inputs, dim=0)
    combined_outputs = torch.cat(all_outputs, dim=0)
    combined_input_features = torch.cat(all_input_features, dim=0)
    combined_output_features = torch.cat(all_output_features, dim=0)
    combined_indices = torch.cat(all_indices, dim=0)
    combined_file_ids = torch.cat(all_file_ids, dim=0)

    print(f"\nTotal {dataset_name} examples: {len(combined_inputs)}")

    # Determine batch size
    if batch_size is None:
        batch_size = len(combined_inputs)

    # Create dataloader with D4 augmentation if enabled
    if training_config.use_d4_augmentation and dataset_name == "train":
        # Create custom dataset that includes file IDs
        class MultiFileD4Dataset(torch.utils.data.Dataset):
            def __init__(
                self,
                inputs,
                outputs,
                input_features,
                output_features,
                indices,
                file_ids,
                augment=True,
                deterministic=False,
            ):
                self.inputs = inputs
                self.outputs = outputs
                self.input_features = input_features
                self.output_features = output_features
                self.indices = indices
                self.file_ids = file_ids
                self.augment = augment
                self.deterministic = deterministic
                self.epoch = 0

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __len__(self):
                if self.augment and self.deterministic:
                    return len(self.inputs) * 8  # All 8 transforms
                return len(self.inputs)

            def __getitem__(self, idx):
                if self.augment and self.deterministic:
                    # Deterministic: cycle through all 8 transforms
                    base_idx = idx // 8
                    transform_idx = idx % 8
                else:
                    base_idx = idx
                    transform_idx = random.randint(0, 7) if self.augment else 0

                # Apply D4 transform
                if transform_idx > 0:
                    # Apply transform to inputs/outputs
                    # This is a simplified version - you'd implement the actual D4 transforms
                    input_one_hot = self.inputs[base_idx]
                    output_one_hot = self.outputs[base_idx]
                    input_features = self.input_features[base_idx]
                    output_features = self.output_features[base_idx]
                else:
                    input_one_hot = self.inputs[base_idx]
                    output_one_hot = self.outputs[base_idx]
                    input_features = self.input_features[base_idx]
                    output_features = self.output_features[base_idx]

                return (
                    input_one_hot,
                    output_one_hot,
                    input_features,
                    output_features,
                    self.indices[base_idx],
                    torch.tensor(transform_idx, dtype=torch.long),
                    self.file_ids[base_idx],
                )

        dataset = MultiFileD4Dataset(
            combined_inputs,
            combined_outputs,
            combined_input_features,
            combined_output_features,
            combined_indices,
            combined_file_ids,
            augment=True,
            deterministic=training_config.d4_deterministic,
        )

        if training_config.d4_deterministic:
            batch_size = batch_size * 8  # Include all transforms in batch

    else:
        # No augmentation for validation/test
        dataset = torch.utils.data.TensorDataset(
            combined_inputs,
            combined_outputs,
            combined_input_features,
            combined_output_features,
            combined_indices,
            torch.zeros(len(combined_inputs), dtype=torch.long),  # No transform
            combined_file_ids,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


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

    training_index_metrics: Dict[tuple[str, int], Dict[str, int]] = {}
    validation_index_metrics: Dict[tuple[str, int], Dict[str, int]] = {}
    force_visualize: bool = False
    last_max_loss: Optional[float] = None
    last_logits_abs_mean: Optional[float] = None
    frame_capture: FrameCapture = FrameCapture()
    file_id_to_name: Dict[int, str] = {}  # Reverse mapping for file IDs

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
        context_dim: int = 32,
        lora_rank: int = 8,
        enable_file_context: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim = feature_dim + dim_d4_embed
        self.num_message_rounds = num_message_rounds
        self.num_rounds_stop_prob = num_rounds_stop_prob
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.enable_file_context = enable_file_context
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

        # File context embedding
        if enable_file_context:
            self.file_context_embedding = FileContextEmbedding(context_dim)
        else:
            self.file_context_embedding = None

        # Replace NCA and message passing with local attention stack
        # Choose num_heads that divides hidden_dim evenly
        # Prefer head_dim to be 8 or 16 for better performance
        if hidden_dim % 8 == 0:
            # Try to get head_dim = 8
            num_heads = hidden_dim // 8
        elif hidden_dim % 16 == 0:
            # Try to get head_dim = 16
            num_heads = hidden_dim // 16
        elif hidden_dim % 4 == 0:
            # Fallback to head_dim = 4
            num_heads = hidden_dim // 4
        else:
            # Last resort: use as many heads as possible
            for h in range(min(16, hidden_dim), 0, -1):
                if hidden_dim % h == 0:
                    num_heads = h
                    break
            else:
                num_heads = 1  # Fallback to single head
        self.attention_stack = LocalAttentionStack(
            hidden_dim=hidden_dim,
            num_layers=4,  # 4 layers of local attention
            num_heads=num_heads,  # Adaptive number of heads
            kernel_sizes=[3, 3, 5, 5],  # Varying kernel sizes
            dilations=[1, 2, 2, 4],  # Increasing dilation for larger receptive field
            dropout=dropout,
            context_dim=hidden_dim,
            file_context_dim=context_dim if enable_file_context else None,
            lora_rank=lora_rank,
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
                eps=1e-12,
            )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        colors: ColorGrid,
        d4_transform_type: D4TransformType,
        file_ids: Optional[torch.Tensor] = None,
        step_type: Literal["train", "val"] = "train",
    ) -> OneHot11:
        """Return combined logits with channel 0 = mask (-1), channels 1..10 = colors 0..9.
        Shape: [B, 30, 30, 11]
        """
        with torch.no_grad():
            order2_raw = self.order2_encoder(colors)
        color_embed = self.color_embed(colors + 1)  # shape [B, 30, 30, 64]
        o1 = self.feature_processor(order2_raw) + color_embed  # shape [B, 30, 30, 64]
        d4_embed = self.d4_embed(d4_transform_type)  # shape [B, 8]
        # Broadcast d4_embed to match spatial dimensions [B, 8] -> [B, 30, 30, 8]
        d4_embed = d4_embed.unsqueeze(1).unsqueeze(2).expand(-1, 30, 30, -1)
        o = torch.cat([o1, d4_embed], dim=-1)  # [B, 30, 30, 64 + 8]
        h = o

        # Get file context embeddings if available
        file_context = None
        if (
            self.enable_file_context
            and file_ids is not None
            and self.file_context_embedding is not None
        ):
            file_context = self.file_context_embedding(file_ids)  # [B, context_dim]

        self.frame_capture.capture(
            self.linear_0(h), round_idx=-1
        )  # Capture initial state
        # Iterative refinement through local attention
        for t in range(self.num_message_rounds):
            apply_noise = t < max(0, self.num_message_rounds - self.num_final_steps)

            # Early stopping during training
            if (
                self.training
                and (t > self.num_message_rounds - self.num_final_steps)
                and (random.random() < self.num_rounds_stop_prob)
            ):
                break

            if apply_noise:
                h = self.drop1(h)

            # Store previous state for residual
            h_prev = h

            # Apply one step of attention stack
            h = self.attention_stack(
                h,
                context=o,  # Pass original features as context
                file_context=file_context,  # Pass file context
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

        self.frame_capture.capture(
            combined_logits, round_idx=self.num_message_rounds
        )  # Capture final state
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
        self.frame_capture.to_gif(
            f"gifs/message_rounds_{self.current_epoch}.gif", duration_ms=100
        )
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

        # Get file IDs if available
        file_ids = getattr(i.inp, "file_ids", None)

        # Create file ID to name mapping if not already done
        if (
            self.enable_file_context
            and file_ids is not None 
            and self.file_context_embedding is not None
            and not self.file_id_to_name
        ):
            # Build reverse mapping from file_context_embedding
            for filename, file_id in self.file_context_embedding.filename_to_id.items():
                self.file_id_to_name[file_id] = filename

        # Debug logging for multi-file training
        if (
            self.enable_file_context
            and file_ids is not None
            and batch_idx == 0
            and self.current_epoch == 0
        ):
            print(f"\n[DEBUG] First batch of first epoch:")
            print(f"  File IDs in batch: {file_ids.unique().tolist()}")
            print(f"  Batch size: {len(file_ids)}")
            print(
                f"  File ID distribution: {[(fid.item(), (file_ids == fid).sum().item()) for fid in file_ids.unique()]}"
            )

            # Log gate values after forward pass
            if self.enable_file_context and hasattr(
                self.attention_stack.layers[0].attention, "qkv_modulation"
            ):
                with torch.no_grad():
                    context = self.file_context_embedding(file_ids)
                    alpha = self.attention_stack.layers[
                        0
                    ].attention.qkv_modulation.gate(context)
                    print(
                        f"  Alpha (gate) values: min={alpha.min().item():.4f}, max={alpha.max().item():.4f}, mean={alpha.mean().item():.4f}"
                    )
                    print(
                        f"  Context embeddings norm: {context.norm(dim=-1).mean().item():.4f}"
                    )

        # Make predictions
        transform_idx = (
            i.inp.transform_idx
            if i.inp.transform_idx is not None
            else torch.zeros_like(i.inp.idx)
        )
        logits = self.forward(
            i.inp.col, transform_idx, file_ids=file_ids, step_type=step_type
        )

        # Compute losses for color predictions (unchanged)
        metrics: Dict[int, Dict[str, float]] = {}
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
            print(
                f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}"
            )

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
        
        # Convert metrics to use (filename, index) tuples as keys
        int_metrics: Dict[tuple[str, int], Dict[str, int]] = {}
        for idx, m in metrics.items():
            # Determine filename for this index
            if file_ids is not None and self.file_id_to_name:
                # Find the file ID for this example index in the batch
                batch_indices = i.out.idx
                mask = batch_indices == idx
                if mask.any():
                    # Get the file ID for this example
                    example_file_id = file_ids[mask][0].item()
                    filename = self.file_id_to_name.get(example_file_id, f"file_{example_file_id}")
                else:
                    filename = "unknown"
            else:
                # Single file mode - use the configured filename
                filename = getattr(self, '_training_filename', 'single_file')
            
            key = (filename, idx)
            int_metrics[key] = {k: int(v) for k, v in m.items()}

        if step_type == "train":
            self.training_index_metrics.update(int_metrics)
        else:
            self.validation_index_metrics.update(int_metrics)

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
        self, metrics: Dict[tuple[str, int], Dict[str, int]], prefix: str = "train"
    ) -> None:
        for (filename, idx) in metrics.keys():
            for key, value in metrics[(filename, idx)].items():
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
            prefix = "train"
        else:
            metrics = self.validation_index_metrics
            prefix = "val"

        if metrics:
            print(
                f"\n{step_type.upper()} Per-Index Accuracy (Epoch {self.current_epoch}):"
            )
            print(
                f"{'Filename':<20} {'Index':<8} {'Output Pixels':<12} {'Output Mask':<12} {'All Perfect':<12}"
            )
            print("-" * 65)

            # Sort by filename first, then by index
            sorted_keys = sorted(metrics.keys(), key=lambda x: (x[0], x[1]))
            
            for key in sorted_keys:
                filename, idx = key
                m = metrics[key]
                out_pix_incor = m[f"{prefix}_n_incorrect_num_color"]
                out_msk_incor = m[f"{prefix}_n_incorrect_num_mask"]

                all_perfect = "✓" if out_pix_incor == 0 else "✗"
                # Truncate filename if too long
                display_filename = filename if len(filename) <= 18 else filename[:15] + "..."
                print(
                    f"{display_filename:<20} {idx:<8} {out_pix_incor:<12} {out_msk_incor:<12} {all_perfect:<12}"
                )

        # Clear metrics for next epoch
        if step_type == "train":
            self.training_index_metrics = {}
        else:
            self.validation_index_metrics = {}

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

    # Check experiment type
    if training_config.experiment_type == "multi_file":
        print("\n=== Multi-File Training with Context-Aware Attention ===")
        print(f"Training on files: {', '.join(training_config.filenames)}")
        print("Architecture: color -> order2 -> context-modulated attention -> color")

        # For debugging, use tiny model sizes
        # if len(training_config.filenames) == 2:  # Debug mode
        #     print("\n*** DEBUG MODE: Using tiny model sizes ***")
        #     training_config.feature_dim = 16
        #     training_config.num_message_rounds = 2
        #     training_config.max_epochs = 1
        #     training_config.context_dim = 8
        #     training_config.lora_rank = 4
    else:
        # Single file training
        filename = training_config.filename
        print("\n=== Order2-based Model with Local Attention ===")
        print(f"Training on 'train' subset of {filename}")
        print(f"Evaluating on 'test' subset of {filename}")
        print("Architecture: color -> order2 -> local attention stack -> color")
    print(f"Using convolution-based local attention")
    print(
        f"D4 Augmentation: {'Enabled' if training_config.use_d4_augmentation else 'Disabled'}"
    )
    if training_config.use_d4_augmentation:
        print(
            f"  Mode: {'Deterministic (8x data)' if training_config.d4_deterministic else 'Random'}"
        )
    print(
        f"Adaptive Gradient Scaler: {'Enabled' if training_config.enable_adaptive_grad_scaler else 'Disabled'}"
    )
    if training_config.enable_adaptive_grad_scaler:
        print(
            f"  Beta: {training_config.grad_scaler_beta}, Warmup: {training_config.grad_scaler_warmup_steps}, Mult: {training_config.grad_scaler_mult}, Z-thresh: {training_config.grad_scaler_z_thresh}"
        )

    rich.print(training_config)

    # Create model first to get file context embedding
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
        context_dim=training_config.context_dim,
        lora_rank=training_config.lora_rank,
        enable_file_context=(training_config.experiment_type == "multi_file"),
    )

    # Store training filename for single-file mode
    if training_config.experiment_type != "multi_file":
        setattr(model, '_training_filename', training_config.filename)

    if training_config.experiment_type == "multi_file":
        # Multi-file training
        print("\nLoading data from multiple files...")

        # Create train loader
        assert (
            model.file_context_embedding is not None
        ), "File context embedding should be initialized for multi-file mode"
        train_loader = create_multi_file_dataloader(
            filenames=training_config.filenames,
            file_context_embedding=model.file_context_embedding,
            training_config=training_config,
            dataset_name="train",
            batch_size=2,  # Very small batch size to avoid OOM with D4 augmentation
            shuffle=True,
        )

        # Create validation loader
        val_loader = create_multi_file_dataloader(
            filenames=training_config.filenames,
            file_context_embedding=model.file_context_embedding,
            training_config=training_config,
            dataset_name="test",
            batch_size=1,  # Process one example at a time for validation
            shuffle=False,
        )

        test_loader = val_loader  # Same as validation for multi-file

        # Get all colors for constraint setting
        all_colors = []
        for filename in training_config.filenames:
            (
                inputs,
                outputs,
                _,
                _,
                _,
            ) = prepare_dataset(
                "processed_data/train_all.npz",
                filter_filename=filename,
                use_features=False,
                dataset_name="train",
                use_train_subset=True,
            )
            input_colors = inputs.argmax(dim=-1)
            output_colors = outputs.argmax(dim=-1)
            input_colors = torch.where(input_colors == 10, -1, input_colors)
            output_colors = torch.where(output_colors == 10, -1, output_colors)
            all_colors.extend([input_colors.flatten(), output_colors.flatten()])

        all_colors = torch.cat(all_colors)

    else:
        # Single file training (original code)
        filename = training_config.filename

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
        train_input_colors = torch.where(
            train_input_colors == 10, -1, train_input_colors
        )
        train_output_colors = torch.where(
            train_output_colors == 10, -1, train_output_colors
        )

        all_colors = torch.cat(
            [train_input_colors.flatten(), train_output_colors.flatten()]
        )

        # Create dataloaders - use D4 augmentation for training if enabled
        if training_config.use_d4_augmentation:
            print(
                f"\nUsing D4 augmentation for training (deterministic={training_config.d4_deterministic})"
            )
            train_loader = create_d4_augmented_dataloader(
                train_inputs,
                train_outputs,
                inputs_features=train_input_features,
                outputs_features=train_output_features,
                indices=train_indices,
                batch_size=len(train_inputs)
                * (
                    8 if training_config.d4_deterministic else 1
                ),  # All transforms in one batch if deterministic
                shuffle=True,
                num_workers=0,  # Eliminate multiprocessing overhead for small datasets
                augment=True,
                deterministic_augmentation=training_config.d4_deterministic,
                repeat_factor=1,
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

    # Compute available colors from training data and set constraints
    all_colors_non_mask: torch.Tensor = all_colors[all_colors != -1]
    available_colors: torch.Tensor = torch.unique(all_colors_non_mask)  # type: ignore[call-arg]
    if training_config.experiment_type == "multi_file":
        print(f"Available colors across all files: {available_colors}")
    else:
        print(f"Available colors for {training_config.filename}: {available_colors}")
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
        if training_config.experiment_type == "multi_file":
            run_name = f"ex37-multifile-{config_hash}"
        else:
            run_name = f"ex36-{training_config.filename}-{config_hash}"

        # Initialize wandb logger with all config parameters
        wandb_logger = WandbLogger(
            project="arcagi",
            name=run_name,
            config=config_dict,  # Pass all training config parameters as hyperparameters
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
    if training_config.experiment_type == "multi_file":
        final_model_path = os.path.join(
            training_config.checkpoint_dir, f"order2_model_multifile.pt"
        )
    else:
        final_model_path = os.path.join(
            training_config.checkpoint_dir,
            f"order2_model_{training_config.filename}.pt",
        )

    # Save both model state dict and metadata
    # Save both model state dict and minimal metadata (avoid typing complaints)
    try:
        hyperparams = dict(model.hparams)  # type: ignore[arg-type]
    except Exception:
        hyperparams = {}
    model_save_data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "filename": (
            training_config.filename
            if training_config.experiment_type != "multi_file"
            else "multi_file"
        ),
        "hyperparameters": hyperparams,
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
