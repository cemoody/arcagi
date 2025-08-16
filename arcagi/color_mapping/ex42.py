import hashlib
import json
import os
import random
import numpy as np

# Import from parent module
import sys
from collections import defaultdict
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
    import wandb
except Exception:
    WandbLogger = None  # type: ignore
    wandb = None  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
# Import data loading functions
from data_loader import prepare_dataset
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
    weight_decay: float = 1e-7
    feature_dim: int = 48
    num_message_rounds: int = 24
    num_final_steps: int = 6
    num_rounds_stop_prob: float = 0.10

    # Self-healing noise parameters
    enable_self_healing: bool = True
    death_prob: float = 0.30
    gaussian_std: float = 0.30
    spatial_corruption_prob: float = 0.30
    dropout: float = 0.30
    dropout_h: float = 0.30

    # Multi-file training parameters
    filenames: List[str] = [ "dc433765.json"]

    # Context-aware attention parameters
    context_dim: int = 64
    lora_rank: int = 64
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    checkpoint_segments: int = 4  # Number of segments to split message rounds into
    gradient_accumulation_steps: int = 1  # Accumulate gradients over multiple steps
    mixed_precision: bool = True  # Use mixed precision training
    use_flash_attention: bool = True  # Use Flash Attention for memory efficiency

    # D4 augmentation parameters
    use_d4_augmentation: bool = True
    d4_deterministic: bool = True  # If True, cycles through all 8 transformations

    # Adaptive gradient scaler parameters
    enable_adaptive_grad_scaler: bool = True
    grad_scaler_beta: float = 0.98
    grad_scaler_warmup_steps: int = 500
    grad_scaler_mult: float = 4.0
    grad_scaler_z_thresh: float = 4.0
    
    # Batch-wide attention parameters
    enable_batch_attention: bool = True  # Enable cross-attention between examples with same file_id


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


def create_few_shot_collate_fn(max_examples_per_file: int = 6):
    """Create a collate function that groups examples from the same file."""
    def collate_fn(batch: List[tuple]) -> List[torch.Tensor]:
        # Group examples by file ID
        file_groups: Dict[int, List[tuple]] = {}
        for example in batch:
            file_id = example[6].item() if len(example) > 6 else 0  # file_id is the 7th element
            if file_id not in file_groups:
                file_groups[file_id] = []
            file_groups[file_id].append(example)
        
        # For each file, take up to max_examples_per_file examples
        selected_examples = []
        for file_id, examples in file_groups.items():
            # Randomly sample if we have more examples than needed
            if len(examples) > max_examples_per_file:
                examples = random.sample(examples, max_examples_per_file)
            selected_examples.extend(examples)
        
        # Now collate the selected examples
        if not selected_examples:
            return []
        
        # Transpose the list of tuples to tuple of lists
        transposed = list(zip(*selected_examples))
        
        # Stack each component
        result = []
        for i, component in enumerate(transposed):
            if isinstance(component[0], torch.Tensor):
                result.append(torch.stack(component))
            else:
                result.append(list(component))
        
        return result
    
    return collate_fn


def create_multi_file_dataloader(
    filenames: List[str],
    file_context_embedding: Any,  # Type compatibility with ex41
    training_config: TrainingConfig,
    dataset_name: str = "train",
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader[Any]:
    """Create a dataloader that loads examples from multiple files with file ID tracking."""

    all_inputs = []
    all_outputs = []
    all_input_features = []
    all_output_features = []
    all_indices = []
    all_file_ids = []

    # Load data from each file
    for file_idx, filename in enumerate(filenames):
        # Use file index as ID for simplicity
        file_id = file_idx

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

    # No D4 augmentation for now - keep it simple
    dataset = torch.utils.data.TensorDataset(
        combined_inputs,
        combined_outputs,
        combined_input_features,
        combined_output_features,
        combined_indices,
        torch.zeros(len(combined_inputs), dtype=torch.long),  # No transform
        combined_file_ids,
    )

    # Use custom collate function to group examples from same file
    collate_fn = create_few_shot_collate_fn(max_examples_per_file=batch_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size * len(filenames),  # Load multiple files per batch
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
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


class GlobalExampleAttention(nn.Module):
    """Global attention across all examples in the same task.
    
    Uses Flash Attention when available for improved memory efficiency.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N is total number of tokens (all examples * 2 * 30 * 30)
            attention_mask: [B, N, N] attention mask
        Returns:
            [B, N, D] attended features
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        
        # Use Flash Attention when no custom mask is needed
        # Note: scaled_dot_product_attention handles the scaling internally
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )  # [B, H, N, D/H]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class FewShotTransformer(nn.Module):
    """Transformer that processes all examples from a task together with tied weights."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create shared layer components (tied weights)
        self.shared_attention = GlobalExampleAttention(hidden_dim, num_heads, dropout, use_flash_attention)
        self.shared_norm1 = ArsinhNorm(hidden_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )
        self.shared_norm2 = ArsinhNorm(hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] features
            attention_mask: [B, N, N] attention mask
        Returns:
            [B, N, D] processed features
        """
        # Apply the same layer multiple times (tied weights)
        for _ in range(self.num_layers):
            # Self-attention with residual
            x_norm = self.shared_norm1(x)
            x = x + self.shared_attention(x_norm, attention_mask)
            
            # MLP with residual
            x_norm = self.shared_norm2(x)
            x = x + self.shared_mlp(x_norm)
            
        return x


class MainModel(pl.LightningModule):
    """Few-shot learning model that attends to all input-output pairs in a task."""

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
        num_transformer_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        dropout_h: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dim_feat: int = 45,
        enable_adaptive_grad_scaler: bool = True,
        grad_scaler_beta: float = 0.98,
        grad_scaler_warmup_steps: int = 500,
        grad_scaler_mult: float = 4.0,
        grad_scaler_z_thresh: float = 4.0,
        dim_d4_embed: int = 8,
        use_gradient_checkpointing: bool = True,
        checkpoint_segments: int = 4,
        mlp_ratio: float = 4.0,
        use_positional_encoding: bool = True,
        # Unused parameters kept for compatibility
        num_message_rounds: int = 6,
        enable_self_healing: bool = True,
        death_prob: float = 0.02,
        gaussian_std: float = 0.05,
        spatial_corruption_prob: float = 0.01,
        num_final_steps: int = 12,
        num_rounds_stop_prob: float = 0.50,
        context_dim: int = 32,
        lora_rank: int = 8,
        enable_file_context: bool = True,
        enable_batch_attention: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim = feature_dim + dim_d4_embed
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_segments = checkpoint_segments
        self.use_positional_encoding = use_positional_encoding
        
        # Order2 feature extractor
        self.order2_encoder = Order2Features()

        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(dim_feat, feature_dim),
            RMSNorm(feature_dim),
            nn.GELU(),
        )
        
        # Embeddings
        self.d4_embed = nn.Embedding(8, dim_d4_embed)
        self.color_embed = nn.Embedding(11, feature_dim)
        
        # Type embedding to distinguish input vs output
        self.type_embed = nn.Embedding(2, hidden_dim)  # 0 for input, 1 for output
        
        # Positional encoding for spatial positions
        if self.use_positional_encoding:
            self.pos_embed = nn.Parameter(torch.randn(1, 30, 30, hidden_dim) * 0.02)
        
        # Main transformer
        self.transformer = FewShotTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            use_flash_attention=use_flash_attention,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 11)
        
        self.drop_h = nn.Dropout(dropout_h)
        self.available_colors: Optional[List[int]] = None

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

    def create_attention_mask(
        self,
        batch_size: int,
        n_examples: int,
        indices: torch.Tensor,
        file_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create attention mask for few-shot learning.
        
        Each example can attend to:
        - All input grids from the same file
        - All output grids from the same file EXCEPT its own output
        - No grids from different files
        
        Args:
            batch_size: Number of tasks in batch
            n_examples: Number of examples per task
            indices: [batch_size] example indices
            file_ids: [batch_size] file IDs
            device: Device to create mask on
            
        Returns:
            attention_mask: [1, total_tokens, total_tokens] where total_tokens = batch_size * n_examples * 2 * 30 * 30
        """
        # Total number of grid cells per example (input + output)
        cells_per_grid = 30 * 30
        grids_per_example = 2  # input and output
        tokens_per_example = grids_per_example * cells_per_grid
        total_tokens = batch_size * tokens_per_example
        
        # Create base mask - all True initially
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)
        
        # For each example in the batch
        for i in range(batch_size):
            # Token range for this example
            start_idx = i * tokens_per_example
            input_start = start_idx
            input_end = start_idx + cells_per_grid
            output_start = start_idx + cells_per_grid
            output_end = start_idx + tokens_per_example
            
            # This example can only attend to examples with the same file_id
            for j in range(batch_size):
                other_start = j * tokens_per_example
                other_input_start = other_start
                other_input_end = other_start + cells_per_grid
                other_output_start = other_start + cells_per_grid
                other_output_end = other_start + tokens_per_example
                
                if file_ids[i] != file_ids[j]:
                    # Different file - no attention
                    mask[start_idx:output_end, other_start:other_output_end] = False
                elif i == j:
                    # Same example - can attend to own input but not own output
                    mask[start_idx:output_end, other_output_start:other_output_end] = False
                # else: same file, different example - can attend to everything
        
        # Add batch dimension
        mask = mask.unsqueeze(0)  # [1, total_tokens, total_tokens]
        
        return mask

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        all_input_colors: torch.Tensor,  # [batch_size, 30, 30]
        all_output_colors: torch.Tensor,  # [batch_size, 30, 30]
        indices: torch.Tensor,  # [batch_size]
        file_ids: torch.Tensor,  # [batch_size]
        d4_transform_type: torch.Tensor,  # [batch_size]
        step_type: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        """Process all examples together with attention masking.
        
        Returns:
            logits: [batch_size, 30, 30, 11] predictions for each example
        """
        batch_size = all_input_colors.shape[0]
        device = all_input_colors.device
        
        # Process inputs through Order2 encoder
        with torch.no_grad():
            input_order2 = self.order2_encoder(all_input_colors)  # [B, 30, 30, 45]
            output_order2 = self.order2_encoder(all_output_colors)  # [B, 30, 30, 45]
        
        # Process features
        input_color_embed = self.color_embed(all_input_colors + 1)  # [B, 30, 30, feature_dim]
        output_color_embed = self.color_embed(all_output_colors + 1)  # [B, 30, 30, feature_dim]
        
        input_features = self.feature_processor(input_order2) + input_color_embed  # [B, 30, 30, feature_dim]
        output_features = self.feature_processor(output_order2) + output_color_embed  # [B, 30, 30, feature_dim]
        
        # Add D4 embeddings
        d4_embed = self.d4_embed(d4_transform_type)  # [B, dim_d4_embed]
        d4_embed = d4_embed.unsqueeze(1).unsqueeze(2).expand(-1, 30, 30, -1)  # [B, 30, 30, dim_d4_embed]
        
        input_features = torch.cat([input_features, d4_embed], dim=-1)  # [B, 30, 30, hidden_dim]
        output_features = torch.cat([output_features, d4_embed], dim=-1)  # [B, 30, 30, hidden_dim]
        
        # Add type embeddings
        input_type_embed = self.type_embed(torch.zeros(batch_size, device=device, dtype=torch.long))  # [B, hidden_dim]
        output_type_embed = self.type_embed(torch.ones(batch_size, device=device, dtype=torch.long))  # [B, hidden_dim]
        
        input_type_embed = input_type_embed.unsqueeze(1).unsqueeze(2).expand(-1, 30, 30, -1)  # [B, 30, 30, hidden_dim]
        output_type_embed = output_type_embed.unsqueeze(1).unsqueeze(2).expand(-1, 30, 30, -1)  # [B, 30, 30, hidden_dim]
        
        input_features = input_features + input_type_embed
        output_features = output_features + output_type_embed
        
        # Add positional embeddings
        if self.use_positional_encoding:
            input_features = input_features + self.pos_embed
            output_features = output_features + self.pos_embed
        
        # Reshape to combine all examples
        # From [B, 30, 30, D] to [B, 900, D]
        input_features_flat = input_features.reshape(batch_size, 900, self.hidden_dim)
        output_features_flat = output_features.reshape(batch_size, 900, self.hidden_dim)

        input_features_ex_idx= torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 900)
        output_features_ex_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 900)

        # Do not attend to the same example
        mask_ex_idx = input_features_ex_idx[:, :, None] != output_features_ex_idx[:, None, :]
        
        # Project to logits
        logits = self.output_proj(output_features_transformed)  # [B, 30, 30, 11]
        
        # Capture final state
        self.frame_capture.capture(logits, round_idx=0)
        
        # Logits explosion detection
        current_logits_abs_mean = float(logits.abs().mean().item())
        try:
            self.log(f"{step_type}_logits_abs_mean", current_logits_abs_mean)  # type: ignore
        except Exception:
            pass
        self.last_logits_abs_mean = current_logits_abs_mean
        
        return logits




    @jaxtyped(typechecker=beartype)
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="train")

    @jaxtyped(typechecker=beartype)
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Only enable frame capture every 10th step
        if batch_idx % 10 == 0:
            self.frame_capture.clear()
            self.frame_capture.enable()
        
        out = self.step(batch, batch_idx, step_type="val")
        
        # Only save GIF every 10th step
        if batch_idx % 10 == 0:
            self.frame_capture.disable()
            
            # Get batch information for filename
            i = batch_to_dataclass(batch)
            file_ids = getattr(i.inp, "file_ids", None)
            
            # Determine filename and index for the first item in batch
            filename = "unknown"
            example_idx = 0
            
            if i.out.idx is not None and len(i.out.idx) > 0:
                example_idx = i.out.idx[0].item()
            
            if file_ids is not None and self.file_id_to_name and len(file_ids) > 0:
                file_id = file_ids[0].item()
                filename = self.file_id_to_name.get(file_id, f"file_{file_id}")
                # Remove file extension and path for cleaner filename
                filename = filename.split('/')[-1].split('.')[0]
            
            gif_filename = f"gifs/message_rounds_{filename}_{example_idx}_{self.current_epoch}.gif"
            self.frame_capture.to_gif(gif_filename, duration_ms=100)
            logger.info(f"Animation saved to {gif_filename}")
        
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
        if file_ids is None:
            raise ValueError("file_ids must be provided for few-shot learning")

        # Make predictions
        transform_idx = (
            i.inp.transform_idx
            if i.inp.transform_idx is not None
            else torch.zeros_like(i.inp.idx)
        )
        
        logits = self.forward(
            all_input_colors=i.inp.col,
            all_output_colors=i.out.col,
            indices=i.inp.idx,
            file_ids=file_ids,
            d4_transform_type=transform_idx,
            step_type=step_type
        )

        # Compute losses for color predictions (unchanged)
        metrics: Dict[int, Dict[str, float]] = {}
        logits_perm = logits.permute(0, 3, 1, 2)  # [B, 11, 30, 30]
        loss = cross_entropy_shifted(logits_perm, i.out.col.long(), start_index=-1)

        image_metrics(i.out, logits, metrics, prefix=step_type)

        # Visualize every 10 epochs
        # For validation, visualize all batches, not just the first one
        if self.force_visualize or (self.current_epoch % 10 == 0 and (batch_idx == 0 or step_type == "val")):
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
            # Multi-file mode - find the file ID for this example index
            if file_ids is not None and self.file_id_to_name:
                batch_indices = i.out.idx
                mask = batch_indices == idx
                if mask.any():
                    # Get the file ID for this example
                    example_file_id = file_ids[mask][0].item()
                    filename = self.file_id_to_name.get(example_file_id, f"file_{example_file_id}")
                else:
                    filename = "unknown"
            else:
                filename = "unknown"
            
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

        # Log detailed per-example metrics to wandb
        self._log_wandb_validation_metrics()

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

    def _log_wandb_validation_metrics(self) -> None:
        """Log detailed per-example validation metrics to wandb."""
        if wandb is None or not hasattr(wandb, 'run') or not wandb.run:
            return
            
        if not self.validation_index_metrics:
            return

        # 1. Create wandb table for detailed per-example metrics
        table_columns = [
            "epoch", "filename", "index", 
            "incorrect_pixels", "incorrect_mask", 
            "total_pixels", "pixel_accuracy", "all_perfect"
        ]
        wandb_table = wandb.Table(columns=table_columns)
        
        # Sort by filename first, then by index for consistent ordering
        sorted_keys = sorted(self.validation_index_metrics.keys(), key=lambda x: (x[0], x[1]))
        
        for (filename, idx), metrics in self.validation_index_metrics.items():
            incorrect_pixels = metrics["val_n_incorrect_num_color"]
            incorrect_mask = metrics["val_n_incorrect_num_mask"]
            total_pixels = 900  # 30x30 grid
            pixel_accuracy = 1 - (incorrect_pixels / total_pixels)
            all_perfect = incorrect_pixels == 0
            
            wandb_table.add_data(
                self.current_epoch,
                filename,
                idx,
                incorrect_pixels,
                incorrect_mask,
                total_pixels,
                pixel_accuracy,
                all_perfect
            )
            
            # 2. Log time-series metrics for each example
            metric_prefix = f"val_example/{filename}_{idx}"
            self.log(f"{metric_prefix}/incorrect_pixels", incorrect_pixels)
            self.log(f"{metric_prefix}/incorrect_mask", incorrect_mask)
            self.log(f"{metric_prefix}/pixel_accuracy", pixel_accuracy)
            self.log(f"{metric_prefix}/all_perfect", float(all_perfect))
        
        # Log the table
        wandb.log({"val_per_example_metrics": wandb_table})
        
        # 3. Aggregate metrics by filename
        file_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "incorrect_pixels": 0.0, 
            "incorrect_mask": 0.0, 
            "count": 0,
            "perfect_count": 0
        })
        
        for (filename, idx), metrics in self.validation_index_metrics.items():
            file_metrics[filename]["incorrect_pixels"] += metrics["val_n_incorrect_num_color"]
            file_metrics[filename]["incorrect_mask"] += metrics["val_n_incorrect_num_mask"]
            file_metrics[filename]["count"] += 1
            if metrics["val_n_incorrect_num_color"] == 0:
                file_metrics[filename]["perfect_count"] += 1
        
        # Log per-file averages
        for filename, agg_metrics in file_metrics.items():
            count = agg_metrics["count"]
            avg_incorrect = agg_metrics["incorrect_pixels"] / count
            avg_mask_incorrect = agg_metrics["incorrect_mask"] / count
            solve_rate = agg_metrics["perfect_count"] / count
            
            self.log(f"val_file/{filename}/avg_incorrect_pixels", avg_incorrect)
            self.log(f"val_file/{filename}/avg_incorrect_mask", avg_mask_incorrect)
            self.log(f"val_file/{filename}/solve_rate", solve_rate)
        
        # 4. Create heatmap visualization
        files = sorted(set(f for f, _ in self.validation_index_metrics.keys()))
        indices = sorted(set(i for _, i in self.validation_index_metrics.keys()))
        
        if len(files) > 0 and len(indices) > 0:
            heatmap_data = np.zeros((len(files), len(indices)))
            
            for i, file in enumerate(files):
                for j, idx in enumerate(indices):
                    if (file, idx) in self.validation_index_metrics:
                        heatmap_data[i, j] = self.validation_index_metrics[(file, idx)]["val_n_incorrect_num_color"]
                    else:
                        heatmap_data[i, j] = np.nan  # Mark missing data
            
            # Create custom chart for heatmap
            wandb.log({
                "val_performance_heatmap": wandb.plots.HeatMap(
                    x_labels=[str(idx) for idx in indices],
                    y_labels=files,
                    matrix_values=heatmap_data.tolist(),
                    show_text=True
                )
            })
        
        # 5. Track overall progress metrics
        solved_examples = sum(
            1 for metrics in self.validation_index_metrics.values() 
            if metrics["val_n_incorrect_num_color"] == 0
        )
        total_examples = len(self.validation_index_metrics)
        
        self.log("val_examples_solved", solved_examples)
        self.log("val_solve_rate", solved_examples / total_examples if total_examples > 0 else 0)
        
        # Log histogram of incorrect pixels
        incorrect_pixels_list = [
            metrics["val_n_incorrect_num_color"] 
            for metrics in self.validation_index_metrics.values()
        ]
        if incorrect_pixels_list:
            wandb.log({
                "val_incorrect_pixels_histogram": wandb.Histogram(incorrect_pixels_list)
            })

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
        # Get batch size
        batch_size = targets.shape[0]
        
        # Visualize all examples in the batch
        for b in range(batch_size):
            pred_colors = predictions[b].argmax(dim=-1).cpu() + start_index
            true_colors = targets[b].cpu()
            
            # Get example index and filename info
            example_idx = i.out.idx[b].item()
            
            # Try to get filename if available
            filename_info = ""
            if hasattr(i.inp, 'file_ids') and i.inp.file_ids is not None and self.file_id_to_name:
                file_id = i.inp.file_ids[b].item()
                filename = self.file_id_to_name.get(file_id, f"file_{file_id}")
                filename_info = f" - File: {filename}"

            # Create visualization
            print(f"\n{'='*60}")
            print(f"{prefix.upper()} - Epoch {self.current_epoch} - Example {example_idx}{filename_info}")
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
    # Enable anomaly detection for debugging gradient issues
    # torch.autograd.set_detect_anomaly(True)
    # print("⚠️  Anomaly detection enabled - training will be slower but will pinpoint NaN/Inf sources")

    # Multi-file training
    print("\n=== Few-Shot Learning with Global Example Attention ===")
    print(f"Training on files: {', '.join(training_config.filenames)}")
    print("Architecture: Transformer that attends to all input-output pairs in a task")
    print(f"Using global attention across all examples (except target output)")
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

    # Create model
    model = MainModel(
        feature_dim=training_config.feature_dim,
        num_transformer_layers=8,  # Use 8 transformer layers
        num_heads=2,  # 8 attention heads
        dropout=training_config.dropout,
        dropout_h=training_config.dropout_h,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        enable_adaptive_grad_scaler=training_config.enable_adaptive_grad_scaler,
        grad_scaler_beta=training_config.grad_scaler_beta,
        grad_scaler_warmup_steps=training_config.grad_scaler_warmup_steps,
        grad_scaler_mult=training_config.grad_scaler_mult,
        grad_scaler_z_thresh=training_config.grad_scaler_z_thresh,
        use_gradient_checkpointing=training_config.use_gradient_checkpointing,
        checkpoint_segments=training_config.checkpoint_segments,
        mlp_ratio=4.0,
        use_positional_encoding=True,
        # Compatibility parameters (unused)
        num_message_rounds=training_config.num_message_rounds,
        enable_self_healing=training_config.enable_self_healing,
        death_prob=training_config.death_prob,
        gaussian_std=training_config.gaussian_std,
        spatial_corruption_prob=training_config.spatial_corruption_prob,
        num_final_steps=training_config.num_final_steps,
        num_rounds_stop_prob=training_config.num_rounds_stop_prob,
        context_dim=training_config.context_dim,
        lora_rank=training_config.lora_rank,
        enable_file_context=True,
        enable_batch_attention=training_config.enable_batch_attention,
    )



    # Multi-file training
    print("\nLoading data from multiple files...")
    
    # Create train loader - need to load all examples from same task in one batch
    train_loader = create_multi_file_dataloader(
        filenames=training_config.filenames,
        file_context_embedding=None,  # Not needed anymore
        training_config=training_config,
        dataset_name="train",
        batch_size=6,  # Load multiple examples per batch for few-shot learning
        shuffle=True,
    )

    # Create validation loader
    val_loader = create_multi_file_dataloader(
        filenames=training_config.filenames,
        file_context_embedding=None,  # Not needed anymore
        training_config=training_config,
        dataset_name="test",
        batch_size=6,  # Same batch size for validation
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

    # Compute available colors from training data and set constraints
    all_colors_non_mask: torch.Tensor = all_colors[all_colors != -1]
    available_colors: torch.Tensor = torch.unique(all_colors_non_mask)  # type: ignore[call-arg]
    print(f"Available colors across all files: {available_colors}")
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
        run_name = f"ex42-multifile-{config_hash}"

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
            precision="bf16-mixed",  # Use bfloat16 mixed precision
        )
    else:
        trainer = pl.Trainer(
            max_epochs=training_config.max_epochs,
            callbacks=[checkpoint_callback],
            default_root_dir=training_config.checkpoint_dir,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            log_every_n_steps=1,
            precision="bf16-mixed",  # Use bfloat16 mixed precision
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
        training_config.checkpoint_dir, f"order2_model_multifile.pt"
    )

    # Save both model state dict and metadata
    # Save both model state dict and minimal metadata (avoid typing complaints)
    try:
        hyperparams = dict(model.hparams)  # type: ignore[arg-type]
    except Exception:
        hyperparams = {}
    model_save_data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "filename": "multi_file",
        "hyperparameters": hyperparams,
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
