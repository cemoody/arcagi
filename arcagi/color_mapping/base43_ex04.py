import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Any, Optional

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.order2 import Order2Features
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx04(TrainingConfig):
    project_name: str = "arcagi-base43-ex04"
    filename_filter: Optional[str] = "007bbfb7"

    max_epochs: int = 1000

    num_message_rounds: int = 64  # Fewer rounds needed with attention
    n_channels: int = 256

    # Model parameters
    embed_dim: int = 128
    dropout: float = 0.1

    # Attention parameters
    num_heads: int = 8
    attention_dropout: float = 0.1

    # RoPE parameters
    rope_base: float = 10000.0


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding for 2D grids."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        x: [B, H, W, C] or [B, H*W, C]
        """
        if x.dim() == 4:
            B, H, W, C = x.shape
            x_flat = x.view(B, H * W, C)
        else:
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x_flat = x

        # Create position indices
        pos_h = torch.arange(H, device=x.device, dtype=x.dtype)
        pos_w = torch.arange(W, device=x.device, dtype=x.dtype)

        # Create 2D positional encodings
        dim_t = torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype)
        freqs = 1.0 / (self.base ** (dim_t / self.dim))

        # Apply to height and width separately
        pos_h = pos_h.unsqueeze(1) * freqs.unsqueeze(0)  # [H, dim/2]
        pos_w = pos_w.unsqueeze(1) * freqs.unsqueeze(0)  # [W, dim/2]

        # Create sin/cos embeddings
        pos_h_sin = torch.sin(pos_h).unsqueeze(1).expand(-1, W, -1)  # [H, W, dim/2]
        pos_h_cos = torch.cos(pos_h).unsqueeze(1).expand(-1, W, -1)  # [H, W, dim/2]
        pos_w_sin = torch.sin(pos_w).unsqueeze(0).expand(H, -1, -1)  # [H, W, dim/2]
        pos_w_cos = torch.cos(pos_w).unsqueeze(0).expand(H, -1, -1)  # [H, W, dim/2]

        # Flatten spatial dimensions
        pos_h_sin = pos_h_sin.reshape(H * W, -1)  # [H*W, dim/2]
        pos_h_cos = pos_h_cos.reshape(H * W, -1)  # [H*W, dim/2]
        pos_w_sin = pos_w_sin.reshape(H * W, -1)  # [H*W, dim/2]
        pos_w_cos = pos_w_cos.reshape(H * W, -1)  # [H*W, dim/2]

        # Apply rotary embeddings
        x_even = x_flat[..., 0::2]  # [B, H*W, dim/2]
        x_odd = x_flat[..., 1::2]  # [B, H*W, dim/2]

        # Height rotation
        x_even_rot = x_even * pos_h_cos - x_odd * pos_h_sin
        x_odd_rot = x_even * pos_h_sin + x_odd * pos_h_cos

        # Width rotation (applied to half of the dimensions)
        half_dim = x_even.shape[-1] // 2
        x_even_rot[..., :half_dim] = (
            x_even[..., :half_dim] * pos_w_cos[..., :half_dim]
            - x_odd[..., :half_dim] * pos_w_sin[..., :half_dim]
        )
        x_odd_rot[..., :half_dim] = (
            x_even[..., :half_dim] * pos_w_sin[..., :half_dim]
            + x_odd[..., :half_dim] * pos_w_cos[..., :half_dim]
        )

        # Interleave back
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)

        if x.dim() == 4:
            x_rot = x_rot.view(B, H, W, C)

        return x_rot


class AttentionNCA(nn.Module):
    """NCA with self-attention for global context awareness."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_heads: int = 8,
        hidden: int = 64,
        kernel_size: int = 3,
        fire_rate: float = 1.0,
        dilation_rates: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Layer norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        # norm3 needs to handle concatenated features: channels + perceptions + channels
        self.norm3 = nn.LayerNorm(
            channels + len(dilation_rates) * out_channels + channels
        )

        # Local perception (NCA-style)
        self.perception_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=dilation * (kernel_size // 2),
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in dilation_rates
            ]
        )

        # Global attention
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)

        # RoPE
        self.rope = RotaryPositionalEmbedding(channels, base=rope_base)

        # Update network (combines local and global features)
        self.update = nn.Sequential(
            nn.Linear(
                channels + len(dilation_rates) * out_channels + channels, hidden * 2
            ),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        self.fire_rate = fire_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        residual = x

        # Local perception
        x_norm = self.norm1(x)
        x_perm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]

        perceptions = []
        for perception_layer in self.perception_layers:
            p = perception_layer(x_perm).permute(0, 2, 3, 1)  # [B, H, W, out_channels]
            perceptions.append(p)

        # Global attention
        x_attn = self.norm2(x)
        x_flat = x_attn.view(B, H * W, C)

        # Apply RoPE
        x_flat = self.rope(x_flat)

        # Multi-head attention
        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_global = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x_global = self.proj(x_global)
        x_global = self.proj_drop(x_global)
        x_global = x_global.view(B, H, W, C)

        # Combine local, global, and original features
        update_in = torch.cat([x_norm] + perceptions + [x_global], dim=-1)
        update_in = self.norm3(update_in)

        # Compute update
        delta = self.update(update_in)

        # Stochastic masking
        if self.training and self.fire_rate < 1.0:
            mask = (torch.rand(B, H, W, 1, device=x.device) < self.fire_rate).float()
            delta = delta * mask

        return residual + delta


class Ex04(nn.Module):
    def __init__(self, config: TrainingConfigEx04):
        super().__init__()
        self.num_message_rounds = config.num_message_rounds
        self.config = config

        # Input embeddings
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Attention-based NCA
        self.model = AttentionNCA(
            channels=config.embed_dim,
            out_channels=config.n_channels,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            rope_base=config.rope_base,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, 11)

        # Frame capture
        self.frame_capture = FrameCapture()

    def forward(
        self,
        input_images: torch.Tensor,
        filenames: torch.Tensor,
        example_indices: torch.Tensor,
        context: torch.Tensor,
        current_epoch: int | None = None,
    ) -> torch.Tensor:
        color_indices = input_images.argmax(dim=-1)  # [B, 30, 30]
        x1 = self.order2(color_indices)  # [B, 30, 30, 45]
        eo = self.input_embed(x1)  # [B, 30, 30, embed_dim]
        ec = self.color_embed(color_indices)  # [B, 30, 30, embed_dim]
        x2 = eo + ec  # [B, 30, 30, embed_dim]

        if current_epoch is not None and current_epoch % 100 == 0:
            self.frame_capture.enable()

        for i in range(self.num_message_rounds):
            x2 = self.model(x2)
            if current_epoch is not None and current_epoch % 10 == 0:
                x3 = self.project_to_color(x2, round_idx=i)

        x3 = self.project_to_color(x2)

        if current_epoch is not None and current_epoch % 100 == 0:
            self.frame_capture.disable()
            self.frame_capture.to_gif(
                f"gifs/message_rounds_{current_epoch}_ex04.gif", duration_ms=100
            )

        return x3

    def project_to_color(
        self, x2: torch.Tensor, round_idx: int | None = None
    ) -> torch.Tensor:
        x2_norm = self.output_norm(x2)
        x2_out = self.output_proj(x2_norm)  # [B, 30, 30, 11]
        if round_idx is not None:
            self.frame_capture.capture(x2_out, round_idx=round_idx)
        return x2_out


class MainModuleEx04(MainModule):
    def __init__(self, config: TrainingConfigEx04):
        super().__init__(config)
        self.config = config
        self.model = Ex04(config)
        self.save_hyperparameters(config.model_dump())


@click.command()
@from_pydantic(
    TrainingConfigEx04,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfigEx04.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(**kwargs: Any) -> None:
    config_key = next(k for k in kwargs if k.startswith("training_config"))
    training_config: TrainingConfigEx04 = kwargs[config_key]
    return main(training_config, MainModuleEx04)


if __name__ == "__main__":
    main_click()
