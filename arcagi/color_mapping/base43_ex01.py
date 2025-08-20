import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.arsinhnorm import ArsinhNorm
from arcagi.order2 import Order2Features
from arcagi.positional_embeddings_2d import RoPE2D
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx01(TrainingConfig):
    project_name: str = "arcagi-base43-ex01"
    filename_filter: Optional[str] = "007bbfb7"

    max_epochs: int = 10000

    # Attention module parameters
    embed_dim: int = 64
    num_heads: int = 8
    num_attention_layers: int = 48
    attention_dropout: float = 0.3

    # RoPE parameters
    rope_base: float = 10000.0


class AttentionBlock(nn.Module):
    """Single attention block with ArsinhNorm normalization and optional 2D RoPE."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        height: int,
        width: int,
        rope_base: float = 10000.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for attention
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

        # RoPE if enabled
        self.rope = RoPE2D(self.head_dim, height, width, base=rope_base)

        # Normalization layers
        self.norm1 = ArsinhNorm(embed_dim)
        self.norm2 = ArsinhNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        normed_x = self.norm1(x)
        B, N, C = normed_x.shape

        # Linear projections
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE expects [B, N, num_heads, head_dim] format
        q_for_rope = q.transpose(1, 2).contiguous()
        k_for_rope = k.transpose(1, 2).contiguous()
        q_rotated, k_rotated = self.rope(q_for_rope, k_for_rope)
        q = q_rotated.transpose(1, 2)
        k = k_rotated.transpose(1, 2)

        # Compute attention using scaled dot product (Flash Attention when available)
        # PyTorch's F.scaled_dot_product_attention automatically uses Flash Attention 2
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = x + attn_out

        # FFN with residual
        normed_x = self.norm2(x)
        x = x + self.ffn_drop(self.ffn(normed_x))

        return x


class Ex01(nn.Module):
    def __init__(self, config: TrainingConfigEx01):
        super().__init__()
        # self.config is already set by parent class
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Create attention block with RoPE support
        self.attention_block = AttentionBlock(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            height=30,  # ARC-AGI uses 30x30 grids
            width=30,
            rope_base=config.rope_base,
            dropout=config.attention_dropout,
        )

        self.output_proj = nn.Linear(config.embed_dim, 11)
        self.num_layers = config.num_attention_layers
        self.frame_capture = FrameCapture()

    def forward(
        self,
        input_images: torch.Tensor,
        filenames: torch.Tensor,
        example_indices: torch.Tensor,
        context: torch.Tensor,
        current_epoch: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using the base model's expected signature.
        x: input tensor of shape [B, 30, 30, 11]
        """
        batch_size = input_images.shape[0]
        color_indices = input_images.argmax(dim=-1)  # [B, 30, 30]
        x1 = self.order2(color_indices).view(batch_size, 30 * 30, 45)
        eo = self.input_embed(x1)  # [B, 30*30, embed_dim]
        ec = self.color_embed(color_indices).view(batch_size, 30 * 30, -1)
        x2 = eo + ec
        self.frame_capture.enable()
        for i in range(self.num_layers):
            x2 = self.attention_block(x2)
            if current_epoch is not None and current_epoch % 10 == 0:
                x3 = self.project_to_color(x2, round_idx=i)
        x3 = self.project_to_color(x2)

        if current_epoch is not None and current_epoch % 100 == 0:
            self.frame_capture.disable()
            self.frame_capture.to_gif(
                f"gifs/message_rounds_{current_epoch}.gif", duration_ms=100
            )
        return x3

    def project_to_color(
        self, x2: torch.Tensor, round_idx: int | None = None
    ) -> torch.Tensor:
        x2 = self.output_proj(x2)
        x3 = x2.view(x2.shape[0], 30, 30, 11)
        if round_idx is not None:
            self.frame_capture.capture(x3, round_idx=round_idx)
        return x3


class MainModuleEx01(MainModule):
    def __init__(self, config: TrainingConfigEx01):
        super().__init__(config)
        self.config = config
        self.model = Ex01(config)
        self.save_hyperparameters(config.model_dump())


@click.command()
@from_pydantic(
    TrainingConfigEx01,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfigEx01.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(**kwargs: Any) -> None:
    # Extract the config from kwargs - it will be named based on the class
    config_key = next(k for k in kwargs if k.startswith("training_config"))
    training_config: TrainingConfigEx01 = kwargs[config_key]
    return main(training_config, MainModuleEx01)


if __name__ == "__main__":
    main_click()
