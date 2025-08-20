import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn
import math

from typing import Any, Optional

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.order2 import Order2Features
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx(TrainingConfig):
    project_name: str = "arcagi-base43"
    filename_filter: Optional[str] = "00d62c1b"
    lr: float = 1e-3

    max_epochs: int = 1000

    num_message_rounds: int = 96
    n_channels: int = 256

    # Attention module parameters
    embed_dim: int = 128
    dropout: float = 0.1

    # RoPE parameters
    rope_base: float = 10000.0


class ArsinhNorm(nn.Module):
    """Inverse Hyperbolic Sine Normalization - more stable than RMSNorm."""

    def __init__(self: "ArsinhNorm", dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply asinh normalization: asinh(x/scale)
        # asinh is smooth, handles pos/neg values, and has bounded gradients
        x_scaled = x / self.scale
        x_normed = torch.asinh(x_scaled)
        return self.weight * x_normed + self.bias


class Noise(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class SpatialMessagePassing(nn.Module):
    """Efficient spatial message passing for local consistency."""

    def __init__(
        self: "SpatialMessagePassing",
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Ensure groups divides hidden_dim evenly
        groups = hidden_dim // 16 if hidden_dim >= 16 else 1
        # Find the largest divisor of hidden_dim that's <= groups
        while groups > 1 and hidden_dim % groups != 0:
            groups -= 1

        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,  # Keep same dimension for residual connection
            kernel_size=3,
            padding=1,
            groups=groups,
        )
        self.norm = ArsinhNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Project to output dimension if needed
        self.out_proj = (
            nn.Linear(hidden_dim, out_dim) if out_dim != hidden_dim else None
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_conv = h.permute(0, 3, 1, 2)
        messages = self.conv(h_conv)
        messages = messages.permute(0, 2, 3, 1)
        messages = self.norm(messages)
        messages = self.activation(messages)
        messages = self.dropout(messages)
        h_out = h + messages

        # Project to output dimension if needed
        if self.out_proj is not None:
            h_out = self.out_proj(h_out)

        return h_out


class NeuralCellularAutomata(nn.Module):
    """
    Neural Cellular Automata for spatial pattern refinement on order2 features.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super(NeuralCellularAutomata, self).__init__()
        self.hidden_dim = hidden_dim

        # Learnable perception filters
        self.perception_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim * 3,  # 3 filters per input channel
            kernel_size=3,
            padding=1,
            groups=hidden_dim,  # Each input channel gets its own set of 3 filters
        )

        self.update_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Single NCA step.
        h: [B, 30, 30, hidden_dim] -> [B, 30, 30, hidden_dim]
        """
        # Convert to conv format: [B, C, H, W]
        x = h.permute(0, 3, 1, 2).contiguous()
        perceived = self.perception_conv(x)
        dx = self.update_net(perceived).permute(0, 2, 3, 1).contiguous()
        return h + dx


class SMPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nca = NeuralCellularAutomata(hidden_dim, dropout=dropout)
        self.smp = SpatialMessagePassing(hidden_dim, out_dim, dropout=dropout)
        self.norm1 = ArsinhNorm(hidden_dim)
        self.norm2 = ArsinhNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dx1 = self.nca(self.norm1(h))
        if self.training:
            dx1 = self.dropout(dx1)
        h = h + dx1
        dx2 = self.smp(self.norm2(h))
        if self.training:
            dx2 = self.dropout(dx2)
        h = h + dx2
        return h


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
            batch_size, height, width, channels = x.shape
            x_flat = x.view(batch_size, height * width, channels)
        else:
            batch_size, seq_len, channels = x.shape
            height = width = int(math.sqrt(seq_len))
            x_flat = x

        # Create position indices
        pos_h = torch.arange(height, device=x.device, dtype=x.dtype)
        pos_w = torch.arange(width, device=x.device, dtype=x.dtype)

        # Create 2D positional encodings
        dim_t = torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype)
        freqs = 1.0 / (self.base ** (dim_t / self.dim))

        # Apply to height and width separately
        pos_h = pos_h.unsqueeze(1) * freqs.unsqueeze(0)  # [H, dim/2]
        pos_w = pos_w.unsqueeze(1) * freqs.unsqueeze(0)  # [W, dim/2]

        # Create sin/cos embeddings
        pos_h_sin = torch.sin(pos_h).unsqueeze(1).expand(-1, width, -1)  # [H, W, dim/2]
        pos_h_cos = torch.cos(pos_h).unsqueeze(1).expand(-1, width, -1)  # [H, W, dim/2]
        pos_w_sin = (
            torch.sin(pos_w).unsqueeze(0).expand(height, -1, -1)
        )  # [H, W, dim/2]
        pos_w_cos = (
            torch.cos(pos_w).unsqueeze(0).expand(height, -1, -1)
        )  # [H, W, dim/2]

        # Flatten spatial dimensions
        pos_h_sin = pos_h_sin.reshape(height * width, -1)  # [H*W, dim/2]
        pos_h_cos = pos_h_cos.reshape(height * width, -1)  # [H*W, dim/2]
        pos_w_sin = pos_w_sin.reshape(height * width, -1)  # [H*W, dim/2]
        pos_w_cos = pos_w_cos.reshape(height * width, -1)  # [H*W, dim/2]

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
            x_rot = x_rot.view(batch_size, height, width, channels)

        return x_rot


class Ex(nn.Module):
    def __init__(self, config: TrainingConfigEx):
        super().__init__()
        self.num_message_rounds = config.num_message_rounds
        self.config = config

        # self.config is already set by parent class
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Initialize RoPE for 2D positional embeddings
        self.rope = RotaryPositionalEmbedding(config.embed_dim, base=config.rope_base)

        # Create attention block with RoPE support
        # Use n_channels as the working dimension throughout
        self.embed_to_channels = nn.Linear(config.embed_dim, config.n_channels)
        self.model = SMPBlock(
            hidden_dim=config.n_channels,
            out_dim=config.n_channels,
            dropout=config.dropout,
        )

        self.output_proj = nn.Linear(config.n_channels, 11)
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
        color_indices = input_images.argmax(dim=-1)  # [B, 30, 30]
        x1 = self.order2(color_indices)  # [B, 30, 30, 45]
        eo = self.input_embed(x1)  # [B, 30, 30, embed_dim]
        # ec = self.color_embed(color_indices)  # [B, 30, 30, embed_dim]
        # x2 = eo + ec  # [B, 30, 30, embed_dim]
        x2 = eo

        # Apply 2D RoPE embeddings
        x2 = self.rope(x2)  # [B, 30, 30, embed_dim]

        # Project to working dimension
        x2 = self.embed_to_channels(x2)  # [B, 30, 30, n_channels]

        self.frame_capture.enable()

        for i in range(self.num_message_rounds):
            x2 = self.model(x2)
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
        x2 = self.output_proj(x2)  # [B, 30, 30, 11]
        if round_idx is not None:
            self.frame_capture.capture(x2, round_idx=round_idx)
        return x2


class MainModuleEx01(MainModule):
    def __init__(self, config: TrainingConfigEx):
        super().__init__(config)
        self.config = config
        self.model = Ex(config)
        self.save_hyperparameters(config.model_dump())


@click.command()
@from_pydantic(
    TrainingConfigEx,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfigEx.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(**kwargs: Any) -> None:
    # Extract the config from kwargs - it will be named based on the class
    config_key = next(k for k in kwargs if k.startswith("training_config"))
    training_config: TrainingConfigEx = kwargs[config_key]
    return main(training_config, MainModuleEx01)


if __name__ == "__main__":
    main_click()
