import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn

from typing import Any, Optional, Tuple

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.order2 import Order2Features
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx(TrainingConfig):
    project_name: str = "arcagi-base43-ex08"
    filename_filter: Optional[str] = "007bbfb7"

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


class NeuralCellularAutomata(nn.Module):
    """
    Multiscale NCA operating over a 2D grid of cells.
    State has C channels per cell. Forward consumes/returns [B, H, W, C].

    Steps:
      1) Perception: multiple depthwise Conv2d with different dilations for multiscale perception.
      2) Update: small MLP on [state || multiscale_perception] per cell -> delta.
      3) Apply: x <- x + mask * delta (stochastic update controlled by fire_rate).
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        hidden: int = 64,
        kernel_size: int = 3,
        fire_rate: float = 1.0,  # probability a cell updates on a step
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),  # different scales
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.channels = channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.fire_rate = fire_rate
        self.dilation_rates = dilation_rates

        # Create multiple perception layers with different dilation rates
        # For depthwise convolution, out_channels must equal in_channels when groups=channels
        self.perception_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,  # Must equal in_channels for depthwise conv
                    kernel_size=kernel_size,
                    padding=dilation
                    * (kernel_size // 2),  # padding = dilation * (kernel_size // 2)
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in dilation_rates
            ]
        )

        # Update rule: concatenates [state, multiscale_perceptions] along channel dim per cell
        # Input size: channels (state) + len(dilation_rates) * channels (perceptions)
        self.update = nn.Sequential(
            nn.Linear(channels + len(dilation_rates) * channels, hidden),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden, out_channels),  # Output the desired number of channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C] -> returns [B, H, W, out_channels]
        """
        B, H, W, C = x.shape
        assert C == self.channels, f"Expected C={self.channels}, got {C}"

        # Perception over spatial dims: Conv2d expects [B, C, H, W]
        x_perm = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply all perception layers with different dilation rates
        perceptions: list[torch.Tensor] = []
        for perception_layer in self.perception_layers:
            p = perception_layer(x_perm).permute(0, 2, 3, 1)  # [B, H, W, C]
            perceptions.append(p)

        # Concatenate state with all perceptions
        upd_in = torch.cat(
            [x] + perceptions, dim=-1
        )  # [B, H, W, C * (1 + len(dilation_rates))]
        output = self.update(upd_in)  # [B, H, W, out_channels]

        # For residual connection, only apply if output channels match input channels
        if self.out_channels == self.channels:
            # Stochastic update mask (per cell)
            if self.training and self.fire_rate < 1.0:
                mask = (
                    torch.rand(B, H, W, 1, device=x.device, dtype=x.dtype)
                    < self.fire_rate
                ).to(x.dtype)
                delta = output * mask
            else:
                delta = output
            output = x + delta

        return output


class Attention(nn.Module):
    """Cross-attention module that attends from context to spatial features."""

    def __init__(self, channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        # Project spatial features to query space
        self.query_proj = nn.Linear(channels, out_channels)
        # Project context to key and value spaces
        self.key_proj = nn.Linear(out_channels, out_channels)
        self.value_proj = nn.Linear(out_channels, out_channels)
        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)

        # Normalization and dropout
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = out_channels**-0.5

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: spatial features [B, H, W, channels]
        c: context features [B, out_channels]
        Returns: attended context [B, out_channels]
        """
        B, H, W, C = x.shape

        # Compute queries from spatial features
        q = self.query_proj(x)  # [B, H, W, out_channels]
        q = q.view(B, H * W, self.out_channels)  # [B, HW, out_channels]

        # Compute keys and values from context
        k = self.key_proj(c).unsqueeze(1)  # [B, 1, out_channels]
        v = self.value_proj(c).unsqueeze(1)  # [B, 1, out_channels]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, HW, 1]
        attn_weights = torch.softmax(scores, dim=1)  # [B, HW, 1]

        # Apply attention to values
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(
            attn_weights.transpose(-2, -1), v.expand(B, H * W, self.out_channels)
        )  # [B, 1, out_channels]
        attended = attended.squeeze(1)  # [B, out_channels]

        # Apply output projection and residual connection
        output = self.out_proj(attended)
        output = self.dropout(output)
        output = self.norm(output + c)

        return output


class NCAAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.channels = channels
        # NCA should accept input of size channels + out_channels (after concatenation)
        # and output channels size (same as input x)
        self.nca = NeuralCellularAutomata(
            channels=channels + out_channels,
            out_channels=channels,  # Output same size as input channels
            dropout=dropout,
        )
        self.attention = Attention(channels, out_channels, dropout=dropout)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = c + self.attention(x, c)  # shape [B, n_c]
        c_expanded = c.unsqueeze(1).unsqueeze(1)  # shape [B, 1, 1, n_c]
        # Expand context to match spatial dimensions
        B, H, W, C = x.shape
        c_expanded = c_expanded.expand(B, H, W, -1)  # shape [B, H, W, n_c]
        xc = torch.cat([x, c_expanded], dim=-1)  # shape [B, H, W, embed_dim + n_c]
        y = self.nca(xc)  # shape [B, H, W, embed_dim]
        return y, c


class Ex(nn.Module):
    def __init__(self, config: TrainingConfigEx):
        super().__init__()
        self.num_message_rounds = config.num_message_rounds
        self.config = config

        # self.config is already set by parent class
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Create attention block with RoPE support
        self.model = NCAAttention(
            channels=config.embed_dim,
            out_channels=config.n_channels,
            dropout=config.dropout,
        )
        self.noise = Noise(config.dropout)

        # Initialize context embedding
        self.context_embed = nn.Parameter(torch.randn(1, config.n_channels))

        self.output_proj = nn.Linear(config.embed_dim, 11)
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
        ec = self.color_embed(color_indices)  # [B, 30, 30, embed_dim]
        x2 = eo + ec  # [B, 30, 30, embed_dim]
        self.frame_capture.enable()

        B = input_images.shape[0]
        c = torch.zeros(B, self.config.n_channels, device=x2.device)

        for i in range(self.num_message_rounds):
            x2, c = self.model(x2, c)
            # if self.training:
            #     x2 = self.noise(x2)
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
