import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn

from typing import Any, Optional

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.order2 import Order2Features
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx03(TrainingConfig):
    project_name: str = "arcagi-base43-ex03"
    filename_filter: Optional[str] = "007bbfb7"

    max_epochs: int = 1000

    num_message_rounds: int = 96
    n_channels: int = 256

    # Attention module parameters
    embed_dim: int = 128
    dropout: float = 0.1

    # RoPE parameters
    rope_base: float = 10000.0

    # Residual scaling
    residual_scale: float = 0.1


class LayerScale(nn.Module):
    """Layer scale for stable training of deep networks."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class ResidualNeuralCellularAutomata(nn.Module):
    """
    Improved NCA with residual connections and layer normalization.
    Uses layer normalization before each operation and residual connections with scaling.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        hidden: int = 64,
        kernel_size: int = 3,
        fire_rate: float = 1.0,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.channels = channels
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.fire_rate = fire_rate
        self.dilation_rates = dilation_rates
        self.residual_scale = residual_scale

        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels + len(dilation_rates) * out_channels)

        # Create multiple perception layers with different dilation rates
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

        # Update rule with improved architecture
        self.update = nn.Sequential(
            nn.Linear(channels + len(dilation_rates) * out_channels, hidden * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        # Layer scale for stable residual connections
        self.layer_scale = LayerScale(channels, init_value=residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C] -> returns [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert C == self.channels, f"Expected C={self.channels}, got {C}"

        # Store residual
        residual = x

        # Apply layer norm
        x_norm = self.norm1(x)

        # Perception over spatial dims: Conv2d expects [B, C, H, W]
        x_perm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply all perception layers with different dilation rates
        perceptions: list[torch.Tensor] = []
        for perception_layer in self.perception_layers:
            p = perception_layer(x_perm).permute(0, 2, 3, 1)  # [B, H, W, C]
            perceptions.append(p)

        # Concatenate normalized state with all perceptions
        upd_in = torch.cat(
            [x_norm] + perceptions, dim=-1
        )  # [B, H, W, C * (1 + len(dilation_rates))]

        # Apply second layer norm
        upd_in = self.norm2(upd_in)

        # Compute update
        delta = self.update(upd_in)  # [B, H, W, C]

        # Apply layer scale
        delta = self.layer_scale(delta)

        # Stochastic update mask (per cell)
        if self.training and self.fire_rate < 1.0:
            mask = (
                torch.rand(B, H, W, 1, device=x.device, dtype=x.dtype) < self.fire_rate
            ).to(x.dtype)
            delta = delta * mask

        # Residual connection
        x = residual + delta
        return x


class Ex03(nn.Module):
    def __init__(self, config: TrainingConfigEx03):
        super().__init__()
        self.num_message_rounds = config.num_message_rounds
        self.config = config

        # Input embeddings
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Improved NCA with residual connections
        self.model = ResidualNeuralCellularAutomata(
            channels=config.embed_dim,
            out_channels=config.n_channels,
            dropout=config.dropout,
            residual_scale=config.residual_scale,
        )

        # Output projection with layer norm
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, 11)

        # Frame capture for visualization
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

        # Enable frame capture for visualization epochs
        if current_epoch is not None and current_epoch % 100 == 0:
            self.frame_capture.enable()

        # Message passing rounds
        for i in range(self.num_message_rounds):
            x2 = self.model(x2)
            if current_epoch is not None and current_epoch % 10 == 0:
                x3 = self.project_to_color(x2, round_idx=i)

        # Final projection
        x3 = self.project_to_color(x2)

        # Save visualization if enabled
        if current_epoch is not None and current_epoch % 100 == 0:
            self.frame_capture.disable()
            self.frame_capture.to_gif(
                f"gifs/message_rounds_{current_epoch}_ex03.gif", duration_ms=100
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


class MainModuleEx03(MainModule):
    def __init__(self, config: TrainingConfigEx03):
        super().__init__(config)
        self.config = config
        self.model = Ex03(config)
        self.save_hyperparameters(config.model_dump())


@click.command()
@from_pydantic(
    TrainingConfigEx03,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfigEx03.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(**kwargs: Any) -> None:
    # Extract the config from kwargs - it will be named based on the class
    config_key = next(k for k in kwargs if k.startswith("training_config"))
    training_config: TrainingConfigEx03 = kwargs[config_key]
    return main(training_config, MainModuleEx03)


if __name__ == "__main__":
    main_click()
