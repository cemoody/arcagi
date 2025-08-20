import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.order2 import Order2Features
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx05(TrainingConfig):
    project_name: str = "arcagi-base43-ex05"
    filename_filter: Optional[str] = "007bbfb7"

    max_epochs: int = 1000

    num_message_rounds: int = 48  # Fewer rounds per scale
    n_channels: int = 256

    # Model parameters
    embed_dim: int = 128
    dropout: float = 0.1

    # Hierarchical parameters
    num_scales: int = 3
    scale_factors: tuple[float, ...] = (
        1.0,
        0.5,
        0.25,
    )  # Full, half, quarter resolution


class HierarchicalNCA(nn.Module):
    """
    Hierarchical NCA that processes multiple scales simultaneously.
    Information flows both within scales (horizontal) and between scales (vertical).
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_scales: int = 3,
        scale_factors: tuple[float, ...] = (1.0, 0.5, 0.25),
        hidden: int = 64,
        kernel_size: int = 3,
        fire_rate: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.scale_factors = scale_factors
        self.fire_rate = fire_rate

        # Create NCA modules for each scale
        self.scale_ncas = nn.ModuleList()
        for i in range(num_scales):
            scale_nca = nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(channels),
                    "perception": nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=channels,
                        bias=False,
                    ),
                    "update_net": nn.Sequential(
                        nn.Linear(channels + out_channels, hidden),
                        nn.Dropout(dropout),
                        nn.GELU(),
                        nn.Linear(hidden, channels),
                    ),
                }
            )
            self.scale_ncas.append(scale_nca)

        # Cross-scale interaction modules
        self.upsample_modules = nn.ModuleList()
        self.downsample_modules = nn.ModuleList()

        for i in range(num_scales - 1):
            # Downsampling: from finer to coarser scale
            self.downsample_modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
            )

            # Upsampling: from coarser to finer scale
            self.upsample_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels, channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GELU(),
                )
            )

        # Cross-scale fusion
        self.fusion_norms = nn.ModuleList(
            [nn.LayerNorm(channels * 2) for _ in range(num_scales)]
        )
        self.fusion_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels * 2, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, channels),
                )
                for _ in range(num_scales)
            ]
        )

    def create_pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Create multi-scale pyramid from input."""
        B, H, W, C = x.shape
        pyramid = [x]

        # Create coarser scales
        x_conv = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        for i in range(1, self.num_scales):
            scale_factor = self.scale_factors[i] / self.scale_factors[0]
            x_scaled = F.interpolate(
                x_conv, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
            pyramid.append(x_scaled.permute(0, 2, 3, 1))  # Back to [B, H', W', C]

        return pyramid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # Create multi-scale representations
        pyramid = self.create_pyramid(x)

        # Process each scale independently
        scale_updates = []
        for i, (scale_x, scale_nca) in enumerate(zip(pyramid, self.scale_ncas)):
            # Local perception and update at this scale
            x_norm = scale_nca["norm"](scale_x)
            x_perm = x_norm.permute(0, 3, 1, 2)  # [B, C, H', W']

            perception = scale_nca["perception"](x_perm)
            perception = perception.permute(0, 2, 3, 1)  # [B, H', W', out_channels]

            update_in = torch.cat([x_norm, perception], dim=-1)
            delta = scale_nca["update_net"](update_in)

            scale_updates.append(delta)

        # Cross-scale interactions
        fused_updates = []

        for i in range(self.num_scales):
            features_to_fuse = []

            # Get update from current scale
            features_to_fuse.append(scale_updates[i])

            # Get information from other scales
            if i > 0:  # Has finer scale
                # Downsample from finer scale
                finer_update = scale_updates[i - 1]
                finer_perm = finer_update.permute(0, 3, 1, 2)
                downsampled = self.downsample_modules[i - 1](finer_perm)
                downsampled = downsampled.permute(0, 2, 3, 1)

                # Ensure size matches
                target_h, target_w = pyramid[i].shape[1:3]
                if downsampled.shape[1:3] != (target_h, target_w):
                    downsampled_perm = downsampled.permute(0, 3, 1, 2)
                    downsampled_perm = F.interpolate(
                        downsampled_perm,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    downsampled = downsampled_perm.permute(0, 2, 3, 1)

                features_to_fuse.append(downsampled)

            if i < self.num_scales - 1:  # Has coarser scale
                # Upsample from coarser scale
                coarser_update = scale_updates[i + 1]
                coarser_perm = coarser_update.permute(0, 3, 1, 2)
                upsampled = self.upsample_modules[i](coarser_perm)
                upsampled = upsampled.permute(0, 2, 3, 1)

                # Ensure size matches
                target_h, target_w = pyramid[i].shape[1:3]
                if upsampled.shape[1:3] != (target_h, target_w):
                    upsampled_perm = upsampled.permute(0, 3, 1, 2)
                    upsampled_perm = F.interpolate(
                        upsampled_perm,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    upsampled = upsampled_perm.permute(0, 2, 3, 1)

                features_to_fuse.append(upsampled)

            # Fuse features from different scales
            if len(features_to_fuse) > 1:
                fused = torch.cat(features_to_fuse, dim=-1)
                fused = self.fusion_norms[i](fused)
                fused_update = self.fusion_mlps[i](fused)
            else:
                fused_update = features_to_fuse[0]

            fused_updates.append(fused_update)

        # Apply updates with stochastic masking
        final_pyramid = []
        for i, (scale_x, fused_update) in enumerate(zip(pyramid, fused_updates)):
            if self.training and self.fire_rate < 1.0:
                mask_shape = (scale_x.shape[0], scale_x.shape[1], scale_x.shape[2], 1)
                mask = (
                    torch.rand(mask_shape, device=x.device) < self.fire_rate
                ).float()
                fused_update = fused_update * mask

            final_pyramid.append(scale_x + fused_update)

        # Return the finest scale result
        return final_pyramid[0]


class Ex05(nn.Module):
    def __init__(self, config: TrainingConfigEx05):
        super().__init__()
        self.num_message_rounds = config.num_message_rounds
        self.config = config

        # Input embeddings
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Hierarchical NCA
        self.model = HierarchicalNCA(
            channels=config.embed_dim,
            out_channels=config.n_channels,
            num_scales=config.num_scales,
            scale_factors=config.scale_factors,
            dropout=config.dropout,
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
                f"gifs/message_rounds_{current_epoch}_ex05.gif", duration_ms=100
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


class MainModuleEx05(MainModule):
    def __init__(self, config: TrainingConfigEx05):
        super().__init__(config)
        self.config = config
        self.model = Ex05(config)
        self.save_hyperparameters(config.model_dump())


@click.command()
@from_pydantic(
    TrainingConfigEx05,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfigEx05.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(**kwargs: Any) -> None:
    config_key = next(k for k in kwargs if k.startswith("training_config"))
    training_config: TrainingConfigEx05 = kwargs[config_key]
    return main(training_config, MainModuleEx05)


if __name__ == "__main__":
    main_click()
