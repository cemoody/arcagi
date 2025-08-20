import click
from pydanclick import from_pydantic

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional

from arcagi.color_mapping.base43 import TrainingConfig, MainModule, main
from arcagi.order2 import Order2Features
from arcagi.utils.frame_capture import FrameCapture


class TrainingConfigEx06(TrainingConfig):
    project_name: str = "arcagi-base43-ex06"
    filename_filter: Optional[str] = "007bbfb7"

    max_epochs: int = 1000

    max_message_rounds: int = 128  # Maximum rounds
    min_message_rounds: int = 16  # Minimum rounds
    n_channels: int = 256

    # Model parameters
    embed_dim: int = 128
    dropout: float = 0.1

    # Adaptive computation parameters
    halt_threshold: float = 0.01  # Halting threshold
    halt_penalty: float = 0.01  # Penalty for computation time


class AdaptiveNCA(nn.Module):
    """
    NCA with adaptive computation time - dynamically decides when to stop processing.
    Uses a halting mechanism similar to Adaptive Computation Time (ACT).
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        hidden: int = 64,
        kernel_size: int = 3,
        fire_rate: float = 1.0,
        dilation_rates: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        halt_threshold: float = 0.01,
    ):
        super().__init__()
        self.channels = channels
        self.halt_threshold = halt_threshold

        # Main NCA components
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels + len(dilation_rates) * out_channels)

        # Perception layers
        self.perception_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=dilation * (kernel_size // 2),
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in dilation_rates
            ]
        )

        # Update network
        self.update = nn.Sequential(
            nn.Linear(channels + len(dilation_rates) * out_channels, hidden * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        # Halting unit - produces probability of halting at each position
        self.halt_unit = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        self.fire_rate = fire_rate

    def forward_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single NCA step returning update and halt probabilities."""
        B, H, W, C = x.shape

        # Compute halt probabilities before update
        halt_probs = self.halt_unit(x).squeeze(-1)  # [B, H, W]

        # Regular NCA update
        x_norm = self.norm1(x)
        x_perm = x_norm.permute(0, 3, 1, 2)

        perceptions = []
        for perception_layer in self.perception_layers:
            p = perception_layer(x_perm).permute(0, 2, 3, 1)
            perceptions.append(p)

        update_in = torch.cat([x_norm] + perceptions, dim=-1)
        update_in = self.norm2(update_in)
        delta = self.update(update_in)

        # Stochastic masking
        if self.training and self.fire_rate < 1.0:
            mask = (torch.rand(B, H, W, 1, device=x.device) < self.fire_rate).float()
            delta = delta * mask

        return delta, halt_probs

    def forward(
        self,
        x: torch.Tensor,
        max_steps: int,
        min_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with adaptive computation.
        Returns: (output, halted_mask, ponder_cost)
        """
        B, H, W, C = x.shape
        device = x.device

        # Initialize halting variables
        halted = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        accumulated_halt = torch.zeros(B, H, W, device=device)
        step_outputs = torch.zeros_like(x)
        ponder_cost = torch.zeros(B, H, W, device=device)

        for step in range(max_steps):
            # Get update and halt probabilities
            delta, halt_probs = self.forward_step(x)

            # Update positions that haven't halted
            active = ~halted
            x = x + delta * active.unsqueeze(-1).float()

            # Accumulate outputs weighted by halt probability
            if step >= min_steps - 1:  # Only start halting after minimum steps
                # Compute halting updates
                still_active = active.float()

                # Update accumulated halt probabilities
                accumulated_halt = accumulated_halt + halt_probs * still_active

                # Determine which positions should halt
                should_halt = (accumulated_halt > 1.0 - self.halt_threshold) & active

                # For positions that halt, add remaining probability
                halt_weight = torch.where(
                    should_halt,
                    1.0 - (accumulated_halt - halt_probs),  # Remainder
                    halt_probs * still_active,  # Normal weight
                )

                # Accumulate weighted outputs
                step_outputs = step_outputs + x * halt_weight.unsqueeze(-1)

                # Update halted mask
                halted = halted | should_halt

                # Update ponder cost
                ponder_cost = ponder_cost + still_active

                # Early stopping if all positions have halted
                if halted.all():
                    break
            else:
                # Before min_steps, just accumulate normally
                ponder_cost = ponder_cost + torch.ones_like(ponder_cost)

        # Handle any remaining active positions
        if not halted.all():
            remaining = (~halted).float()
            remaining_weight = remaining * (1.0 - accumulated_halt)
            step_outputs = step_outputs + x * remaining_weight.unsqueeze(-1)

        # Normalize ponder cost
        ponder_cost = ponder_cost / max_steps

        return step_outputs, halted, ponder_cost


class Ex06(nn.Module):
    def __init__(self, config: TrainingConfigEx06):
        super().__init__()
        self.max_message_rounds = config.max_message_rounds
        self.min_message_rounds = config.min_message_rounds
        self.config = config

        # Input embeddings
        self.input_embed = nn.Linear(45, config.embed_dim)
        self.color_embed = nn.Embedding(11, config.embed_dim)
        self.order2 = Order2Features()

        # Adaptive NCA
        self.model = AdaptiveNCA(
            channels=config.embed_dim,
            out_channels=config.n_channels,
            dropout=config.dropout,
            halt_threshold=config.halt_threshold,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, 11)

        # Frame capture
        self.frame_capture = FrameCapture()

        # For tracking computation cost
        self.halt_penalty = config.halt_penalty

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

        # Run adaptive NCA
        x2, halted, ponder_cost = self.model(
            x2, max_steps=self.max_message_rounds, min_steps=self.min_message_rounds
        )

        # Store ponder cost for loss computation
        self.last_ponder_cost = ponder_cost.mean()

        # Log some statistics during training
        if self.training and current_epoch is not None:
            avg_steps = ponder_cost.mean() * self.max_message_rounds
            min_steps = ponder_cost.min() * self.max_message_rounds
            max_steps = ponder_cost.max() * self.max_message_rounds

            if current_epoch % 10 == 0:
                print(
                    f"Epoch {current_epoch}: Avg steps: {avg_steps:.1f}, "
                    f"Min: {min_steps:.1f}, Max: {max_steps:.1f}"
                )

        # Project to output
        x3 = self.project_to_color(x2)

        return x3

    def project_to_color(
        self, x2: torch.Tensor, round_idx: int | None = None
    ) -> torch.Tensor:
        x2_norm = self.output_norm(x2)
        x2_out = self.output_proj(x2_norm)  # [B, 30, 30, 11]
        if round_idx is not None:
            self.frame_capture.capture(x2_out, round_idx=round_idx)
        return x2_out


class MainModuleEx06(MainModule):
    def __init__(self, config: TrainingConfigEx06):
        super().__init__(config)
        self.config = config
        self.model = Ex06(config)
        self.save_hyperparameters(config.model_dump())

    def step(
        self,
        batch: tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        is_train: bool = True,
    ) -> torch.Tensor:
        """Override step to add ponder cost to loss."""
        # Regular forward pass and loss computation
        loss = super().step(batch, batch_idx, is_train)

        # Add ponder cost penalty
        if hasattr(self.model, "last_ponder_cost"):
            ponder_penalty = self.model.halt_penalty * self.model.last_ponder_cost
            loss = loss + ponder_penalty

            # Log ponder cost
            prefix = "train" if is_train else "val"
            self.log(
                f"{prefix}_ponder_cost", self.model.last_ponder_cost, on_step=is_train
            )

        return loss


@click.command()
@from_pydantic(
    TrainingConfigEx06,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfigEx06.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(**kwargs: Any) -> None:
    config_key = next(k for k in kwargs if k.startswith("training_config"))
    training_config: TrainingConfigEx06 = kwargs[config_key]
    return main(training_config, MainModuleEx06)


if __name__ == "__main__":
    main_click()
