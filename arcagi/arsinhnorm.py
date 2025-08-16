import torch
import torch.nn as nn


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
