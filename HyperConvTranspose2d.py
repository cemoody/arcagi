import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional


class HyperConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        condition_dim: int = 64,
        condition_dims: List[int] = [512],
    ) -> None:
        """
        Hypernetwork implementation of ConvTranspose2d that generates weights from a condition vector.

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            output_padding: Additional size added to one side of the output shape
            groups: Number of blocked connections from input channels to output channels
            bias: If True, adds a learnable bias to the output
            dilation: Spacing between kernel elements
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'
            device: Device on which the convolution will be performed
            dtype: Data type of the convolution weights
            condition_dim: Dimension of the condition vector
            condition_dims: List of hidden dimensions for the hypernetwork
        """
        super().__init__(  # type: ignore
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

        # Convert kernel_size to tuple if it's an int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Calculate number of parameters in the convolution
        weight_params = (
            in_channels * (out_channels // groups) * kernel_size[0] * kernel_size[1]
        )
        bias_params = out_channels if bias else 0
        total_params = weight_params + bias_params

        # Build hypernetwork
        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(condition_dim, condition_dims[0]))
        layers.append(nn.GELU())

        # Hidden layers
        for i in range(len(condition_dims) - 1):
            layers.append(nn.Linear(condition_dims[i], condition_dims[i + 1]))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(condition_dims[-1], total_params))

        # Convert to Sequential for proper registration
        self.hypernet = nn.Sequential(*layers)

        # Store dimensions for weight reshaping
        self.weight_shape = (
            in_channels,
            out_channels // groups,
            kernel_size[0],
            kernel_size[1],
        )
        self.bias_shape = (out_channels,) if bias else None
        self.weight_params = weight_params
        self.has_bias = bias

    def forward(
        self,
        input: torch.Tensor,
        output_size: Optional[List[int]] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dynamically generated weights from condition.

        Args:
            input: Input tensor of shape [B, in_channels, H, W]
            output_size: Size of the output (unused, kept for compatibility with parent class)
            condition: Condition tensor of shape [condition_dim]

        Returns:
            Output tensor of shape [B, out_channels, H_out, W_out]
        """
        # Check if condition was provided
        if condition is None:
            raise ValueError("Condition tensor must be provided")

        # Generate weights from condition
        params = self.hypernet(condition)

        # Split into weight and bias
        if self.has_bias:
            weight_params = params[: self.weight_params]
            bias_params = params[self.weight_params :]

            # Reshape to proper dimensions
            weight = weight_params.view(self.weight_shape)
            bias = bias_params.view(self.bias_shape)
        else:
            weight = params.view(self.weight_shape)
            bias = None

        # Use F.conv_transpose2d with generated weights
        return F.conv_transpose2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding if isinstance(self.padding, (int, tuple)) else 0,
            self.output_padding,
            self.groups,
            self.dilation,
        )
