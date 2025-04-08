import pytest
import torch
from typing import Tuple, List
from arcagi.HyperConvTranspose2d import HyperConvTranspose2d


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, kernel_size, stride, padding, output_padding, input_size, expected_output_size",
    [
        # Basic case
        (2, 16, 8, 3, 1, 0, 0, (16, 16), (18, 18)),
        # With stride 2 (upsampling)
        (2, 32, 16, 4, 2, 1, 0, (8, 8), (16, 16)),
        # With output padding
        (2, 64, 32, 3, 2, 1, 1, (10, 10), (20, 20)),
        # Rectangular input
        (2, 16, 8, 3, 2, 1, 0, (8, 12), (15, 23)),
        # Different kernel sizes
        (2, 32, 16, 5, 2, 2, 0, (10, 10), (19, 19)),
    ],
)
def test__forward__should_match_expected_size(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    input_size: Tuple[int, int],
    expected_output_size: Tuple[int, int],
) -> None:
    # Arrange
    condition_dim: int = 64
    condition_dims: List[int] = [128, 256]

    # Create the hypernetwork layer
    layer = HyperConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        condition_dim=condition_dim,
        condition_dims=condition_dims,
    )

    # Create input tensor and condition vector
    input_tensor = torch.randn(batch_size, in_channels, input_size[0], input_size[1])
    condition = torch.randn(batch_size, condition_dim)

    # Act
    output = layer(input_tensor, condition=condition)

    # Assert
    assert output.shape == (
        batch_size,
        out_channels,
        expected_output_size[0],
        expected_output_size[1],
    ), (
        f"Expected output shape {(batch_size, out_channels, expected_output_size[0], expected_output_size[1])}, "
        f"but got {output.shape}"
    )
