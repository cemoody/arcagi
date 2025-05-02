# %%python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Optional,
    Union,
)

# Type alias for PyTorch Lightning metrics
from torch import Tensor

# Type alias for metrics
Metric = Union[Tensor, float, int]

# --- Helper Modules ---


class DoubleConv(nn.Module):
    """(Convolution => Norm => ReLU) * 2"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()  # type: ignore
        if not mid_channels:
            mid_channels = out_channels
        # Use padding='same' to preserve dimensions with 3x3 kernels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding="same", bias=False
            ),
            nn.LayerNorm([mid_channels, 30, 30]),  # Normalize over C,H,W
            nn.GELU(),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding="same", bias=False
            ),
            nn.LayerNorm([out_channels, 30, 30]),  # Normalize over C,H,W
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()  # type: ignore
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super().__init__()  # type: ignore
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CHW
        # Handle potential size mismatch from pooling (e.g., 15x15 -> 30x30 vs 30x30)
        diffY: int = x2.size()[2] - x1.size()[2]
        diffX: int = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2's spatial dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map features to class logits"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()  # type: ignore
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate mechanism for U-Net skip connections.
    Filters features from the encoder path based on context from the decoder path.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        """
        Args:
            F_g: Channels in the gating signal (from decoder).
            F_l: Channels in the input signal (from encoder).
            F_int: Channels in the intermediate layer.
        """
        super(AttentionGate, self).__init__()  # type: ignore
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LayerNorm([F_int, 15, 15]),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LayerNorm([F_int, 15, 15]),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LayerNorm([1, 15, 15]),
            nn.Sigmoid(),
        )
        self.relu = nn.GELU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from the decoder path.
            x: Input signal from the encoder path.
        """
        # Ensure g and x have the same spatial dimensions before processing
        # Usually g is upsampled before being passed to the AG
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Multiply the input signal (x) by the attention coefficients (psi)
        return x * psi


class ASPPConv(nn.Sequential):
    """Convolution with specified dilation rate"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.LayerNorm([out_channels, 7, 7]),
            nn.GELU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    """Global Average Pooling branch for ASPP"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__()  # type: ignore
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.LayerNorm([out_channels, 1, 1]),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]  # Get the last two dimensions as a tuple
        x = self.aspp_pooling(x)
        # Upsample back to original size
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)  # type: ignore


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""

    def __init__(
        self, in_channels: int, out_channels: int, atrous_rates: List[int]
    ) -> None:
        super(ASPP, self).__init__()  # type: ignore
        modules: List[nn.Module] = []
        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.LayerNorm([out_channels, 7, 7]),
                nn.GELU(),
            )
        )

        # Dilated Convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global Pooling Branch
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Final convolution to consolidate features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.LayerNorm([out_channels, 7, 7]),
            nn.GELU(),
            nn.Dropout(0.5),
        )  # Optional dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: List[torch.Tensor] = []
        for conv in self.convs:
            res.append(conv(x))
        res_cat = torch.cat(res, dim=1)
        return self.project(res_cat)


# --- PyTorch Lightning Module ---


class UNet30x30(nn.Module):
    """
    A specialized U-Net architecture designed specifically for 30x30 images with 11 channels.

    This model is optimized for processing small, fixed-size images where the input and output
    dimensions are identical (30x30 with 11 channels). It includes:

    - Two downsampling steps (30x30 -> 15x15 -> 7x7)
    - ASPP bottleneck for enhanced feature extraction
    - Two upsampling steps to restore original dimensions
    - Optional attention gates for better feature selection
    - Optional residual connection from input to output

    The model enforces that inputs and outputs must be 30x30 with 11 channels.
    """

    def __init__(
        self,
        n_channels: int = 11,  # Input channels (11 for one-hot encoded colors)
        n_classes: int = 11,  # Output classes (11 colors)
        base_features: int = 32,  # Number of features in the first layer
        bilinear_upsample: bool = True,  # Use bilinear upsampling? (Recommended True)
        learning_rate: float = 1e-4,
        dice_ce_weight_ratio: float = 0.5,
    ) -> None:  # Weight for Dice Loss (CE weight = 1 - ratio)
        super().__init__()  # type: ignore

        # Enforce that input and output dimensions match for residual connection
        assert (
            n_channels == n_classes == 11
        ), "Input channels and output classes must be 11 for this model"

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear_upsample
        self.learning_rate = learning_rate
        self.dice_ce_weight_ratio = dice_ce_weight_ratio

        factor = 2 if bilinear_upsample else 1
        f = base_features  # 32

        # --- Encoder ---
        # Input: 30x30
        self.inc = DoubleConv(n_channels, f)  # Output: f x 30x30
        self.down1 = Down(f, f * 2)  # Output: f*2 x 15x15
        self.down2 = Down(
            f * 2, f * 4
        )  # Output: f*4 x 7x7 (or 8x8 if floor/ceil differs)

        # --- Bottleneck with ASPP ---
        # ASPP Input: f*4 x 7x7
        # Use dilation rates suitable for small feature maps (e.g., 7x7)
        aspp_rates: List[int] = [1, 2, 3]  # Smaller rates for smaller feature maps
        aspp_out_channels = f * 8  # Intermediate channel size in ASPP
        self.aspp = ASPP(
            in_channels=f * 4, out_channels=aspp_out_channels, atrous_rates=aspp_rates
        )
        # ASPP Output: f*8 x 7x7

        # --- Decoder ---
        # Note: Decoder input channels need to match concatenation
        # Up1 input: ASPP output (f*8) + skip connection (f*4) = f*12
        self.up1_in_channels = aspp_out_channels + f * 4
        self.up1 = Up(
            self.up1_in_channels, f * 2 * factor, bilinear_upsample
        )  # Output: f*2 x 15x15

        # Up2 input: Up1 output (f*2) + skip connection (f) = f*3
        self.up2_in_channels = f * 2 + f
        self.up2 = Up(
            self.up2_in_channels, f * factor, bilinear_upsample
        )  # Output: f x 30x30

        # --- Attention Gates (Optional) ---
        # AG for skip connection 1 (down1 output)
        self.att1 = AttentionGate(F_g=f * 2 * factor, F_l=f * 2, F_int=f * 1)
        # AG for skip connection 2 (inc output)
        self.att2 = AttentionGate(F_g=f * factor, F_l=f, F_int=f // 2)

        # --- Output Layer ---
        self.outc = OutConv(f, n_classes)  # Output: n_classes x 30x30

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input dimensions (30x30 with 11 channels)
        _, channels, height, width = x.shape
        assert (
            channels == self.n_channels
        ), f"Expected {self.n_channels} input channels, got {channels}"
        assert height == width == 30, f"Expected 30x30 input, got {height}x{width}"

        # Store original input for residual connection
        input_x = x

        # Encoder
        x1 = self.inc(x)  # -> f x 30x30
        x2 = self.down1(x1)  # -> f*2 x 15x15
        x3 = self.down2(x2)  # -> f*4 x 7x7

        # Bottleneck
        x_bottle = self.aspp(x3)  # -> f*8 x 7x7

        # Decoder
        # Apply Attention Gate 1 (optional)
        x2_att = self.att1(
            g=self.up1.up(x_bottle), x=x2
        )  # Gate uses upsampled bottleneck output
        x_up1 = self.up1(
            x_bottle, x2_att
        )  # Upsample bottleneck, concat gated x2 -> f*2 x 15x15

        # Apply Attention Gate 2 (optional)
        x1_att = self.att2(g=self.up2.up(x_up1), x=x1)  # Gate uses upsampled up1 output
        x_up2 = self.up2(x_up1, x1_att)  # Upsample up1, concat gated x1 -> f x 30x30

        # Output
        logits = self.outc(x_up2)  # -> n_classes x 30x30

        # Direct addition since channels match
        logits = logits + input_x

        return logits


class ResidualRepeatedFixedWeightsUNet(nn.Module):

    def __init__(self, num_layers: int, base_module: nn.Module) -> None:
        super().__init__()  # type: ignore
        self.repeated_module = base_module
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_layers):
            x = self.repeated_module(x) + x
        return x
