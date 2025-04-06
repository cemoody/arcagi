import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any

# Import with explicit type
from HyperConvTranspose2d import HyperConvTranspose2d
from HyperConv2d import HyperConv2d


class CVAE(nn.Module):
    def __init__(
        self,
        img_size: int = 30,
        n_colors: int = 11,
        latent_dim: int = 64,
        condition_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        is_hypernetwork: bool = False,
    ) -> None:
        """
        Conditional Variational Autoencoder that transforms input images to output images.

        Args:
            img_size: Size of the square images (both height and width)
            n_colors: Number of possible color values (-1 to 9 inclusive)
            latent_dim: Dimension of the latent space
            condition_dim: Dimension of the condition vector (1 for binary is_input flag)
            hidden_dims: List of hidden dimensions for the encoder/decoder networks
            is_hypernetwork: If True, use condition to generate conv layer weights instead of concatenating
        """
        super().__init__()  # type: ignore

        # Store basic parameters
        self.img_size: int = img_size
        self.n_colors: int = n_colors
        self.latent_dim: int = latent_dim
        self.condition_dim: int = condition_dim
        self.is_hypernetwork: bool = is_hypernetwork
        self.hidden_dims: List[int] = (
            [32, 64, 128, 256] if hidden_dims is None else hidden_dims
        )

        # Calculate flattened dimension size
        self.flatten_dim: int = (
            self.hidden_dims[-1] * (img_size // (2 ** len(self.hidden_dims))) ** 2
        )

        # HyperConv layers storage for condition passing if needed
        if is_hypernetwork:
            self.hyper_conv_encoder_layers: List[Any] = []
            self.hyper_conv_layers: List[Any] = []

        # Build encoder and decoder
        self._build_encoder()
        self._build_latent_layers()
        self._build_decoder()

    def _build_encoder(self) -> None:
        """Build the encoder part of the network"""
        modules: List[nn.Module] = []
        in_channels: int = 1  # Single channel for integer inputs

        for h_dim in self.hidden_dims:
            if self.is_hypernetwork:
                self._add_hyper_conv_encoder_layer(modules, in_channels, h_dim)
            else:
                self._add_standard_conv_layer(modules, in_channels, h_dim)
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def _add_hyper_conv_encoder_layer(
        self, modules: List[nn.Module], in_channels: int, h_dim: int
    ) -> None:
        """Add a hypernetwork convolutional layer to the encoder"""
        output_size = self.img_size // (2 * (2 ** len(modules)))
        conv_layer = HyperConv2d(
            in_channels,
            h_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            condition_dim=self.condition_dim,
            condition_dims=[512],
        )
        self.hyper_conv_encoder_layers.append(conv_layer)
        modules.append(
            nn.Sequential(
                conv_layer,
                nn.LayerNorm([h_dim, output_size, output_size]),
                nn.GELU(),
            )
        )

    def _add_standard_conv_layer(
        self, modules: List[nn.Module], in_channels: int, h_dim: int
    ) -> None:
        """Add a standard convolutional layer to the encoder"""
        output_size = self.img_size // (2 * (2 ** len(modules)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.LayerNorm([h_dim, output_size, output_size]),
                nn.GELU(),
            )
        )

    def _build_latent_layers(self) -> None:
        """Build the layers for latent space processing"""
        # Linear layers for latent space
        self.condition_encoder = nn.Linear(self.condition_dim, 32)
        self.fc_mu = nn.Linear(self.flatten_dim + 32, self.latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim + 32, self.latent_dim)

        # Linear layer from latent space to features
        if self.is_hypernetwork:
            self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim)
        else:
            self.decoder_input = nn.Linear(self.latent_dim + 32, self.flatten_dim)

        # Unflatten layer
        self.unflatten = nn.Unflatten(
            1,
            (
                self.hidden_dims[-1],
                self.img_size // (2 ** len(self.hidden_dims)),
                self.img_size // (2 ** len(self.hidden_dims)),
            ),
        )

    def _build_decoder(self) -> None:
        """Build the decoder part of the network"""
        modules: List[nn.Sequential] = []

        # Reverse hidden dimensions for decoder
        reversed_hidden_dims = self.hidden_dims.copy()
        reversed_hidden_dims.reverse()

        # Build transposed convolution layers
        for i in range(len(reversed_hidden_dims) - 1):
            output_size = self.img_size // (2 ** (len(reversed_hidden_dims) - i - 1))
            if self.is_hypernetwork:
                self._add_hyper_conv_transpose_layer(
                    modules,
                    reversed_hidden_dims[i],
                    reversed_hidden_dims[i + 1],
                    output_size,
                )
            else:
                self._add_standard_conv_transpose_layer(
                    modules,
                    reversed_hidden_dims[i],
                    reversed_hidden_dims[i + 1],
                    output_size,
                )

        self.decoder = nn.Sequential(*modules)

        # Build final layer
        self._build_final_layer(reversed_hidden_dims[-1])

    def _add_hyper_conv_transpose_layer(
        self,
        modules: List[nn.Sequential],
        in_channels: int,
        out_channels: int,
        output_size: int,
    ) -> None:
        """Add a hypernetwork transposed convolutional layer to the decoder"""
        conv_layer = HyperConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            condition_dim=self.condition_dim,
            condition_dims=[512],
        )
        self.hyper_conv_layers.append(conv_layer)
        modules.append(
            nn.Sequential(
                conv_layer,
                nn.LayerNorm([out_channels, output_size, output_size]),
                nn.GELU(),
            )
        )

    def _add_standard_conv_transpose_layer(
        self,
        modules: List[nn.Sequential],
        in_channels: int,
        out_channels: int,
        output_size: int,
    ) -> None:
        """Add a standard transposed convolutional layer to the decoder"""
        conv_layer = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        modules.append(
            nn.Sequential(
                conv_layer,
                nn.LayerNorm([out_channels, output_size, output_size]),
                nn.GELU(),
            )
        )

    def _build_final_layer(self, hidden_dim: int) -> None:
        """Build the final output layer of the network"""
        if self.is_hypernetwork:
            final_conv_transpose = HyperConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                condition_dim=self.condition_dim,
            )
            self.hyper_conv_layers.append(final_conv_transpose)

            # Final layer with regular Conv2d
            final_conv = nn.Conv2d(hidden_dim, self.n_colors, kernel_size=3, padding=1)

            self.final_layer = nn.Sequential(
                final_conv_transpose,
                nn.LayerNorm([hidden_dim, self.img_size, self.img_size]),
                nn.GELU(),
                final_conv,
            )
        else:
            final_conv_transpose = nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )

            final_conv = nn.Conv2d(hidden_dim, self.n_colors, kernel_size=3, padding=1)

            self.final_layer = nn.Sequential(
                final_conv_transpose,
                nn.LayerNorm([hidden_dim, self.img_size, self.img_size]),
                nn.GELU(),
                final_conv,
            )

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation z and condition c into an image.

        Args:
            z: Latent vector of shape [B, latent_dim]
            c: Condition tensor of shape [B, condition_dim]

        Returns:
            Reconstructed image logits of shape [B, n_colors, H, W]
        """
        if self.is_hypernetwork:
            # Use latent vector directly without concatenating with condition
            x = self.decoder_input(z)
            x = self.unflatten(x)

            # Process through decoder except for HyperConvTranspose2d layers
            for module in self.decoder:
                # For regular layers
                for layer in module.children():
                    if isinstance(layer, HyperConvTranspose2d):
                        x = layer(x, condition=c)
                    else:
                        x = layer(x)
            # Apply final layer manually
            for layer in self.final_layer.children():
                if isinstance(layer, HyperConvTranspose2d):
                    x = layer(x, condition=c)
                else:
                    x = layer(x)
        else:
            # Encode condition
            c_encoded = self.condition_encoder(c)

            # Combine latent with condition
            z_c = torch.cat([z, c_encoded], dim=1)

            # Decode
            x = self.decoder_input(z_c)
            x = self.unflatten(x)
            x = self.decoder(x)
            x = self.final_layer(x)

        return x

    def encode(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input image x and condition c into latent space.

        Args:
            x: Input tensor of shape [B, 1, H, W]
            c: Condition tensor of shape [B, condition_dim]

        Returns:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        # Encode image
        if self.is_hypernetwork:
            # Process through encoder manually for HyperConv2d layers
            encoded_x = x  # Start with input tensor
            for module in self.encoder:
                for layer in module.children():
                    if isinstance(layer, HyperConv2d):
                        encoded_x = layer(encoded_x, condition=c)
                    else:
                        encoded_x = layer(encoded_x)
            x = encoded_x
        else:
            x = self.encoder(x)

        x = torch.flatten(x, start_dim=1)

        # Encode condition
        c_encoded = self.condition_encoder(c)

        # Combine image features with condition
        x_c = torch.cat([x, c_encoded], dim=1)

        # Get latent distribution parameters
        mu = self.fc_mu(x_c)
        log_var = self.fc_var(x_c)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.

        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CVAE.

        Args:
            x: Input tensor of shape [B, 1, H, W]
            c: Condition tensor of shape [B, condition_dim]

        Returns:
            x_recon: Reconstructed image logits
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)

        return x_recon, mu, log_var

    def transform_image(
        self,
        x: torch.Tensor,
        c_in: torch.Tensor = torch.tensor([[1.0]]),
        c_out: torch.Tensor = torch.tensor([[0.0]]),
    ) -> torch.Tensor:
        """
        Encode an image with one condition (is_input=True) and decode with
        another condition (is_input=False).

        Args:
            x: Input tensor of shape [B, 1, H, W]
            c_in: Input condition (default: is_input=True)
            c_out: Output condition (default: is_input=False)

        Returns:
            Transformed image logits
        """
        # Move conditions to the same device as the input
        c_in = c_in.to(x.device)
        c_out = c_out.to(x.device)

        # Encode with input condition
        with torch.no_grad():
            mu, log_var = self.encode(x, c_in)
            z = self.reparameterize(mu, log_var)

            # Decode with output condition
            transformed_image = self.decode(z, c_out)

        return transformed_image

    def loss_function(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kld_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CVAE loss function = Reconstruction loss + KL divergence loss.

        Args:
            x_recon: Reconstructed image logits
            x: Target image
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            kld_weight: Weight for the KL divergence term

        Returns:
            Total loss, reconstruction loss, and KL divergence loss
        """
        # For integer inputs, we use cross entropy loss
        # Reshape for cross entropy: [B, C, H, W] -> [B, C, H*W]
        B, C = x_recon.size(0), x_recon.size(1)
        x_recon_flat = x_recon.view(B, C, -1)

        # Target should be [B, H*W] with integer values
        x_flat = x.view(B, -1).long()

        # Cross entropy loss
        recon_loss = F.cross_entropy(x_recon_flat, x_flat, reduction="sum")

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        loss = recon_loss + kld_weight * kld_loss

        return loss, recon_loss, kld_loss


def create_cvae(
    img_size: int = 30,
    n_colors: int = 11,
    latent_dim: int = 64,
    is_hypernetwork: bool = False,
) -> CVAE:
    """
    Creates and initializes a CVAE model.

    Args:
        img_size: Size of the square images
        n_colors: Number of possible color values
        latent_dim: Dimension of the latent space
        is_hypernetwork: If True, use condition to generate conv layer weights

    Returns:
        Initialized CVAE model
    """
    model = CVAE(
        img_size=img_size,
        n_colors=n_colors,
        latent_dim=latent_dim,
        is_hypernetwork=is_hypernetwork,
    )
    return model


def preprocess_int_array(int_array: torch.Tensor) -> torch.Tensor:
    """
    Converts a 2D integer array to a tensor suitable for the CVAE.

    Args:
        int_array: Tensor or array with integer values

    Returns:
        Preprocessed tensor with shape [B, 1, H, W]
    """
    # Convert to tensor if it's not already
    if not torch.is_tensor(int_array):
        int_array = torch.tensor(int_array, dtype=torch.float32)

    # Ensure 2D array is shaped [Batch, 1, Height, Width]
    if len(int_array.shape) == 2:
        # Single image: [H, W] -> [1, 1, H, W]
        return int_array.unsqueeze(0).unsqueeze(0)
    elif len(int_array.shape) == 3:
        # Batch of images: [B, H, W] -> [B, 1, H, W]
        return int_array.unsqueeze(1)
    else:
        # Already in correct format
        return int_array


def postprocess_to_ints(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts CVAE logits output back to integers.

    Args:
        logits: Model output logits of shape [B, n_colors, H, W]

    Returns:
        Tensor with integer values of shape [B, H, W]
    """
    # Get the most likely class for each pixel
    # Input shape: [B, n_colors, H, W]
    # Output shape: [B, H, W] with integer values
    probs = F.softmax(logits, dim=1)
    _, indices = torch.max(probs, dim=1)
    return indices
