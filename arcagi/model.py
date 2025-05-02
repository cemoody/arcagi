import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from loguru import logger

# Import with explicit type
from arcagi.HyperConvTranspose2d import HyperConvTranspose2d


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
        # Use n_colors as input channels instead of hardcoding to 1
        in_channels: int = self.n_colors

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
        # Calculate correct output size based on img_size and how many times we've halved the dimensions
        current_dim = (
            self.img_size if len(modules) == 0 else self.img_size // (2 ** len(modules))
        )
        output_size = current_dim // 2  # This layer will halve the dimensions

        conv_layer = HyperConvTranspose2d(
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
        # Calculate correct output size based on img_size and how many times we've halved the dimensions
        current_dim = (
            self.img_size if len(modules) == 0 else self.img_size // (2 ** len(modules))
        )
        output_size = current_dim // 2  # This layer will halve the dimensions

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

        # We'll use these as placeholders but rebuild them dynamically in forward pass
        # based on actual input dimensions
        self.fc_mu = nn.Linear(1, self.latent_dim)  # Will be rebuilt dynamically
        self.fc_var = nn.Linear(1, self.latent_dim)  # Will be rebuilt dynamically

        # Linear layer from latent space to features
        if self.is_hypernetwork:
            self.decoder_input = nn.Linear(
                self.latent_dim, 1
            )  # Will be rebuilt dynamically
        else:
            self.decoder_input = nn.Linear(
                self.latent_dim + 32, 1
            )  # Will be rebuilt dynamically

        # Unflatten layer - will be set dynamically during forward pass
        self.unflatten = nn.Unflatten(1, (1, 1, 1))  # Placeholder, will be rebuilt

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
        logger.debug(
            f"Decode input latent shape: {z.shape}, condition shape: {c.shape}"
        )

        if self.is_hypernetwork:
            # Use latent vector directly without concatenating with condition
            # Get last encoder output shape to determine flatten_dim
            with torch.no_grad():
                # Create a dummy input with batch size 1
                dummy_input = torch.zeros(
                    1, self.n_colors, self.img_size, self.img_size, device=z.device
                )
                dummy_c = torch.zeros(1, self.condition_dim, device=z.device)

                # Run through encoder to get the output dimensions
                x_dummy = dummy_input
                for module in self.encoder:
                    for layer in module.children():
                        if isinstance(layer, HyperConvTranspose2d):
                            x_dummy = layer(x_dummy, condition=dummy_c)
                        elif isinstance(layer, nn.LayerNorm):
                            # Skip pre-built LayerNorm to avoid shape issues
                            # We'll apply normalization directly with correct shapes
                            continue
                        else:
                            x_dummy = layer(x_dummy)

                # Calculate flatten_dim from encoder output
                flatten_dim = torch.flatten(x_dummy, start_dim=1).shape[1]

                # Create dynamic decoder input layer
                decoder_input = nn.Linear(self.latent_dim, flatten_dim).to(z.device)
                if (
                    hasattr(self, "decoder_input")
                    and self.decoder_input.out_features == flatten_dim
                ):
                    # Copy weights if possible
                    decoder_input.weight.data = self.decoder_input.weight.data
                    decoder_input.bias.data = self.decoder_input.bias.data
                self.decoder_input = decoder_input

                # Create dynamic unflatten layer
                self.unflatten = nn.Unflatten(
                    1,
                    (
                        x_dummy.shape[1],
                        x_dummy.shape[2],
                        x_dummy.shape[3],
                    ),
                )

            # Now decode
            x = self.decoder_input(z)
            logger.debug(f"Decoder input after linear layer: {x.shape}")

            x = self.unflatten(x)
            logger.debug(f"After unflatten: {x.shape}")

            # Process through decoder except for HyperConvTranspose2d layers
            for i, module in enumerate(self.decoder):
                # For each module, process layers individually
                layer_list = list(module.children())

                # Process the conv layer first
                if isinstance(layer_list[0], HyperConvTranspose2d):
                    x = layer_list[0](x, condition=c)
                else:
                    x = layer_list[0](x)

                # Get current dimensions and apply LayerNorm with correct shape
                _, curr_channels, curr_height, curr_width = x.shape
                layer_norm = nn.LayerNorm([curr_channels, curr_height, curr_width]).to(
                    x.device
                )
                x = layer_norm(x)

                # Process remaining layers (skip original LayerNorm)
                for layer in layer_list[2:]:
                    x = layer(x)

                logger.debug(f"Decoder module {i} output shape: {x.shape}")

            # Apply final layer manually with dynamic LayerNorm
            final_layers = list(self.final_layer.children())

            # Process first conv layer
            if isinstance(final_layers[0], HyperConvTranspose2d):
                x = final_layers[0](x, condition=c)
            else:
                x = final_layers[0](x)

            # Apply dynamic LayerNorm
            _, curr_channels, curr_height, curr_width = x.shape
            layer_norm = nn.LayerNorm([curr_channels, curr_height, curr_width]).to(
                x.device
            )
            x = layer_norm(x)

            # Process remaining layers
            for layer in final_layers[2:]:
                x = layer(x)

            logger.debug(f"Final layer output shape: {x.shape}")
        else:
            # Encode condition
            c_encoded = self.condition_encoder(c)
            logger.debug(f"Encoded condition shape: {c_encoded.shape}")

            # Combine latent with condition
            z_c = torch.cat([z, c_encoded], dim=1)
            logger.debug(f"Combined latent+condition shape: {z_c.shape}")

            # Get last encoder output shape to determine flatten_dim
            with torch.no_grad():
                # Create a dummy input with batch size 1
                dummy_input = torch.zeros(
                    1, self.n_colors, self.img_size, self.img_size, device=z.device
                )

                # Run through encoder to get the output dimensions
                x_dummy = dummy_input
                for module in self.encoder:
                    for layer in module.children():
                        if isinstance(layer, nn.LayerNorm):
                            # Skip pre-built LayerNorm to avoid shape issues
                            continue
                        else:
                            x_dummy = layer(x_dummy)

                # Calculate flatten_dim from encoder output
                flatten_dim = torch.flatten(x_dummy, start_dim=1).shape[1]

                # Create dynamic decoder input layer
                decoder_input = nn.Linear(self.latent_dim + 32, flatten_dim).to(
                    z.device
                )
                if (
                    hasattr(self, "decoder_input")
                    and self.decoder_input.out_features == flatten_dim
                ):
                    # Copy weights if possible
                    decoder_input.weight.data = self.decoder_input.weight.data
                    decoder_input.bias.data = self.decoder_input.bias.data
                self.decoder_input = decoder_input

                # Create dynamic unflatten layer
                self.unflatten = nn.Unflatten(
                    1,
                    (
                        x_dummy.shape[1],
                        x_dummy.shape[2],
                        x_dummy.shape[3],
                    ),
                )

            # Decode
            x = self.decoder_input(z_c)
            logger.debug(f"After decoder input linear: {x.shape}")

            x = self.unflatten(x)
            logger.debug(f"After unflatten: {x.shape}")

            # Process through decoder with dynamic LayerNorm
            for i, module_idx in enumerate(range(len(self.decoder))):
                module = self.decoder[module_idx]

                # For each module, process layers individually
                layer_list = list(module.children())

                # Process the conv layer first
                x = layer_list[0](x)

                # Get current dimensions and apply LayerNorm with correct shape
                _, curr_channels, curr_height, curr_width = x.shape
                layer_norm = nn.LayerNorm([curr_channels, curr_height, curr_width]).to(
                    x.device
                )
                x = layer_norm(x)

                # Process remaining layers (skip original LayerNorm)
                for layer in layer_list[2:]:
                    x = layer(x)

                logger.debug(f"Decoder module {i} output shape: {x.shape}")

            # Apply final layer with dynamic LayerNorm
            final_layers = list(self.final_layer.children())

            # Process first conv layer
            x = final_layers[0](x)

            # Apply dynamic LayerNorm
            _, curr_channels, curr_height, curr_width = x.shape
            layer_norm = nn.LayerNorm([curr_channels, curr_height, curr_width]).to(
                x.device
            )
            x = layer_norm(x)

            # Process remaining layers
            for layer in final_layers[2:]:
                x = layer(x)

            logger.debug(f"Final output shape: {x.shape}")

        # Ensure output size is exactly img_size x img_size
        # This handles any size mismatches from the transposed convolutions
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            # Use resize to ensure exact output size
            x = F.interpolate(  # type: ignore
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        return x  # type: ignore

    def encode(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input image x and condition c into latent space.

        Args:
            x: Input tensor of shape [B, channels, H, W]
            c: Condition tensor of shape [B, condition_dim]

        Returns:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        logger.debug(f"Encode input shape: {x.shape}, condition shape: {c.shape}")

        # Encode image
        if self.is_hypernetwork:
            # Process through encoder manually for HyperConvTranspose2d layers
            encoded_x = x  # Start with input tensor
            for i, module in enumerate(self.encoder):
                # For the LayerNorm layers, we need to adjust the expected dimensions
                layer_list = list(module.children())

                # Process the conv layer first
                if isinstance(layer_list[0], HyperConvTranspose2d):
                    encoded_x = layer_list[0](encoded_x, condition=c)
                else:
                    encoded_x = layer_list[0](encoded_x)

                # Get the current output dimensions
                _, curr_channels, curr_height, curr_width = encoded_x.shape

                # Create a new LayerNorm with the correct dimensions
                layer_norm = nn.LayerNorm([curr_channels, curr_height, curr_width]).to(
                    encoded_x.device
                )

                # Process through the new layer_norm and remaining layers
                encoded_x = layer_norm(encoded_x)
                for layer in layer_list[2:]:  # Skip the original LayerNorm
                    encoded_x = layer(encoded_x)

                logger.debug(f"Encoder layer {i} output shape: {encoded_x.shape}")
            x = encoded_x
        else:
            for i, module_idx in enumerate(range(len(self.encoder))):
                module = self.encoder[module_idx]

                # For the LayerNorm layers, we need to adjust the expected dimensions
                layer_list = list(module.children())

                # Process the conv layer first
                x = layer_list[0](x)

                # Get the current output dimensions
                _, curr_channels, curr_height, curr_width = x.shape

                # Create a new LayerNorm with the correct dimensions
                layer_norm = nn.LayerNorm([curr_channels, curr_height, curr_width]).to(
                    x.device
                )

                # Process through the new layer_norm and remaining layers
                x = layer_norm(x)
                for layer in layer_list[2:]:  # Skip the original LayerNorm
                    x = layer(x)

                logger.debug(f"Encoder module {i} output shape: {x.shape}")

        x = torch.flatten(x, start_dim=1)
        logger.debug(f"Flattened encoder output shape: {x.shape}")

        # Encode condition
        c_encoded = self.condition_encoder(c)
        logger.debug(f"Encoded condition shape: {c_encoded.shape}")

        # Combine image features with condition
        x_c = torch.cat([x, c_encoded], dim=1)
        logger.debug(f"Combined features shape: {x_c.shape}")

        # Get actual flatten dimension
        flatten_dim = x_c.shape[1]

        # Create dynamic mu and var encoders
        fc_mu = nn.Linear(flatten_dim, self.latent_dim).to(x.device)
        fc_var = nn.Linear(flatten_dim, self.latent_dim).to(x.device)

        # Initialize with same weights if possible (for consistency with pretrained models)
        if self.fc_mu.in_features == flatten_dim:
            fc_mu.weight.data = self.fc_mu.weight.data
            fc_mu.bias.data = self.fc_mu.bias.data
            fc_var.weight.data = self.fc_var.weight.data
            fc_var.bias.data = self.fc_var.bias.data

        # Update model's layers for future use
        self.fc_mu = fc_mu
        self.fc_var = fc_var

        # Get latent distribution parameters
        mu = self.fc_mu(x_c)
        log_var = self.fc_var(x_c)
        logger.debug(f"Latent mu shape: {mu.shape}, log_var shape: {log_var.shape}")

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
        logger.debug(f"Reparameterized latent shape: {z.shape}")
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
        logger.debug(f"Forward input shape: {x.shape}, condition shape: {c.shape}")

        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)

        logger.debug(f"Forward output shape: {x_recon.shape}")
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
    original_shape = int_array.shape
    if len(int_array.shape) == 2:
        # Single image: [H, W] -> [1, 1, H, W]
        result = int_array.unsqueeze(0).unsqueeze(0)
    elif len(int_array.shape) == 3:
        # Batch of images: [B, H, W] -> [B, 1, H, W]
        result = int_array.unsqueeze(1)
    else:
        # Already in correct format
        result = int_array

    logger.debug(
        f"Preprocessed int array from shape {original_shape} to {result.shape}"
    )
    return result


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
    logger.debug(f"Postprocessed from shape {logits.shape} to {indices.shape}")
    return indices


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    # Configure logger to show debug messages
    logger.remove()
    logger.add(lambda msg: print(msg), level="DEBUG")

    logger.info("Starting model test with 30x30 input")

    # Initialize the model
    model = CVAE(
        img_size=30,
        n_colors=11,
        latent_dim=64,
        condition_dim=1,
        hidden_dims=None,
        is_hypernetwork=False,
    )

    # Sample data from the context
    # Extract input and output colors from the sample data
    input_colors = torch.ones(1, 17, 30, dtype=torch.int32)
    logger.debug(f"Original input shape: {input_colors.shape}")

    # Create a condition tensor (can be any value for testing)
    condition: torch.Tensor = torch.tensor([[0.5]])

    # Preprocess the input
    processed_input = preprocess_int_array(input_colors)

    # Create one-hot encoding for the input
    # Replace -1 values with 0 for one-hot encoding
    valid_input = torch.where(
        processed_input == -1,
        torch.tensor(0, dtype=processed_input.dtype),
        processed_input,
    )

    # Convert to one-hot encoding
    one_hot_input = F.one_hot(valid_input.long(), num_classes=11).float()
    one_hot_input = one_hot_input.squeeze(1).permute(0, 3, 1, 2)
    logger.debug(f"One-hot encoded input shape: {one_hot_input.shape}")

    # Run a forward pass
    logger.info("Running forward pass")
    reconstructed, mu, log_var = model(one_hot_input, condition)

    # Convert output back to integers
    output_ints = postprocess_to_ints(reconstructed)

    logger.info("Model initialized and forward pass completed")
    logger.info(f"Input shape: {input_colors.shape}")
    logger.info(f"Output shape: {output_ints.shape}")
    logger.info(f"Mean vector shape: {mu.shape}")
    logger.info(f"Log variance shape: {log_var.shape}")
