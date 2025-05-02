import pytest
import torch
import torch.nn as nn
from arcagi.model import CVAE


def test__build_latent_layers__should_match_expected_size():
    """Test that the latent layers are built with the expected sizes."""
    # Arrange
    img_size: int = 30
    n_colors: int = 11
    latent_dim: int = 64
    condition_dim: int = 1
    hidden_dims: list[int] = [32, 64, 128, 256]

    # Act
    model = CVAE(
        img_size=img_size,
        n_colors=n_colors,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
        is_hypernetwork=False,
    )

    # Assert
    # Check condition encoder dimensions
    assert isinstance(model.condition_encoder, nn.Linear)
    assert model.condition_encoder.in_features == condition_dim
    assert model.condition_encoder.out_features == 32

    # Note: We no longer check fc_mu, fc_var, decoder_input and unflatten dimensions
    # directly as they are created dynamically during forward pass

    # Run a forward pass to dynamically create the layers
    test_input = torch.zeros(1, n_colors, img_size, img_size)
    condition = torch.zeros(1, condition_dim)
    with torch.no_grad():
        reconstructed, mu, log_var = model(test_input, condition)

    # Check that the forward pass produced outputs with expected shapes
    assert mu.shape == (1, latent_dim)
    assert log_var.shape == (1, latent_dim)
    assert reconstructed.shape == (1, n_colors, img_size, img_size)


def test__build_latent_layers__hypernetwork_should_match_expected_size():
    """Test that the latent layers are built with the expected sizes when using hypernetwork."""
    # Arrange
    img_size: int = 16  # Reduce size to save memory
    n_colors: int = 11
    latent_dim: int = 32  # Reduce dimensions
    condition_dim: int = 1
    hidden_dims: list[int] = [16, 32, 64]  # Fewer hidden layers with smaller dimensions

    # Act
    model = CVAE(
        img_size=img_size,
        n_colors=n_colors,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
        is_hypernetwork=True,
    )

    # Assert
    # Check condition encoder dimensions
    assert isinstance(model.condition_encoder, nn.Linear)
    assert model.condition_encoder.in_features == condition_dim
    assert model.condition_encoder.out_features == 32

    # Note: We no longer check fc_mu, fc_var, decoder_input and unflatten dimensions
    # directly as they are created dynamically during forward pass

    # Run a forward pass to dynamically create the layers
    test_input = torch.zeros(1, n_colors, img_size, img_size)
    condition = torch.zeros(1, condition_dim)
    with torch.no_grad():
        reconstructed, mu, log_var = model(test_input, condition)

    # Check that the forward pass produced outputs with expected shapes
    assert mu.shape == (1, latent_dim)
    assert log_var.shape == (1, latent_dim)
    assert reconstructed.shape == (1, n_colors, img_size, img_size)


def test_channel_mismatch_error():
    """Test that the model handles one-hot encoded inputs correctly."""
    # Arrange
    img_size = 30
    n_colors = 11
    latent_dim = 64
    condition_dim = 1
    hidden_dims = [32, 64, 128, 256]

    model = CVAE(
        img_size=img_size,
        n_colors=n_colors,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
        is_hypernetwork=False,
    )

    # Create test input - simulating one-hot encoded input
    # Shape: [batch_size, n_colors, height, width]
    one_hot_input = torch.zeros(1, n_colors, 17, 30, dtype=torch.float32)

    # Create condition tensor
    condition = torch.tensor([[0.5]], dtype=torch.float32)

    # This should pass without raising an error
    reconstructed, mu, log_var = model(one_hot_input, condition)

    # Basic assertions to verify output shapes
    assert reconstructed.shape == (1, n_colors, img_size, img_size)
    assert mu.shape == (1, latent_dim)
    assert log_var.shape == (1, latent_dim)


@pytest.mark.parametrize("is_hypernetwork", [False, True])
def test__forward__matches_expected_shape(is_hypernetwork: bool):
    """Test that the model forward pass produces outputs with expected shapes."""
    # Arrange
    img_size = 30
    n_colors = 11
    latent_dim = 64
    condition_dim = 1
    batch_size = 2

    # Initialize the model with the specified hypernetwork setting
    model = CVAE(
        img_size=img_size,
        n_colors=n_colors,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        is_hypernetwork=is_hypernetwork,
    )

    # Create input tensor with expected shape
    input_tensor: torch.Tensor = torch.rand(batch_size, n_colors, img_size, img_size)

    # Create condition tensor
    condition: torch.Tensor = torch.rand(batch_size, condition_dim)

    # Act - perform forward pass
    reconstructed, mu, log_var = model(input_tensor, condition)

    # Assert - check output shapes
    assert reconstructed.shape == (
        batch_size,
        n_colors,
        img_size,
        img_size,
    ), f"Expected reconstructed shape {(batch_size, n_colors, img_size, img_size)}, got {reconstructed.shape}"
    assert mu.shape == (
        batch_size,
        latent_dim,
    ), f"Expected mu shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert log_var.shape == (
        batch_size,
        latent_dim,
    ), f"Expected log_var shape {(batch_size, latent_dim)}, got {log_var.shape}"
