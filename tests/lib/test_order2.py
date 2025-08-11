"""
Tests for the Order2Features PyTorch module.

These tests verify that the PyTorch implementation matches the behavior
of the original numpy implementation in preprocess.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from arcagi.preprocess import compute_order2_features
from lib.order2 import Order2Features, create_order2_transform


class TestOrder2Features:
    """Test suite for Order2Features module."""

    @pytest.fixture
    def transform(self):
        """Create an Order2Features transform instance."""
        return Order2Features()

    def test_output_shape(self, transform):
        """Test that output has correct shape."""
        # Test various input sizes
        test_cases = [
            (1, 30, 30),  # Standard ARC size
            (2, 30, 30),  # Batch of 2
            (1, 5, 5),  # Smaller grid
            (3, 10, 15),  # Non-square grid
        ]

        for B, H, W in test_cases:
            x = torch.randint(0, 10, (B, H, W))
            output = transform(x)
            assert output.shape == (B, H, W, 45), f"Wrong shape for input {(B, H, W)}"

    def test_binary_output(self, transform):
        """Test that all output values are binary (0 or 1)."""
        x = torch.randint(-1, 10, (2, 10, 10))
        output = transform(x)

        # Check all values are 0 or 1
        assert torch.all(
            (output == 0) | (output == 1)
        ), "Output contains non-binary values"

    def test_matches_numpy_implementation(self, transform):
        """Test that PyTorch implementation matches the numpy version."""
        # Create several test cases
        test_grids = [
            # Uniform grid (all same color)
            np.ones((30, 30), dtype=np.int32) * 3,
            # Checkerboard pattern
            np.array([[i % 2 for i in range(30)] for j in range(30)], dtype=np.int32),
            # Random grid with masks
            np.random.randint(-1, 10, size=(30, 30), dtype=np.int32),
            # Small test grid
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, 0], [1, 2, 3, 4]],
                dtype=np.int32,
            ),
        ]

        for grid_np in test_grids:
            # Compute features with numpy implementation
            features_np = compute_order2_features(grid_np)

            # Compute features with PyTorch implementation
            grid_torch = torch.from_numpy(grid_np).unsqueeze(0)  # Add batch dimension
            features_torch_full = (
                transform(grid_torch).squeeze(0).numpy()
            )  # Remove batch dimension

            # Compare only the first 44 features to the numpy implementation
            np.testing.assert_array_equal(
                features_torch_full[..., :44],
                features_np,
                err_msg=f"Mismatch for grid shape {grid_np.shape}",
            )

    def test_pairwise_features(self, transform):
        """Test pairwise color matching features specifically."""
        # Create a simple 3x3 grid where we know the expected output
        x = torch.tensor([[[0, 0, 1], [0, 0, 1], [2, 2, 1]]], dtype=torch.long)

        output = transform(x)

        # Check center cell (1,1) which has neighborhood:
        # [0, 0, 1]
        # [0, 0, 1]
        # [2, 2, 1]
        center_features = output[0, 1, 1, :36]  # First 36 are pairwise

        # We expect many 0s (same color) and some 1s (different colors)
        # For example, positions 0,1 have same color (0), so feature should be 0
        # Positions 0,2 have different colors (0,1), so feature should be 1

        # Count number of same-color pairs (should be > 0)
        num_same = (center_features == 0).sum()
        assert num_same > 0, "Should have some same-color pairs"

        # Count number of different-color pairs (should be > 0)
        num_diff = (center_features == 1).sum()
        assert num_diff > 0, "Should have some different-color pairs"

    def test_mask_features(self, transform):
        """Test mask detection features specifically."""
        # Create grid with mask values
        x = torch.tensor([[[-1, 0, 1], [2, 3, 4], [5, 6, -1]]], dtype=torch.long)

        output = transform(x)

        # Check center cell (1,1) mask features (positions 36:44)
        center_mask_features = output[0, 1, 1, 36:44]

        # The 8 neighbors of center are: [-1, 0, 1, 2, 4, 5, 6, -1]
        # Expected mask features: [0, 1, 1, 1, 1, 1, 1, 0]
        # (0 for positions with -1, 1 for positions without -1)
        expected = torch.tensor([0, 1, 1, 1, 1, 1, 1, 0], dtype=torch.float32)
        torch.testing.assert_close(center_mask_features, expected)

        # is_mask (last feature) should be 0 for center cell (not a mask)
        center_is_mask = output[0, 1, 1, 44]
        assert center_is_mask.item() == 0

    def test_boundary_handling(self, transform):
        """Test that boundaries are handled correctly (padded with -1)."""
        # Small grid to easily check boundaries
        x = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)

        output = transform(x)

        # Check top-left corner (0,0)
        # Its 3x3 neighborhood should have -1s for out-of-bounds positions
        corner_mask_features = output[0, 0, 0, 36:44]

        # The corner has only 3 real neighbors, 5 should be out of bounds (-1)
        # So we expect 3 ones and 5 zeros in mask features
        num_valid_neighbors = corner_mask_features.sum()
        assert (
            num_valid_neighbors == 3
        ), f"Corner should have 3 valid neighbors, got {num_valid_neighbors}"

    def test_batch_processing(self, transform):
        """Test that batch processing works correctly."""
        # Create batch of different grids
        batch_size = 4
        x = torch.stack(
            [torch.ones(10, 10) * i for i in range(batch_size)], dim=0
        ).long()

        output = transform(x)

        # Check shape
        assert output.shape == (batch_size, 10, 10, 45)

        # For uniform color grids, interior cells will have all 0s for pairwise features
        # But boundary cells will have some 1s due to padding with -1
        # Let's check interior cells only

        for i in range(batch_size):
            # Check a center cell (away from boundaries)
            center_pairwise = output[i, 5, 5, :36]
            assert torch.all(
                center_pairwise == 0
            ), f"Batch item {i} center cell should have all matching colors"

            # Check that different batches with different uniform colors
            # still have the same pattern (all zeros for center cells)
            if i > 0:
                assert torch.all(
                    output[0, 5, 5, :36] == output[i, 5, 5, :36]
                ), "Center cells should have same pattern for uniform grids"

    def test_device_compatibility(self, transform):
        """Test that the module works on different devices."""
        x = torch.randint(0, 10, (1, 5, 5))

        # Test on CPU
        transform_cpu = transform.cpu()
        x_cpu = x.cpu()
        output_cpu = transform_cpu(x_cpu)
        assert output_cpu.device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            transform_cuda = transform.cuda()
            x_cuda = x.cuda()
            output_cuda = transform_cuda(x_cuda)
            assert output_cuda.device.type == "cuda"

            # Results should be the same
            torch.testing.assert_close(output_cpu, output_cuda.cpu())

    def test_gradient_flow(self, transform):
        """Test that the module can be used in a computational graph."""
        # Note: The transform uses discrete operations (comparisons)
        # so gradients won't flow through it meaningfully
        x = torch.randint(0, 10, (1, 5, 5))

        # Forward pass should work
        output = transform(x)

        # We can compute a loss
        loss = output.sum()

        # Loss should be a scalar
        assert loss.shape == ()

    def test_deterministic(self, transform):
        """Test that the transform is deterministic."""
        x = torch.randint(-1, 10, (2, 15, 15))

        # Run multiple times
        outputs = [transform(x) for _ in range(5)]

        # All outputs should be identical
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i])

    def test_create_transform_function(self):
        """Test the create_order2_transform convenience function."""
        transform = create_order2_transform()
        assert isinstance(transform, Order2Features)

        # Test it works
        x = torch.randint(0, 10, (1, 5, 5))
        output = transform(x)
        assert output.shape == (1, 5, 5, 45)


class TestEdgeCases:
    """Test edge cases and special inputs."""

    @pytest.fixture
    def transform(self):
        return Order2Features()

    def test_all_mask_input(self, transform):
        """Test input that is all mask values."""
        x = torch.ones(1, 5, 5).long() * -1
        output = transform(x)

        # All pairwise features should be 0 (same color)
        pairwise = output[..., :36]
        assert torch.all(pairwise == 0), "All mask values should have matching colors"

        # All mask features should be 0 (all neighbors are masks)
        mask_features = output[..., 36:44]
        assert torch.all(
            mask_features == 0
        ), "All neighbors should be detected as masks"

        # is_mask feature (last feature) should be 1 everywhere
        is_mask = output[..., 44]
        assert torch.all(is_mask == 1), "Center cells should all be masks"

    def test_single_pixel(self, transform):
        """Test 1x1 input (all neighbors out of bounds)."""
        x = torch.tensor([[[5]]], dtype=torch.long)  # Need batch dimension
        output = transform(x)

        assert output.shape == (1, 1, 1, 45)

        # All neighbors are out of bounds (-1)
        # So all pairwise comparisons with center (5) should be 1 (different)
        # Except center compared with itself

        # All mask features should be 0 (all neighbors are -1)
        mask_features = output[0, 0, 0, 36:44]
        assert torch.all(mask_features == 0), "All neighbors should be out of bounds"

    def test_large_color_values(self, transform):
        """Test with color values at the upper limit (9)."""
        x = torch.randint(7, 10, (1, 5, 5))  # Colors 7, 8, 9
        output = transform(x)

        # Should work normally
        assert output.shape == (1, 5, 5, 45)
        assert torch.all((output == 0) | (output == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
