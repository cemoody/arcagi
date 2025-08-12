"""
Test suite for D4 augmentation dataset using pytest.

This tests the D4AugmentedDataset class to ensure:
1. Transformations are applied correctly
2. Same transformation is applied to input/output pairs
3. All 8 D4 transformations work as expected
4. Features and indices are handled properly
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from arcagi.d4_augmentation_dataset import (
    D4AugmentedDataset, 
    D4CollateFunction,
    create_d4_augmented_dataloader
)


@pytest.fixture
def simple_data():
    """Create simple test data."""
    n_samples = 5
    inputs = torch.randn(n_samples, 4, 4, 2)
    outputs = torch.randn(n_samples, 4, 4, 2)
    return inputs, outputs


@pytest.fixture
def test_pattern():
    """Create a distinctive test pattern to verify transformations."""
    # Create a 4x4 pattern that's easy to track through transformations
    pattern = torch.zeros(4, 4)
    # Create an L-shape pattern
    pattern[0, :] = torch.tensor([1, 2, 3, 4])  # Top row
    pattern[:, 0] = torch.tensor([1, 5, 6, 7])  # Left column
    return pattern


@pytest.fixture
def full_data():
    """Create full test data with features and indices."""
    n_samples = 4
    inputs = torch.randn(n_samples, 4, 4, 2)
    outputs = torch.randn(n_samples, 4, 4, 2)
    input_features = torch.randn(n_samples, 4, 4, 3)
    output_features = torch.randn(n_samples, 4, 4, 3)
    indices = torch.arange(n_samples)
    return inputs, outputs, input_features, output_features, indices


class TestD4AugmentedDataset:
    """Test the D4AugmentedDataset class."""
    
    def test_basic_functionality(self, simple_data):
        """Test basic dataset creation and indexing."""
        inputs, outputs = simple_data
        n_samples = len(inputs)
        
        # Create dataset without augmentation
        dataset = D4AugmentedDataset(inputs, outputs, augment=False)
        
        assert len(dataset) == n_samples
        
        # Test indexing
        sample = dataset[0]
        assert len(sample) == 2
        assert sample[0].shape == inputs[0].shape
        assert sample[1].shape == outputs[0].shape
        assert torch.allclose(sample[0], inputs[0])
        assert torch.allclose(sample[1], outputs[0])
    
    def test_deterministic_augmentation(self, simple_data):
        """Test that deterministic augmentation creates 8x samples."""
        inputs, outputs = simple_data
        n_samples = len(inputs)
        
        # Create dataset with deterministic augmentation
        dataset = D4AugmentedDataset(
            inputs, outputs, 
            augment=True, 
            deterministic_augmentation=True
        )
        
        assert len(dataset) == n_samples * 8
        
        # Check that we can access all augmented samples
        for i in range(len(dataset)):
            sample = dataset[i]
            assert len(sample) == 2
    
    def test_transformation_correctness(self, test_pattern):
        """Test that D4 transformations are applied correctly."""
        # Create inputs with the test pattern
        inputs = test_pattern.unsqueeze(0).unsqueeze(-1)  # Shape: (1, 4, 4, 1)
        outputs = test_pattern.rot90(1).unsqueeze(0).unsqueeze(-1)  # Different pattern
        
        # Create dataset with deterministic augmentation
        dataset = D4AugmentedDataset(
            inputs, outputs,
            augment=True,
            deterministic_augmentation=True
        )
        
        # Test identity transformation (index 0)
        inp, out = dataset[0]
        assert torch.allclose(inp[:, :, 0], inputs[0, :, :, 0])
        
        # Test 90 degree rotation (index 1)
        inp, out = dataset[1]
        expected_rot90 = torch.rot90(inputs[0, :, :, 0], k=-1, dims=(0, 1))
        assert torch.allclose(inp[:, :, 0], expected_rot90)
        
        # Test 180 degree rotation (index 2)
        inp, out = dataset[2]
        expected_rot180 = torch.rot90(inputs[0, :, :, 0], k=2, dims=(0, 1))
        assert torch.allclose(inp[:, :, 0], expected_rot180)
        
        # Test horizontal flip (index 4)
        inp, out = dataset[4]
        expected_flip = torch.flip(inputs[0, :, :, 0], dims=[1])
        assert torch.allclose(inp[:, :, 0], expected_flip)
    
    def test_same_transform_applied(self):
        """Test that the same transformation is applied to input and output."""
        # Create patterns where we can verify the same transform was applied
        inputs = torch.zeros(1, 4, 4, 1)
        outputs = torch.zeros(1, 4, 4, 1)
        
        # Put a marker in top-left of input
        inputs[0, 0, 0, 0] = 1.0
        # Put a marker in bottom-right of output
        outputs[0, 3, 3, 0] = 1.0
        
        dataset = D4AugmentedDataset(
            inputs, outputs,
            augment=True,
            deterministic_augmentation=True
        )
        
        # Test a few transformations
        expected_positions = {
            0: ((0, 0), (3, 3)),  # Identity - top-left stays top-left, bottom-right stays bottom-right
            1: ((0, 3), (3, 0)),  # 90° clockwise rotation - top-left goes to top-right, bottom-right goes to bottom-left
            2: ((3, 3), (0, 0)),  # 180° rotation - top-left goes to bottom-right, bottom-right goes to top-left
            4: ((0, 3), (3, 0)),  # Horizontal flip - top-left goes to top-right, bottom-right goes to bottom-left
        }
        
        for transform_idx, (exp_inp, exp_out) in expected_positions.items():
            inp, out = dataset[transform_idx]
            
            # Find where the markers ended up
            inp_marker_pos = torch.nonzero(inp[:, :, 0] == 1.0)
            out_marker_pos = torch.nonzero(out[:, :, 0] == 1.0)
            
            assert len(inp_marker_pos) == 1, f"Transform {transform_idx}: Input marker not found uniquely"
            assert len(out_marker_pos) == 1, f"Transform {transform_idx}: Output marker not found uniquely"
            
            assert tuple(inp_marker_pos[0].tolist()) == exp_inp, \
                f"Transform {transform_idx}: Input marker at wrong position"
            assert tuple(out_marker_pos[0].tolist()) == exp_out, \
                f"Transform {transform_idx}: Output marker at wrong position"
    
    def test_with_features_and_indices(self, full_data):
        """Test dataset with features and indices."""
        inputs, outputs, input_features, output_features, indices = full_data
        
        dataset = D4AugmentedDataset(
            inputs, outputs,
            inputs_features=input_features,
            outputs_features=output_features,
            indices=indices,
            augment=True,
            deterministic_augmentation=False
        )
        
        # Test that all components are returned
        sample = dataset[0]
        assert len(sample) == 5
        
        inp, out, inp_feat, out_feat, idx = sample
        assert inp.shape == inputs[0].shape
        assert out.shape == outputs[0].shape
        assert inp_feat.shape == input_features[0].shape
        assert out_feat.shape == output_features[0].shape
        assert idx < len(inputs)
    
    def test_random_augmentation(self, simple_data):
        """Test random augmentation mode."""
        inputs, outputs = simple_data
        
        dataset = D4AugmentedDataset(
            inputs, outputs,
            augment=True,
            deterministic_augmentation=False
        )
        
        # Length should be same as original
        assert len(dataset) == len(inputs)
        
        # Multiple calls should potentially give different results
        # (though with probability 1/8 they might be the same)
        samples = [dataset[0] for _ in range(10)]
        # At least some should be different
        assert not all(torch.allclose(samples[0][0], s[0]) for s in samples[1:])
    
    def test_2d_tensors(self):
        """Test with 2D tensors (no channel dimension)."""
        inputs_2d = torch.randn(5, 4, 4)
        outputs_2d = torch.randn(5, 4, 4)
        
        dataset = D4AugmentedDataset(inputs_2d, outputs_2d, augment=True)
        sample = dataset[0]
        
        assert sample[0].shape == (4, 4)
        assert sample[1].shape == (4, 4)


class TestD4DataLoader:
    """Test the DataLoader integration."""
    
    def test_dataloader_creation(self, simple_data):
        """Test basic DataLoader creation."""
        inputs, outputs = simple_data
        batch_size = 2
        
        train_loader = create_d4_augmented_dataloader(
            inputs, outputs,
            batch_size=batch_size,
            shuffle=True,
            augment=True,
            deterministic_augmentation=False
        )
        
        # Test iteration
        batch_count = 0
        for batch in train_loader:
            inp_batch, out_batch = batch
            assert inp_batch.shape[0] <= batch_size
            assert out_batch.shape[0] <= batch_size
            assert inp_batch.shape[1:] == inputs.shape[1:]
            assert out_batch.shape[1:] == outputs.shape[1:]
            batch_count += 1
        
        assert batch_count > 0
    
    def test_no_augmentation_validation(self, simple_data):
        """Test that augmentation can be turned off for validation."""
        inputs, outputs = simple_data
        
        # Create validation loader with no augmentation
        val_loader = create_d4_augmented_dataloader(
            inputs, outputs,
            batch_size=len(inputs),
            shuffle=False,
            augment=False
        )
        
        # Check that data is unchanged
        for batch in val_loader:
            inp_batch, out_batch = batch
            assert torch.allclose(inp_batch, inputs)
            assert torch.allclose(out_batch, outputs)
            break  # Only one batch expected
    
    def test_deterministic_dataloader(self):
        """Test DataLoader with deterministic augmentation."""
        inputs = torch.randn(2, 4, 4, 1)
        outputs = torch.randn(2, 4, 4, 1)
        
        loader = create_d4_augmented_dataloader(
            inputs, outputs,
            batch_size=4,
            shuffle=False,
            augment=True,
            deterministic_augmentation=True
        )
        
        # Should have 2 * 8 = 16 samples total, so 4 batches
        batch_count = sum(1 for _ in loader)
        assert batch_count == 4

    @pytest.mark.parametrize("deterministic", [False, True])
    def test_repeat_factor_sampling(self, deterministic):
        """Test that repeat_factor repeats sampling within an epoch."""
        n_samples = 3
        inputs = torch.randn(n_samples, 4, 4, 1)
        outputs = torch.randn(n_samples, 4, 4, 1)

        repeat_factor = 5
        batch_size = 2

        loader = create_d4_augmented_dataloader(
            inputs,
            outputs,
            batch_size=batch_size,
            shuffle=True,  # ignored when sampler is used
            augment=True,
            deterministic_augmentation=deterministic,
            repeat_factor=repeat_factor,
        )

        total_seen = 0
        for batch in loader:
            inp_batch, out_batch = batch[:2]
            assert inp_batch.shape[0] <= batch_size
            total_seen += inp_batch.shape[0]

        base = n_samples * (8 if deterministic else 1)
        assert total_seen == base * repeat_factor


class TestD4CollateFunction:
    """Test the custom collate function."""
    
    def test_collate_basic(self):
        """Test basic collate functionality."""
        # Create simple batch
        batch = [
            (torch.ones(4, 4, 1), torch.zeros(4, 4, 1)),
            (torch.ones(4, 4, 1) * 2, torch.zeros(4, 4, 1)),
            (torch.ones(4, 4, 1) * 3, torch.zeros(4, 4, 1)),
        ]
        
        collate_fn = D4CollateFunction(apply_random_d4=False)
        result = collate_fn(batch)
        
        assert len(result) == 2
        assert result[0].shape == (3, 4, 4, 1)
        assert result[1].shape == (3, 4, 4, 1)
        
        # Check values are preserved when no augmentation
        assert torch.allclose(result[0][0], batch[0][0])
        assert torch.allclose(result[0][1], batch[1][0])
        assert torch.allclose(result[0][2], batch[2][0])
    
    def test_collate_with_augmentation(self):
        """Test collate with augmentation."""
        # Create batch with distinctive patterns
        batch = []
        for i in range(3):
            inp = torch.zeros(4, 4, 1)
            inp[0, 0, 0] = i + 1  # Different marker for each
            out = torch.zeros(4, 4, 1)
            batch.append((inp, out))
        
        collate_fn = D4CollateFunction(apply_random_d4=True)
        result = collate_fn(batch)
        
        assert len(result) == 2
        assert result[0].shape == (3, 4, 4, 1)
        
        # At least some samples should be transformed
        # (with very low probability all could be identity)
        original_positions = [(0, 0), (0, 0), (0, 0)]
        transformed_count = 0
        
        for i in range(3):
            marker_pos = torch.nonzero(result[0][i, :, :, 0] == i + 1)
            if len(marker_pos) == 1 and tuple(marker_pos[0].tolist()) != original_positions[i]:
                transformed_count += 1
        
        # This test might rarely fail due to randomness, but it's very unlikely
        # all 3 samples get identity transform (probability = (1/8)^3 ≈ 0.002)
        assert transformed_count > 0, "No transformations detected (very unlikely, re-run test)"


@pytest.mark.parametrize("augment,expected_factor", [
    (True, 8),
    (False, 1),
])
def test_augmentation_factor(augment, expected_factor):
    """Test that augmentation factor is correct."""
    n_samples = 3
    inputs = torch.randn(n_samples, 4, 4, 1)
    outputs = torch.randn(n_samples, 4, 4, 1)
    
    dataset = D4AugmentedDataset(
        inputs, outputs,
        augment=augment,
        deterministic_augmentation=True
    )
    
    assert len(dataset) == n_samples * expected_factor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])