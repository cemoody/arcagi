import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
import torch

from arcagi.data_loader import (
    create_dataloader,
    filter_by_filename,
    load_npz_data,
    one_hot_to_categorical,
    prepare_dataset,
)


class TestDataLoaderV2(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with temporary NPZ files."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.npz_path = Path(self.temp_dir) / "test_data.npz"
        self.npz_features_path = Path(self.temp_dir) / "test_data_features.npz"

        # Create test data
        self.test_inputs = np.array(
            [
                [[0, 1], [2, 3]],  # Example 1
                [[4, 5], [6, 7]],  # Example 2
                [[8, 9], [-1, 0]],  # Example 3 with mask
            ],
            dtype=np.int32,
        )
        # Expand to 30x30 matrices filled with -1
        expanded_inputs = np.full((3, 30, 30), -1, dtype=np.int32)
        expanded_outputs = np.full((3, 30, 30), -1, dtype=np.int32)

        # Place test data in center
        for i in range(3):
            expanded_inputs[i, 14:16, 14:16] = self.test_inputs[i]
            expanded_outputs[i, 14:16, 14:16] = self.test_inputs[i]

        self.test_filenames = ["file1.json", "file2.json", "file1.json"]
        self.test_indices = [0, 0, 1]

        # Save basic NPZ file (without features)
        np.savez(
            str(self.npz_path),
            inputs=expanded_inputs,
            outputs=expanded_outputs,
            filenames=np.array(self.test_filenames),
            indices=np.array(self.test_indices),
        )

        # Create test features data (44-dimensional)
        test_features = np.random.randint(0, 2, (3, 30, 30, 44), dtype=np.uint8)

        # Save NPZ file with features
        np.savez(
            str(self.npz_features_path),
            inputs=expanded_inputs,
            outputs=expanded_outputs,
            filenames=np.array(self.test_filenames),
            indices=np.array(self.test_indices),
            inputs_features=test_features,
            outputs_features=test_features,
            feature_names=np.array(["order2"], dtype="U32"),
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_npz_data_basic(self):
        """Test basic NPZ loading without features."""
        filenames, indices, inputs, outputs, inputs_features, outputs_features = (
            load_npz_data(str(self.npz_path), use_features=False)
        )

        # Check basic properties
        self.assertEqual(len(filenames), 3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(inputs.shape, (3, 30, 30, 11))  # One-hot encoded
        self.assertEqual(outputs.shape, (3, 30, 30, 11))
        self.assertIsNone(inputs_features)
        self.assertIsNone(outputs_features)

        # Check filenames and indices
        self.assertEqual(filenames, self.test_filenames)
        self.assertEqual(indices, self.test_indices)

        # Check one-hot encoding for first example
        # Color 0 should be 1 at position (14, 14) and 0 elsewhere in that channel
        self.assertEqual(inputs[0, 14, 14, 0].item(), 1.0)  # Color 0
        self.assertEqual(inputs[0, 14, 15, 1].item(), 1.0)  # Color 1
        self.assertEqual(inputs[0, 15, 14, 2].item(), 1.0)  # Color 2
        self.assertEqual(inputs[0, 15, 15, 3].item(), 1.0)  # Color 3

        # Check mask encoding (-1 should map to channel 10)
        self.assertEqual(inputs[0, 0, 0, 10].item(), 1.0)  # Mask at (0,0)

    def test_load_npz_data_with_features(self):
        """Test NPZ loading with features."""
        filenames, indices, inputs, outputs, inputs_features, outputs_features = (
            load_npz_data(str(self.npz_features_path), use_features=True)
        )

        # Check basic properties
        self.assertEqual(len(filenames), 3)
        self.assertEqual(inputs.shape, (3, 30, 30, 11))
        self.assertEqual(outputs.shape, (3, 30, 30, 11))

        # Check features
        self.assertIsNotNone(inputs_features)
        self.assertIsNotNone(outputs_features)
        self.assertEqual(inputs_features.shape, (3, 30, 30, 44))
        self.assertEqual(outputs_features.shape, (3, 30, 30, 44))
        self.assertEqual(inputs_features.dtype, torch.uint8)

    def test_one_hot_to_categorical(self):
        """Test conversion from one-hot to categorical."""
        # Create simple one-hot tensor
        one_hot = torch.zeros(2, 2, 11)
        one_hot[0, 0, 5] = 1  # Color 5
        one_hot[0, 1, 10] = 1  # Mask
        one_hot[1, 0, 0] = 1  # Color 0
        one_hot[1, 1, 9] = 1  # Color 9

        categorical = one_hot_to_categorical(one_hot)

        self.assertEqual(categorical[0, 0].item(), 5)
        self.assertEqual(categorical[0, 1].item(), 10)  # Mask
        self.assertEqual(categorical[1, 0].item(), 0)
        self.assertEqual(categorical[1, 1].item(), 9)

    def test_filter_by_filename(self):
        """Test filtering by filename."""
        filenames, indices, inputs, outputs, inputs_features, outputs_features = (
            load_npz_data(str(self.npz_path), use_features=False)
        )

        # Filter by "file1.json" (should get examples 0 and 2)
        (
            filtered_inputs,
            filtered_outputs,
            filtered_features_in,
            filtered_features_out,
        ) = filter_by_filename(filenames, inputs, outputs, "file1.json", "test")

        self.assertEqual(filtered_inputs.shape[0], 2)  # Two examples
        self.assertEqual(filtered_outputs.shape[0], 2)
        self.assertIsNone(filtered_features_in)
        self.assertIsNone(filtered_features_out)

    def test_prepare_dataset(self):
        """Test dataset preparation."""
        inputs, outputs, inputs_features, outputs_features = prepare_dataset(
            str(self.npz_path), use_features=False
        )

        self.assertEqual(inputs.shape, (3, 30, 30, 11))
        self.assertEqual(outputs.shape, (3, 30, 30, 11))
        self.assertIsNone(inputs_features)
        self.assertIsNone(outputs_features)

    def test_prepare_dataset_with_features(self):
        """Test dataset preparation with features."""
        inputs, outputs, inputs_features, outputs_features = prepare_dataset(
            str(self.npz_features_path), use_features=True
        )

        self.assertEqual(inputs.shape, (3, 30, 30, 11))
        self.assertEqual(outputs.shape, (3, 30, 30, 11))
        self.assertIsNotNone(inputs_features)
        self.assertIsNotNone(outputs_features)
        self.assertEqual(inputs_features.shape, (3, 30, 30, 44))
        self.assertEqual(outputs_features.shape, (3, 30, 30, 44))

    def test_create_dataloader(self):
        """Test DataLoader creation."""
        inputs, outputs, inputs_features, outputs_features = prepare_dataset(
            str(self.npz_path), use_features=False
        )

        dataloader = create_dataloader(
            inputs, outputs, batch_size=2, shuffle=False, num_workers=0
        )

        # Test that we can iterate through the dataloader
        batches: List = list(dataloader)
        self.assertEqual(len(batches), 2)  # 3 examples, batch_size=2 -> 2 batches

        # Check first batch
        batch_inputs, batch_outputs = batches[0]
        self.assertEqual(batch_inputs.shape, (2, 30, 30, 11))
        self.assertEqual(batch_outputs.shape, (2, 30, 30, 11))

    def test_create_dataloader_with_features(self):
        """Test DataLoader creation with features."""
        inputs, outputs, inputs_features, outputs_features = prepare_dataset(
            str(self.npz_features_path), use_features=True
        )

        dataloader = create_dataloader(
            inputs,
            outputs,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            inputs_features=inputs_features,
            outputs_features=outputs_features,
        )

        # Test that we can iterate through the dataloader
        batches: List = list(dataloader)
        self.assertEqual(len(batches), 2)  # 3 examples, batch_size=2 -> 2 batches

        # Check first batch (should have 4 tensors: inputs, outputs, inputs_features, outputs_features)
        batch = batches[0]
        self.assertEqual(len(batch), 4)
        batch_inputs, batch_outputs, batch_inputs_features, batch_outputs_features = (
            batch
        )
        self.assertEqual(batch_inputs.shape, (2, 30, 30, 11))
        self.assertEqual(batch_outputs.shape, (2, 30, 30, 11))
        self.assertEqual(batch_inputs_features.shape, (2, 30, 30, 44))
        self.assertEqual(batch_outputs_features.shape, (2, 30, 30, 44))


if __name__ == "__main__":
    unittest.main()
