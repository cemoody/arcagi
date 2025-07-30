import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from arcagi.data_loader import (
    create_dataloader,
    filter_by_filename,
    load_feature_mapping_dataloader,
    load_npz_data,
    one_hot_to_categorical,
    prepare_dataset,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def test_data():
    """Create test data arrays."""
    test_inputs = np.array(
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
        expanded_inputs[i, 14:16, 14:16] = test_inputs[i]
        expanded_outputs[i, 14:16, 14:16] = test_inputs[i]

    test_filenames = ["file1.json", "file2.json", "file1.json"]
    test_indices = [0, 0, 1]

    return {
        "expanded_inputs": expanded_inputs,
        "expanded_outputs": expanded_outputs,
        "test_filenames": test_filenames,
        "test_indices": test_indices,
        "test_inputs": test_inputs,
    }


@pytest.fixture
def npz_files(temp_dir, test_data):
    """Create test NPZ files."""
    npz_path = Path(temp_dir) / "test_data.npz"
    npz_features_path = Path(temp_dir) / "test_data_features.npz"
    npz_feature_mapping_path = Path(temp_dir) / "test_data_feature_mapping.npz"

    # Save basic NPZ file (without features)
    np.savez(
        str(npz_path),
        inputs=test_data["expanded_inputs"],
        outputs=test_data["expanded_outputs"],
        filenames=np.array(test_data["test_filenames"]),
        indices=np.array(test_data["test_indices"]),
    )

    # Create test features data (44-dimensional)
    test_features = np.random.randint(0, 2, (3, 30, 30, 44), dtype=np.uint8)

    # Save NPZ file with features
    np.savez(
        str(npz_features_path),
        inputs=test_data["expanded_inputs"],
        outputs=test_data["expanded_outputs"],
        filenames=np.array(test_data["test_filenames"]),
        indices=np.array(test_data["test_indices"]),
        inputs_features=test_features,
        outputs_features=test_features,
        feature_names=np.array(["order2"], dtype="U32"),
    )

    # Create test masks for feature mapping
    test_masks_input = np.random.randint(0, 2, (3, 30, 30), dtype=bool)
    test_masks_output = np.random.randint(0, 2, (3, 30, 30), dtype=bool)

    # Create subset information
    subset_is_train = np.array([True, False, True], dtype=bool)

    # Create 147-dimensional features for feature mapping
    test_features_147 = np.random.rand(3, 30, 30, 147).astype(np.float32)

    # Save NPZ file with feature mapping data
    np.savez(
        str(npz_feature_mapping_path),
        inputs=test_data["expanded_inputs"],
        outputs=test_data["expanded_outputs"],
        filenames=np.array(test_data["test_filenames"]),
        indices=np.array(test_data["test_indices"]),
        inputs_features=test_features_147,
        outputs_features=test_features_147,
        inputs_mask=test_masks_input,
        outputs_mask=test_masks_output,
        subset_example_index_is_train=subset_is_train,
        feature_names=np.array(["order2", "edges"], dtype="U32"),
    )

    return {
        "npz_path": npz_path,
        "npz_features_path": npz_features_path,
        "npz_feature_mapping_path": npz_feature_mapping_path,
    }


def test_load_npz_data_basic(npz_files, test_data):
    """Test basic NPZ loading without features."""
    result = load_npz_data(str(npz_files["npz_path"]), use_features=False)
    filenames, indices, inputs, outputs, inputs_features, outputs_features = result[:6]

    # Check basic properties
    assert len(filenames) == 3
    assert len(indices) == 3
    assert inputs.shape == (3, 30, 30, 11)  # One-hot encoded
    assert outputs.shape == (3, 30, 30, 11)
    assert inputs_features is None
    assert outputs_features is None

    # Check filenames and indices
    assert filenames == test_data["test_filenames"]
    assert indices == test_data["test_indices"]

    # Check one-hot encoding for first example
    # Color 0 should be 1 at position (14, 14) and 0 elsewhere in that channel
    assert inputs[0, 14, 14, 0].item() == 1.0  # Color 0
    assert inputs[0, 14, 15, 1].item() == 1.0  # Color 1
    assert inputs[0, 15, 14, 2].item() == 1.0  # Color 2
    assert inputs[0, 15, 15, 3].item() == 1.0  # Color 3

    # Check mask encoding (-1 should map to channel 10)
    assert inputs[0, 0, 0, 10].item() == 1.0  # Mask at (0,0)


def test_load_npz_data_with_features(npz_files):
    """Test NPZ loading with features."""
    result = load_npz_data(str(npz_files["npz_features_path"]), use_features=True)
    filenames, _, inputs, outputs, inputs_features, outputs_features = result[:6]

    # Check basic properties
    assert len(filenames) == 3
    assert inputs.shape == (3, 30, 30, 11)
    assert outputs.shape == (3, 30, 30, 11)

    # Check features
    assert inputs_features is not None
    assert outputs_features is not None
    assert inputs_features.shape == (3, 30, 30, 44)
    assert outputs_features.shape == (3, 30, 30, 44)
    assert inputs_features.dtype == torch.uint8


def test_load_npz_data_with_masks_and_indices(npz_files, test_data):
    """Test NPZ loading with masks and indices for feature mapping."""
    result = load_npz_data(
        str(npz_files["npz_feature_mapping_path"]),
        use_features=True,
        load_masks_and_indices=True,
    )
    (
        filenames,
        indices,
        inputs,
        outputs,
        inputs_features,
        outputs_features,
        subset_is_train,
        inputs_mask,
        outputs_mask,
        indices_tensor,
    ) = result

    # Check basic properties
    assert len(filenames) == 3
    assert inputs.shape == (3, 30, 30, 11)
    assert outputs.shape == (3, 30, 30, 11)

    # Check features
    assert inputs_features is not None
    assert outputs_features is not None
    assert inputs_features.shape == (3, 30, 30, 147)
    assert outputs_features.shape == (3, 30, 30, 147)

    # Check masks
    assert inputs_mask is not None
    assert outputs_mask is not None
    assert inputs_mask.shape == (3, 30, 30)
    assert outputs_mask.shape == (3, 30, 30)
    assert inputs_mask.dtype == torch.bool
    assert outputs_mask.dtype == torch.bool

    # Check indices tensor
    assert indices_tensor is not None
    assert indices_tensor.shape == (3,)
    assert indices_tensor.dtype == torch.long
    assert indices_tensor.tolist() == test_data["test_indices"]

    # Check subset information
    assert subset_is_train is not None
    assert len(subset_is_train) == 3
    assert subset_is_train.dtype == bool


def test_one_hot_to_categorical():
    """Test conversion from one-hot to categorical."""
    # Create simple one-hot tensor
    one_hot = torch.zeros(2, 2, 11)
    one_hot[0, 0, 5] = 1  # Color 5
    one_hot[0, 1, 10] = 1  # Mask
    one_hot[1, 0, 0] = 1  # Color 0
    one_hot[1, 1, 9] = 1  # Color 9

    categorical = one_hot_to_categorical(one_hot)

    assert categorical[0, 0].item() == 5
    assert categorical[0, 1].item() == 10  # Mask
    assert categorical[1, 0].item() == 0
    assert categorical[1, 1].item() == 9


def test_filter_by_filename(npz_files):
    """Test filtering by filename."""
    result = load_npz_data(str(npz_files["npz_path"]), use_features=False)
    filenames, _, inputs, outputs, _, _ = result[:6]

    # Filter by "file1.json" (should get examples 0 and 2)
    (
        filtered_inputs,
        filtered_outputs,
        filtered_features_in,
        filtered_features_out,
    ) = filter_by_filename(filenames, inputs, outputs, "file1.json", "test")

    assert filtered_inputs.shape[0] == 2  # Two examples
    assert filtered_outputs.shape[0] == 2
    assert filtered_features_in is None
    assert filtered_features_out is None


def test_prepare_dataset(npz_files):
    """Test dataset preparation."""
    inputs, outputs, inputs_features, outputs_features = prepare_dataset(
        str(npz_files["npz_path"]), use_features=False
    )

    assert inputs.shape == (3, 30, 30, 11)
    assert outputs.shape == (3, 30, 30, 11)
    assert inputs_features is None
    assert outputs_features is None


def test_prepare_dataset_with_features(npz_files):
    """Test dataset preparation with features."""
    inputs, outputs, inputs_features, outputs_features = prepare_dataset(
        str(npz_files["npz_features_path"]), use_features=True
    )

    assert inputs.shape == (3, 30, 30, 11)
    assert outputs.shape == (3, 30, 30, 11)
    assert inputs_features is not None
    assert outputs_features is not None
    assert inputs_features.shape == (3, 30, 30, 44)
    assert outputs_features.shape == (3, 30, 30, 44)


def test_create_dataloader(npz_files):
    """Test DataLoader creation."""
    inputs, outputs, _, _ = prepare_dataset(
        str(npz_files["npz_path"]), use_features=False
    )

    dataloader = create_dataloader(
        inputs, outputs, batch_size=2, shuffle=False, num_workers=0
    )

    # Test that we can iterate through the dataloader
    batches = list(dataloader)
    assert len(batches) == 2  # 3 examples, batch_size=2 -> 2 batches

    # Check first batch
    batch_inputs, batch_outputs = batches[0]
    assert batch_inputs.shape == (2, 30, 30, 11)
    assert batch_outputs.shape == (2, 30, 30, 11)


def test_create_dataloader_with_features(npz_files):
    """Test DataLoader creation with features."""
    inputs, outputs, inputs_features, outputs_features = prepare_dataset(
        str(npz_files["npz_features_path"]), use_features=True
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
    batches = list(dataloader)
    assert len(batches) == 2  # 3 examples, batch_size=2 -> 2 batches

    # Check first batch (should have 4 tensors: inputs, outputs, inputs_features, outputs_features)
    batch = batches[0]
    assert len(batch) == 4
    batch_inputs, batch_outputs, batch_inputs_features, batch_outputs_features = batch
    assert batch_inputs.shape == (2, 30, 30, 11)
    assert batch_outputs.shape == (2, 30, 30, 11)
    assert batch_inputs_features.shape == (2, 30, 30, 44)
    assert batch_outputs_features.shape == (2, 30, 30, 44)


def test_load_feature_mapping_dataloader_basic(npz_files):
    """Test basic feature mapping dataloader functionality."""
    dataloader = load_feature_mapping_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=2,
        shuffle=False,
    )

    # Test that we can iterate through the dataloader
    batches = list(dataloader)
    assert len(batches) == 2  # 3 examples, batch_size=2 -> 2 batches

    # Check first batch structure
    batch = batches[0]
    assert len(batch) == 6  # 6 tensors expected
    (
        input_features,
        output_features,
        inputs_mask,
        outputs_mask,
        indices,
        subset_is_train,
    ) = batch

    # Check shapes
    assert input_features.shape == (2, 30, 30, 147)
    assert output_features.shape == (2, 30, 30, 147)
    assert inputs_mask.shape == (2, 30, 30)
    assert outputs_mask.shape == (2, 30, 30)
    assert indices.shape == (2,)
    assert subset_is_train.shape == (2,)

    # Check data types
    assert input_features.dtype == torch.float32
    assert output_features.dtype == torch.float32
    assert inputs_mask.dtype == torch.bool
    assert outputs_mask.dtype == torch.bool
    assert indices.dtype == torch.long
    assert subset_is_train.dtype == torch.bool


def test_load_feature_mapping_dataloader_filename_filter(npz_files):
    """Test feature mapping dataloader with filename filtering."""
    # Filter by "file1" (should get examples 0 and 2)
    dataloader = load_feature_mapping_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,  # Large batch to get all examples
        shuffle=False,
        filename_filter="file1",
    )

    batches = list(dataloader)
    assert len(batches) == 1  # All examples should fit in one batch

    batch = batches[0]
    (
        input_features,
        output_features,
        inputs_mask,
        outputs_mask,
        indices,
        subset_is_train,
    ) = batch

    # Should have 2 examples (indices 0 and 2 from original data)
    assert input_features.shape[0] == 2
    assert indices.tolist() == [
        0,
        1,
    ]  # Original indices were [0, 0, 1], filtered to [0, 1]


def test_load_feature_mapping_dataloader_subset_filter(npz_files):
    """Test feature mapping dataloader with subset filtering."""
    # Filter by train subset (should get examples 0 and 2)
    dataloader = load_feature_mapping_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,
        shuffle=False,
        use_train_subset=True,
    )

    batches = list(dataloader)
    assert len(batches) == 1

    batch = batches[0]
    (
        input_features,
        output_features,
        inputs_mask,
        outputs_mask,
        indices,
        subset_is_train,
    ) = batch

    # Should have 2 examples (train subset)
    assert input_features.shape[0] == 2
    # All should be from train subset
    assert subset_is_train.all().item()

    # Test test subset
    dataloader_test = load_feature_mapping_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,
        shuffle=False,
        use_train_subset=False,
    )

    batches_test = list(dataloader_test)
    assert len(batches_test) == 1

    batch_test = batches_test[0]
    input_features_test, _, _, _, _, subset_is_train_test = batch_test

    # Should have 1 example (test subset)
    assert input_features_test.shape[0] == 1
    # All should be from test subset
    assert not subset_is_train_test.all().item()


def test_load_feature_mapping_dataloader_data_augmentation(npz_files):
    """Test feature mapping dataloader with data augmentation."""
    dataloader = load_feature_mapping_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,
        shuffle=False,
        data_augment_factor=2,
    )

    batches = list(dataloader)
    assert len(batches) == 1

    batch = batches[0]
    (
        input_features,
        output_features,
        inputs_mask,
        outputs_mask,
        indices,
        subset_is_train,
    ) = batch

    # Should have 6 examples (3 original * 2 augmentation factor)
    assert input_features.shape[0] == 6
    assert output_features.shape[0] == 6
    assert inputs_mask.shape[0] == 6
    assert outputs_mask.shape[0] == 6
    assert indices.shape[0] == 6
    assert subset_is_train.shape[0] == 6


def test_load_feature_mapping_dataloader_empty_filter(npz_files):
    """Test feature mapping dataloader with filter that matches no examples."""
    dataloader = load_feature_mapping_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=2,
        shuffle=False,
        filename_filter="nonexistent_file",
    )

    # Should return empty dataloader
    batches = list(dataloader)
    assert len(batches) == 0


def test_load_feature_mapping_dataloader_missing_data_error(npz_files):
    """Test that feature mapping dataloader raises errors for missing required data."""
    # Test with NPZ file that doesn't have features
    with pytest.raises(ValueError, match="Feature mapping requires features data"):
        load_feature_mapping_dataloader(str(npz_files["npz_path"]))

    # Test with NPZ file that has features but no masks
    with pytest.raises(ValueError, match="Feature mapping requires mask data"):
        load_feature_mapping_dataloader(str(npz_files["npz_features_path"]))
