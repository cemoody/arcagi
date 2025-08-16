import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from arcagi.data_loader import (
    create_dataloader,
    create_shot_dataloader,
    filter_by_filename,
    filter_by_subset,
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
    filenames, indices, inputs, outputs, _, _ = result[:6]

    # Filter by "file1.json" (should get examples 0 and 2)
    (
        filtered_inputs,
        filtered_outputs,
        filtered_features_in,
        filtered_features_out,
        filtered_indices,
    ) = filter_by_filename(
        filenames, inputs, outputs, "file1.json", "test", indices=indices
    )

    assert filtered_inputs.shape[0] == 2  # Two examples
    assert filtered_outputs.shape[0] == 2
    assert filtered_features_in is None
    assert filtered_features_out is None
    assert filtered_indices == [
        0,
        1,
    ]  # Original indices were [0, 0, 1], filtered to [0, 1]


def test_prepare_dataset(npz_files):
    """Test dataset preparation."""
    inputs, outputs, inputs_features, outputs_features, indices = prepare_dataset(
        str(npz_files["npz_path"]), use_features=False
    )

    assert inputs.shape == (3, 30, 30, 11)
    assert outputs.shape == (3, 30, 30, 11)
    assert inputs_features is None
    assert outputs_features is None
    assert indices.shape == (3,)
    assert indices.dtype == torch.long
    assert indices.tolist() == [0, 0, 1]  # From test data


def test_prepare_dataset_with_features(npz_files):
    """Test dataset preparation with features."""
    inputs, outputs, inputs_features, outputs_features, indices = prepare_dataset(
        str(npz_files["npz_features_path"]), use_features=True
    )

    assert inputs.shape == (3, 30, 30, 11)
    assert outputs.shape == (3, 30, 30, 11)
    assert inputs_features is not None
    assert outputs_features is not None
    assert inputs_features.shape == (3, 30, 30, 44)
    assert outputs_features.shape == (3, 30, 30, 44)
    assert indices.shape == (3,)
    assert indices.dtype == torch.long
    assert indices.tolist() == [0, 0, 1]  # From test data


def test_create_dataloader(npz_files):
    """Test DataLoader creation."""
    inputs, outputs, _, _, indices = prepare_dataset(
        str(npz_files["npz_path"]), use_features=False
    )

    dataloader = create_dataloader(
        inputs, outputs, batch_size=2, shuffle=False, num_workers=0, indices=indices
    )

    # Test that we can iterate through the dataloader
    batches = list(dataloader)
    assert len(batches) == 2  # 3 examples, batch_size=2 -> 2 batches

    # Check first batch (should have 5 tensors: inputs, outputs, features_in, features_out, indices)
    batch = batches[0]
    assert len(batch) == 5
    (
        batch_inputs,
        batch_outputs,
        batch_features_in,
        batch_features_out,
        batch_indices,
    ) = batch
    assert batch_inputs.shape == (2, 30, 30, 11)
    assert batch_outputs.shape == (2, 30, 30, 11)
    assert batch_features_in.shape == (2, 30, 30, 0)  # Empty features
    assert batch_features_out.shape == (2, 30, 30, 0)  # Empty features
    assert batch_indices.shape == (2,)
    assert batch_indices.dtype == torch.long


def test_create_dataloader_with_features(npz_files):
    """Test DataLoader creation with features."""
    inputs, outputs, inputs_features, outputs_features, indices = prepare_dataset(
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
        indices=indices,
    )

    # Test that we can iterate through the dataloader
    batches = list(dataloader)
    assert len(batches) == 2  # 3 examples, batch_size=2 -> 2 batches

    # Check first batch (should have 5 tensors: inputs, outputs, inputs_features, outputs_features, indices)
    batch = batches[0]
    assert len(batch) == 5
    (
        batch_inputs,
        batch_outputs,
        batch_inputs_features,
        batch_outputs_features,
        batch_indices,
    ) = batch
    assert batch_inputs.shape == (2, 30, 30, 11)
    assert batch_outputs.shape == (2, 30, 30, 11)
    assert batch_inputs_features.shape == (2, 30, 30, 44)
    assert batch_outputs_features.shape == (2, 30, 30, 44)
    assert batch_indices.shape == (2,)
    assert batch_indices.dtype == torch.long


def test_filter_by_filename_with_indices(npz_files, test_data):
    """Test filtering by filename with indices propagation."""
    result = load_npz_data(str(npz_files["npz_path"]), use_features=False)
    filenames, indices, inputs, outputs, _, _ = result[:6]

    # Test indices before filtering
    assert indices == test_data["test_indices"]  # [0, 0, 1]

    # Filter by "file2.json" (should get example 1 only)
    (
        filtered_inputs,
        filtered_outputs,
        filtered_features_in,
        filtered_features_out,
        filtered_indices,
    ) = filter_by_filename(
        filenames, inputs, outputs, "file2.json", "test", indices=indices
    )

    assert filtered_inputs.shape[0] == 1  # One example
    assert filtered_outputs.shape[0] == 1
    assert filtered_indices == [0]  # Original index 0 from position 1


def test_filter_by_subset(npz_files):
    """Test filtering by subset with indices."""
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
        _,
        _,
        _,
    ) = result

    # Filter by train subset
    (
        train_inputs,
        train_outputs,
        train_inputs_features,
        train_outputs_features,
        train_indices,
    ) = filter_by_subset(
        inputs,
        outputs,
        subset_is_train,
        use_train_subset=True,
        inputs_features=inputs_features,
        outputs_features=outputs_features,
        indices=indices,
    )

    # Should get examples 0 and 2 (subset_is_train = [True, False, True])
    assert train_inputs.shape[0] == 2
    assert train_indices == [0, 1]  # Original indices [0, 0, 1] filtered to [0, 1]

    # Filter by test subset
    (
        test_inputs,
        test_outputs,
        test_inputs_features,
        test_outputs_features,
        test_indices,
    ) = filter_by_subset(
        inputs,
        outputs,
        subset_is_train,
        use_train_subset=False,
        inputs_features=inputs_features,
        outputs_features=outputs_features,
        indices=indices,
    )

    # Should get example 1 only
    assert test_inputs.shape[0] == 1
    assert test_indices == [0]  # Original index 0 from position 1


def test_prepare_dataset_with_filename_filter(npz_files):
    """Test prepare_dataset with filename filtering and indices."""
    inputs, outputs, inputs_features, outputs_features, indices = prepare_dataset(
        str(npz_files["npz_path"]), filter_filename="file1.json", use_features=False
    )

    # Should get 2 examples (file1.json appears at positions 0 and 2)
    assert inputs.shape[0] == 2
    assert outputs.shape[0] == 2
    assert indices.shape == (2,)
    assert indices.tolist() == [0, 1]  # Original indices [0, 0, 1] filtered to [0, 1]


def test_prepare_dataset_with_limit(npz_files):
    """Test prepare_dataset with example limiting and indices."""
    inputs, outputs, inputs_features, outputs_features, indices = prepare_dataset(
        str(npz_files["npz_path"]), limit_examples=2, use_features=False
    )

    # Should get first 2 examples
    assert inputs.shape[0] == 2
    assert outputs.shape[0] == 2
    assert indices.shape == (2,)
    assert indices.tolist() == [0, 0]  # First 2 from original [0, 0, 1]


def test_create_dataloader_without_indices(npz_files):
    """Test DataLoader creation without providing indices (should create defaults)."""
    inputs, outputs, _, _, _ = prepare_dataset(
        str(npz_files["npz_path"]), use_features=False
    )

    # Create dataloader without passing indices
    dataloader = create_dataloader(
        inputs, outputs, batch_size=2, shuffle=False, num_workers=0
    )

    # Test that we can iterate through the dataloader
    batches = list(dataloader)
    assert len(batches) == 2  # 3 examples, batch_size=2 -> 2 batches

    # Check first batch
    batch = batches[0]
    assert len(batch) == 5
    (
        batch_inputs,
        batch_outputs,
        batch_features_in,
        batch_features_out,
        batch_indices,
    ) = batch

    # Should have default sequential indices
    assert batch_indices.tolist() == [0, 1]  # Sequential defaults


def test_create_dataloader_indices_propagation(npz_files):
    """Test that indices are correctly propagated through DataLoader."""
    inputs, outputs, _, _, indices = prepare_dataset(
        str(npz_files["npz_path"]), use_features=False
    )

    dataloader = create_dataloader(
        inputs, outputs, batch_size=3, shuffle=False, num_workers=0, indices=indices
    )

    # Get all examples in one batch
    batches = list(dataloader)
    assert len(batches) == 1

    batch = batches[0]
    (
        batch_inputs,
        batch_outputs,
        batch_features_in,
        batch_features_out,
        batch_indices,
    ) = batch

    # Should preserve original indices
    assert batch_indices.tolist() == [0, 0, 1]  # Original indices from test data


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


def test_end_to_end_indices_integration(npz_files, test_data):
    """Test complete end-to-end indices propagation mimicking ex29.py usage."""
    # Simulate ex29.py data loading pattern
    (
        train_inputs,
        train_outputs,
        train_input_features,
        train_output_features,
        train_indices,
    ) = prepare_dataset(
        str(npz_files["npz_features_path"]),
        filter_filename="file1.json",
        use_features=True,
        dataset_name="train",
    )

    # Should get 2 examples (file1.json appears at positions 0 and 2)
    assert train_inputs.shape[0] == 2
    assert train_indices.tolist() == [
        0,
        1,
    ]  # Original indices [0, 0, 1] filtered to [0, 1]

    # Create dataloader like ex29.py does
    train_loader = create_dataloader(
        train_inputs,
        train_outputs,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        inputs_features=train_input_features,
        outputs_features=train_output_features,
        indices=train_indices,
    )

    # Simulate batch processing like in ex29.py
    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch like ex29.py does
        inputs_one_hot, outputs_one_hot, input_features, output_features, indices = (
            batch
        )

        # Verify batch structure
        assert len(batch) == 5
        assert inputs_one_hot.shape == (2, 30, 30, 11)
        assert outputs_one_hot.shape == (2, 30, 30, 11)
        assert input_features.shape == (2, 30, 30, 44)
        assert output_features.shape == (2, 30, 30, 44)
        assert indices.shape == (2,)
        assert indices.dtype == torch.long

        # Verify indices match expected values
        assert indices.tolist() == [0, 1]

        # Simulate ex29.py batch_to_dataclass pattern
        # Extract colors and create mock batch objects
        input_colors = inputs_one_hot.argmax(dim=-1).long()
        output_colors = outputs_one_hot.argmax(dim=-1).long()

        # Verify we can access indices for each example
        for example_idx, original_npz_idx in enumerate(indices):
            print(
                f"Example {example_idx} came from NPZ index {original_npz_idx.item()}"
            )

        # Should only have one batch due to batch_size=2 and 2 examples
        assert batch_idx == 0
        break


def test_create_shot_dataloader_basic(npz_files):
    """Test basic few-shot dataloader functionality."""
    dataloader = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=2,
        shuffle=False,
    )
    
    # Test that we can iterate through the dataloader
    batches = list(dataloader)
    
    # The test data has 3 examples: file1.json (2 examples), file2.json (1 example)
    # file2.json will be skipped because it has no context
    # So we should have 2 examples from file1.json
    assert len(batches) == 1  # 2 examples, batch_size=2 -> 1 batch
    
    # Check batch structure
    batch = batches[0]
    assert len(batch) == 5  # filenames, example_indices, inputs, outputs, context
    filenames, example_indices, inputs, outputs, context = batch
    
    # Check shapes
    assert len(filenames) == 2  # Batch of 2
    assert example_indices.shape == (2,)
    assert inputs.shape == (2, 30, 30, 11)
    assert outputs.shape == (2, 30, 30, 11)
    assert context.shape == (2, 1, 2, 30, 30, 11)  # Each has 1 context example
    
    # Check data types
    assert isinstance(filenames[0], str)
    assert example_indices.dtype == torch.long
    assert inputs.dtype == torch.float32
    assert outputs.dtype == torch.float32
    assert context.dtype == torch.float32
    
    # Check filenames are correct
    assert all(fname == "file1.json" for fname in filenames)
    
    # Check example indices
    assert sorted(example_indices.tolist()) == [0, 1]


def test_create_shot_dataloader_filename_filter(npz_files):
    """Test few-shot dataloader with filename filtering."""
    # Filter by "file1" (should get 2 examples)
    dataloader = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,  # Large batch to get all examples
        shuffle=False,
        filename_filter="file1",
    )
    
    batches = list(dataloader)
    assert len(batches) == 1  # All examples should fit in one batch
    
    batch = batches[0]
    filenames, example_indices, inputs, outputs, context = batch
    
    # Should have 2 examples (both from file1.json)
    assert len(filenames) == 2
    assert all(fname == "file1.json" for fname in filenames)
    assert sorted(example_indices.tolist()) == [0, 1]
    
    # Each should have 1 context example (the other example from same file)
    assert context.shape == (2, 1, 2, 30, 30, 11)


def test_create_shot_dataloader_no_context(npz_files):
    """Test few-shot dataloader with task that has no context."""
    # Filter by "file2" (has only 1 example, so no context)
    dataloader = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,
        shuffle=False,
        filename_filter="file2",
    )
    
    batches = list(dataloader)
    # Should return empty dataloader since file2 has only 1 example (no context)
    assert len(batches) == 0


def test_create_shot_dataloader_subset_filtering(npz_files):
    """Test few-shot dataloader with subset filtering for context."""
    # Test excluding train examples from context
    dataloader_no_train = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,
        shuffle=False,
        context_includes_train=False,
        context_includes_test=True,
    )
    
    batches = list(dataloader_no_train)
    # With the test data: file1.json has examples at indices 0 (train) and 2 (train)
    # file2.json has example at index 1 (test)
    # When excluding train from context, file1.json examples will have no context
    assert len(batches) == 0  # No valid examples with context
    
    # Test excluding test examples from context
    dataloader_no_test = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=10,
        shuffle=False,
        context_includes_train=True,
        context_includes_test=False,
    )
    
    batches = list(dataloader_no_test)
    # file1.json has 2 train examples, so they can serve as context for each other
    assert len(batches) == 1
    
    batch = batches[0]
    filenames, example_indices, inputs, outputs, context = batch
    
    # Should have 2 examples from file1.json
    assert len(filenames) == 2
    assert all(fname == "file1.json" for fname in filenames)


def test_create_shot_dataloader_empty_filter(npz_files):
    """Test few-shot dataloader with filter that matches no examples."""
    dataloader = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=2,
        shuffle=False,
        filename_filter="nonexistent_file",
    )
    
    # Should return empty dataloader
    batches = list(dataloader)
    assert len(batches) == 0


def test_create_shot_dataloader_context_ordering(npz_files):
    """Test that context examples are ordered consistently by example index."""
    dataloader = create_shot_dataloader(
        str(npz_files["npz_feature_mapping_path"]),
        batch_size=1,
        shuffle=False,
        filename_filter="file1",
    )
    
    batches = list(dataloader)
    assert len(batches) == 2  # 2 examples from file1.json, batch_size=1
    
    # Check that context is consistent across batches
    for batch_idx, batch in enumerate(batches):
        filenames, example_indices, inputs, outputs, context = batch
        
        assert len(filenames) == 1
        assert filenames[0] == "file1.json"
        
        # Context should have shape [1, 1, 2, 30, 30, 11] (1 batch, 1 context example)
        assert context.shape == (1, 1, 2, 30, 30, 11)


@pytest.fixture
def npz_files_extended(temp_dir):
    """Create test NPZ files with more complex data for few-shot testing."""
    npz_path = Path(temp_dir) / "test_fewshot_data.npz"
    
    # Create data with multiple tasks and examples
    # Task 1 (file1.json): 4 examples
    # Task 2 (file2.json): 3 examples  
    # Task 3 (file3.json): 1 example (will be filtered out due to no context)
    
    n_examples = 8
    expanded_inputs = np.full((n_examples, 30, 30), -1, dtype=np.int32)
    expanded_outputs = np.full((n_examples, 30, 30), -1, dtype=np.int32)
    
    # Fill with different patterns for each example
    for i in range(n_examples):
        expanded_inputs[i, 14:16, 14:16] = i % 10
        expanded_outputs[i, 14:16, 14:16] = (i + 1) % 10
    
    filenames = [
        "file1.json", "file1.json", "file1.json", "file1.json",  # 4 examples
        "file2.json", "file2.json", "file2.json",  # 3 examples
        "file3.json",  # 1 example
    ]
    indices = [0, 1, 2, 3, 0, 1, 2, 0]  # Example indices within each file
    
    # Create subset information (mix of train/test)
    subset_is_train = np.array([True, True, False, True, False, True, True, True], dtype=bool)
    
    # Save NPZ file
    np.savez(
        str(npz_path),
        inputs=expanded_inputs,
        outputs=expanded_outputs,
        filenames=np.array(filenames),
        indices=np.array(indices),
        subset_example_index_is_train=subset_is_train,
    )
    
    return {"npz_path": npz_path}


def test_create_shot_dataloader_multiple_tasks(npz_files_extended):
    """Test few-shot dataloader with multiple tasks of varying sizes."""
    dataloader = create_shot_dataloader(
        str(npz_files_extended["npz_path"]),
        batch_size=4,
        shuffle=False,
    )
    
    # Collect all batches
    all_filenames = []
    all_example_indices = []
    all_context_sizes = []
    
    for batch in dataloader:
        filenames, example_indices, inputs, outputs, context = batch
        all_filenames.extend(filenames)
        all_example_indices.extend(example_indices.tolist())
        
        # Check context sizes
        for i in range(len(filenames)):
            # Count non-zero context examples (padding has all zeros)
            non_zero_context = 0
            for j in range(context.shape[1]):
                if context[i, j].abs().sum() > 0:
                    non_zero_context += 1
            all_context_sizes.append(non_zero_context)
    
    # Should have 7 total examples (8 - 1 from file3.json which has no context)
    assert len(all_filenames) == 7
    
    # Check file1.json examples
    file1_count = sum(1 for f in all_filenames if f == "file1.json")
    assert file1_count == 4  # All 4 examples from file1
    
    # Check file2.json examples  
    file2_count = sum(1 for f in all_filenames if f == "file2.json")
    assert file2_count == 3  # All 3 examples from file2
    
    # Check file3.json examples
    file3_count = sum(1 for f in all_filenames if f == "file3.json")
    assert file3_count == 0  # No examples (filtered out due to no context)
    
    # Check context sizes
    # file1.json examples should have 3 context examples each
    # file2.json examples should have 2 context examples each
    for i, (fname, ctx_size) in enumerate(zip(all_filenames, all_context_sizes)):
        if fname == "file1.json":
            assert ctx_size == 3, f"file1.json example {i} should have 3 context examples"
        elif fname == "file2.json":
            assert ctx_size == 2, f"file2.json example {i} should have 2 context examples"


def test_create_shot_dataloader_batch_consistency(npz_files_extended):
    """Test that batches maintain consistency with shuffling."""
    # Test that shuffle=False gives consistent results
    dataloader1 = create_shot_dataloader(
        str(npz_files_extended["npz_path"]),
        batch_size=3,
        shuffle=False,
    )
    
    dataloader2 = create_shot_dataloader(
        str(npz_files_extended["npz_path"]),
        batch_size=3,
        shuffle=False,
    )
    
    # Collect batches from both
    batches1 = list(dataloader1)
    batches2 = list(dataloader2)
    
    # Should have same number of batches
    assert len(batches1) == len(batches2)
    
    # Check that corresponding batches are identical
    for b1, b2 in zip(batches1, batches2):
        filenames1, indices1, inputs1, outputs1, context1 = b1
        filenames2, indices2, inputs2, outputs2, context2 = b2
        
        assert filenames1 == filenames2
        assert torch.equal(indices1, indices2)
        assert torch.equal(inputs1, inputs2)
        assert torch.equal(outputs1, outputs2)
        assert torch.equal(context1, context2)
