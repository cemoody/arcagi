from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset


def load_npz_data(
    npz_path: str,
    use_features: bool = False,
    load_masks_and_indices: bool = False,
) -> Tuple[
    List[str],
    List[int],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[NDArray[np.bool_]],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Loads data from an NPZ file created by preprocess_v2.py and creates PyTorch tensors.

    Args:
        npz_path: Path to the NPZ file
        use_features: Whether to load concatenated features (if available)
        load_masks_and_indices: Whether to load mask data and indices (for feature mapping)

    Returns:
        filenames: List of filename strings
        indices: List of example indices within each file
        inputs: torch.Tensor of shape (n_examples, 30, 30, 11) with one-hot encoded colors
        outputs: torch.Tensor of shape (n_examples, 30, 30, 11) with one-hot encoded colors
        inputs_features: torch.Tensor of shape (n_examples, 30, 30, n_features) with binary features (if use_features=True and available)
        outputs_features: torch.Tensor of shape (n_examples, 30, 30, n_features) with binary features (if use_features=True and available)
        subset_example_index_is_train: np.ndarray of booleans indicating if example is from "train" subset (True) or "test" subset (False)
        inputs_mask: torch.Tensor of shape (n_examples, 30, 30) with boolean masks (if load_masks_and_indices=True and available)
        outputs_mask: torch.Tensor of shape (n_examples, 30, 30) with boolean masks (if load_masks_and_indices=True and available)
        indices_tensor: torch.Tensor of shape (n_examples,) with example indices (if load_masks_and_indices=True and available)
    """
    # Load the NPZ file
    data = np.load(npz_path)

    # Extract basic data
    filenames: List[str] = data["filenames"].tolist()
    indices: List[int] = data["indices"].tolist()
    inputs_raw: NDArray[np.int32] = data["inputs"]  # Shape: (n_examples, 30, 30)
    outputs_raw: NDArray[np.int32] = data["outputs"]  # Shape: (n_examples, 30, 30)

    # Load subset information if available
    subset_example_index_is_train = None
    if "subset_example_index_is_train" in data:
        subset_example_index_is_train = data["subset_example_index_is_train"]

    # Get number of examples
    num_examples: int = len(filenames)

    # Convert raw color data to one-hot encoded tensors
    inputs: torch.Tensor = torch.zeros((num_examples, 30, 30, 11), dtype=torch.float32)
    outputs: torch.Tensor = torch.zeros((num_examples, 30, 30, 11), dtype=torch.float32)

    # Convert to torch tensors first
    inputs_raw_tensor = torch.from_numpy(inputs_raw).long()  # type: ignore
    outputs_raw_tensor = torch.from_numpy(outputs_raw).long()  # type: ignore

    # Fill in one-hot encoded values for colors 0-9
    for color in range(10):
        inputs[:, :, :, color] = (inputs_raw_tensor == color).float()
        outputs[:, :, :, color] = (outputs_raw_tensor == color).float()

    # Color 10 represents mask (-1)
    inputs[:, :, :, 10] = (inputs_raw_tensor == -1).float()
    outputs[:, :, :, 10] = (outputs_raw_tensor == -1).float()

    # Load features if requested and available
    inputs_features: Optional[torch.Tensor] = None
    outputs_features: Optional[torch.Tensor] = None

    if use_features:
        if "inputs_features" in data and "outputs_features" in data:
            inputs_features_raw: NDArray[np.uint8] = data["inputs_features"]
            outputs_features_raw: NDArray[np.uint8] = data["outputs_features"]
            inputs_features = torch.from_numpy(inputs_features_raw)  # type: ignore
            outputs_features = torch.from_numpy(outputs_features_raw)  # type: ignore
            feature_names = data.get("feature_names", [])
            print(
                f"Loaded features: {inputs_features.shape}, features: {feature_names}"
            )
        else:
            print("Warning: Features requested but not found in NPZ file")

    # Load masks and indices if requested and available
    inputs_mask: Optional[torch.Tensor] = None
    outputs_mask: Optional[torch.Tensor] = None
    indices_tensor: Optional[torch.Tensor] = None

    if load_masks_and_indices:
        if "inputs_mask" in data and "outputs_mask" in data:
            inputs_mask = torch.from_numpy(data["inputs_mask"]).bool()
            outputs_mask = torch.from_numpy(data["outputs_mask"]).bool()
            print(
                f"Loaded masks: inputs_mask={inputs_mask.shape}, outputs_mask={outputs_mask.shape}"
            )
        else:
            print("Warning: Masks requested but not found in NPZ file")

        # Convert indices to tensor
        indices_tensor = torch.from_numpy(np.array(indices)).long()

    return (
        filenames,
        indices,
        inputs,
        outputs,
        inputs_features,
        outputs_features,
        subset_example_index_is_train,
        inputs_mask,
        outputs_mask,
        indices_tensor,
    )


def one_hot_to_categorical(
    one_hot_tensor: torch.Tensor, last_value: int = 10
) -> torch.Tensor:
    """Convert one-hot encoded tensor to categorical values."""
    # Get the indices of the maximum values along the last dimension
    cat: torch.Tensor = torch.argmax(one_hot_tensor, dim=-1)
    sum_vals: torch.Tensor = one_hot_tensor.sum(dim=-1)
    cat[sum_vals <= 1e-6] = last_value
    return cat


def filter_by_filename(
    filenames: List[str],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    target_filename: str,
    dataset_name: str = "data",
    inputs_features: Optional[torch.Tensor] = None,
    outputs_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Filter tensors to only include examples from a specific filename.

    Args:
        filenames: List of filenames for each example
        inputs: Input tensor to filter
        outputs: Output tensor to filter
        target_filename: Filename to filter by
        dataset_name: Name of dataset for logging (e.g., "training", "validation")
        inputs_features: Optional invariant input features to filter
        outputs_features: Optional invariant output features to filter

    Returns:
        Filtered inputs, outputs, and optionally invariant tensors

    Raises:
        ValueError: If no examples found with the target filename (only for non-validation sets)
    """
    print(f"Filtering {dataset_name} to filename: {target_filename}")

    # Find indices where filename matches
    mask = [fname == target_filename for fname in filenames]
    filtered_indices = [i for i, m in enumerate(mask) if m]

    if not filtered_indices:
        if dataset_name.lower() == "validation":
            print(
                f"Warning: No {dataset_name} examples found with filename: {target_filename}"
            )
            empty_inputs = torch.empty(0, 30, 30, 11)
            empty_outputs = torch.empty(0, 30, 30, 11)
            empty_features = (
                torch.empty(0, 30, 30, 44) if inputs_features is not None else None
            )
            return empty_inputs, empty_outputs, empty_features, empty_features
        else:
            raise ValueError(
                f"No {dataset_name} examples found with filename: {target_filename}"
            )

    # Filter the tensors
    filtered_inputs = inputs[filtered_indices]
    filtered_outputs = outputs[filtered_indices]

    filtered_inputs_features = None
    filtered_outputs_features = None
    if inputs_features is not None:
        filtered_inputs_features = inputs_features[filtered_indices]
    if outputs_features is not None:
        filtered_outputs_features = outputs_features[filtered_indices]

    print(
        f"Found {len(filtered_indices)} {dataset_name} examples with filename {target_filename}"
    )

    return (
        filtered_inputs,
        filtered_outputs,
        filtered_inputs_features,
        filtered_outputs_features,
    )


def filter_by_subset(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    subset_is_train: Optional[NDArray[np.bool_]],
    use_train_subset: bool = True,
    inputs_features: Optional[torch.Tensor] = None,
    outputs_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Filter tensors to only include examples from a specific subset (train or test within files).

    Args:
        inputs: Input tensor to filter
        outputs: Output tensor to filter
        subset_is_train: Boolean array indicating which examples are from "train" subset
        use_train_subset: If True, return examples from "train" subset; if False, return examples from "test" subset
        inputs_features: Optional input features to filter
        outputs_features: Optional output features to filter

    Returns:
        Filtered inputs, outputs, and optionally feature tensors
    """
    if subset_is_train is None:
        print("Warning: No subset information available, returning all examples")
        return inputs, outputs, inputs_features, outputs_features

    # Create mask for the desired subset
    if use_train_subset:
        mask = subset_is_train
        subset_name = "train"
    else:
        mask = ~subset_is_train
        subset_name = "test"

    # Find indices that match the subset
    filtered_indices = np.where(mask)[0]

    if len(filtered_indices) == 0:
        print(f"Warning: No examples found in {subset_name} subset")
        empty_inputs = torch.empty(0, 30, 30, 11)
        empty_outputs = torch.empty(0, 30, 30, 11)
        empty_features = (
            torch.empty(0, 30, 30, 44) if inputs_features is not None else None
        )
        return empty_inputs, empty_outputs, empty_features, empty_features

    # Filter the tensors
    filtered_inputs = inputs[filtered_indices]
    filtered_outputs = outputs[filtered_indices]

    filtered_inputs_features = None
    filtered_outputs_features = None
    if inputs_features is not None:
        filtered_inputs_features = inputs_features[filtered_indices]
    if outputs_features is not None:
        filtered_outputs_features = outputs_features[filtered_indices]

    print(f"Found {len(filtered_indices)} examples in {subset_name} subset")

    return (
        filtered_inputs,
        filtered_outputs,
        filtered_inputs_features,
        filtered_outputs_features,
    )


def prepare_dataset(
    npz_path: str,
    filter_filename: Optional[str] = None,
    limit_examples: Optional[int] = None,
    use_features: bool = False,
    dataset_name: str = "data",
    use_train_subset: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load and prepare a dataset from an NPZ file with optional filtering.

    Args:
        npz_path: Path to the NPZ file
        filter_filename: Optional filename to filter examples
        limit_examples: Optional limit on number of examples
        use_features: Whether to load invariant features
        dataset_name: Name of dataset for logging
        use_train_subset: Optional bool to filter by subset (True=train subset, False=test subset, None=all)

    Returns:
        Prepared inputs, outputs, and optionally invariant tensors
    """
    print(f"Loading {dataset_name} from {npz_path}...")
    (
        filenames,
        _,
        inputs,
        outputs,
        inputs_features,
        outputs_features,
        subset_is_train,
    ) = load_npz_data(npz_path, use_features=use_features)[
        :7
    ]  # Take only first 7 elements

    # Apply subset filtering if requested
    if use_train_subset is not None:
        inputs, outputs, inputs_features, outputs_features = filter_by_subset(
            inputs,
            outputs,
            subset_is_train,
            use_train_subset,
            inputs_features,
            outputs_features,
        )

    # Apply filename filtering if requested
    if filter_filename is not None:
        # We need to update filenames list if subset filtering was applied
        if use_train_subset is not None and subset_is_train is not None:
            mask = subset_is_train if use_train_subset else ~subset_is_train
            filtered_filenames: List[str] = [filenames[i] for i in np.where(mask)[0]]
        else:
            filtered_filenames = filenames

        inputs, outputs, inputs_features, outputs_features = filter_by_filename(
            filtered_filenames,
            inputs,
            outputs,
            filter_filename,
            dataset_name,
            inputs_features,
            outputs_features,
        )

    # Apply limit if requested
    if limit_examples is not None:
        inputs = inputs[:limit_examples]
        outputs = outputs[:limit_examples]
        if inputs_features is not None:
            inputs_features = inputs_features[:limit_examples]
        if outputs_features is not None:
            outputs_features = outputs_features[:limit_examples]
        print(f"Limited to {limit_examples} {dataset_name} examples")

    print(
        f"{dataset_name.capitalize()} shape: inputs={inputs.shape}, outputs={outputs.shape}"
    )
    if inputs_features is not None:
        print(
            f"{dataset_name.capitalize()} invariant shape: inputs_features={inputs_features.shape}"
        )

    return inputs, outputs, inputs_features, outputs_features


def create_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    inputs_features: Optional[torch.Tensor] = None,
    outputs_features: Optional[torch.Tensor] = None,
) -> DataLoader[Any]:
    """Create a DataLoader from input and output tensors.

    Args:
        inputs: Input tensor
        outputs: Output tensor
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        inputs_features: Optional invariant input features
        outputs_features: Optional invariant output features

    Returns:
        DataLoader instance
    """
    if inputs_features is not None and outputs_features is not None:
        dataset = TensorDataset(inputs, outputs, inputs_features, outputs_features)
    else:
        dataset = TensorDataset(inputs, outputs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_feature_data(
    train_path: str,
    val_path: str,
    filename_filter: Optional[str] = None,
    use_features: bool = True,
    use_train_subset: Optional[bool] = None,
    use_val_subset: Optional[bool] = None,
    data_augment_factor: int = 1,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Load training and validation feature data with subset filtering support.

    Args:
        train_path: Path to training NPZ file
        val_path: Path to validation NPZ file
        filename_filter: Optional filename to filter by
        use_features: Whether to load and use features
        use_train_subset: Filter train data by subset (True=train subset, False=test subset, None=all)
        use_val_subset: Filter val data by subset (True=train subset, False=test subset, None=all)
        data_augment_factor: Factor to repeat data for augmentation

    Returns:
        train_inputs, train_outputs, train_features, val_inputs, val_outputs, val_features
    """
    # Load training data
    train_inputs, train_outputs, train_inputs_features, _ = prepare_dataset(
        train_path,
        filter_filename=filename_filter,
        use_features=use_features,
        dataset_name="training",
        use_train_subset=use_train_subset,
    )

    # Load validation data
    val_inputs, val_outputs, val_inputs_features, _ = prepare_dataset(
        val_path,
        filter_filename=filename_filter,
        use_features=use_features,
        dataset_name="validation",
        use_train_subset=use_val_subset,
    )

    # Apply data augmentation if requested
    if data_augment_factor > 1:
        print(f"Applying data augmentation factor: {data_augment_factor}")
        train_inputs = train_inputs.repeat(data_augment_factor, 1, 1, 1)
        train_outputs = train_outputs.repeat(data_augment_factor, 1, 1, 1)
        if train_inputs_features is not None:
            train_inputs_features = train_inputs_features.repeat(
                data_augment_factor, 1, 1, 1
            )

    # Handle case where features are not available
    if train_inputs_features is None:
        train_inputs_features = torch.empty(train_inputs.shape[0], 30, 30, 0)
    if val_inputs_features is None:
        val_inputs_features = torch.empty(val_inputs.shape[0], 30, 30, 0)

    return (
        train_inputs,
        train_outputs,
        train_inputs_features,
        val_inputs,
        val_outputs,
        val_inputs_features,
    )


def load_feature_mapping_dataloader(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    filename_filter: Optional[str] = None,
    data_augment_factor: int = 1,
    use_train_subset: Optional[bool] = None,
) -> DataLoader[Any]:
    """
    Load feature mapping data from NPZ file with optional filename and subset filtering.

    This function is designed to replace the load_feature_data functions in the main_mapping modules.
    It loads features, masks, and indices in the format expected by feature mapping models.

    Args:
        npz_path: Path to NPZ file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        filename_filter: Filter to specific filename (without .json extension)
        data_augment_factor: Number of times to repeat the dataset per epoch
        use_train_subset: If True, use only "train" subset; if False, use only "test" subset; if None, use all

    Returns:
        DataLoader with batches containing (input_features, output_features, inputs_mask, outputs_mask, indices, subset_is_train)
    """
    print(f"Loading feature mapping data from {npz_path}")

    # Load all data including masks and indices
    (
        filenames,
        _,
        _,  # inputs (one-hot colors, not needed for feature mapping)
        _,  # outputs (one-hot colors, not needed for feature mapping)
        inputs_features,
        outputs_features,
        subset_is_train,
        inputs_mask,
        outputs_mask,
        indices_tensor,
    ) = load_npz_data(npz_path, use_features=True, load_masks_and_indices=True)

    # Check that we have the required data
    if inputs_features is None or outputs_features is None:
        raise ValueError(
            "Feature mapping requires features data, but it was not found in NPZ file"
        )

    if inputs_mask is None or outputs_mask is None:
        raise ValueError(
            "Feature mapping requires mask data, but it was not found in NPZ file"
        )

    if indices_tensor is None:
        raise ValueError(
            "Feature mapping requires indices data, but it was not found in NPZ file"
        )

    # Convert features to float if they aren't already
    inputs_features = inputs_features.float()
    outputs_features = outputs_features.float()

    # Load subset information if available
    subset_is_train_tensor = None
    if subset_is_train is not None:
        subset_is_train_tensor = torch.from_numpy(subset_is_train).bool()

    # Apply filename filter if specified
    if filename_filter is not None:
        # Create mask for matching filenames (with .json extension)
        filename_with_ext = f"{filename_filter}.json"
        mask = np.array([fname == filename_with_ext for fname in filenames])
        filter_indices = np.where(mask)[0]

        if len(filter_indices) == 0:
            print(f"  Warning: No examples found for filename: {filename_filter}")
            # Return empty dataloader
            empty_dataset = TensorDataset(
                torch.empty(0, 30, 30, inputs_features.shape[-1]),
                torch.empty(0, 30, 30, outputs_features.shape[-1]),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.bool),  # subset_is_train
            )
            return DataLoader(empty_dataset, batch_size=batch_size)

        # Filter all arrays
        inputs_features = inputs_features[filter_indices]
        outputs_features = outputs_features[filter_indices]
        inputs_mask = inputs_mask[filter_indices]
        outputs_mask = outputs_mask[filter_indices]
        indices_tensor = indices_tensor[filter_indices]
        if subset_is_train is not None:
            subset_is_train = subset_is_train[filter_indices]
            if subset_is_train_tensor is not None:
                subset_is_train_tensor = subset_is_train_tensor[filter_indices]

        print(f"  Filtered to filename: {filename_filter}")
        print(f"  Found {len(filter_indices)} examples")

    # Apply subset filtering if requested
    if use_train_subset is not None and subset_is_train is not None:
        # Create mask for the desired subset
        if use_train_subset:
            subset_mask = subset_is_train
            subset_name = "train"
        else:
            subset_mask = ~subset_is_train
            subset_name = "test"

        # Find indices that match the subset
        subset_indices = np.where(subset_mask)[0]

        if len(subset_indices) == 0:
            print(f"  Warning: No examples found in {subset_name} subset")
            # Return empty dataloader
            empty_dataset = TensorDataset(
                torch.empty(0, 30, 30, inputs_features.shape[-1]),
                torch.empty(0, 30, 30, outputs_features.shape[-1]),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, 30, 30, dtype=torch.bool),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.bool),  # subset_is_train
            )
            return DataLoader(empty_dataset, batch_size=batch_size)

        # Filter to subset
        inputs_features = inputs_features[subset_indices]
        outputs_features = outputs_features[subset_indices]
        inputs_mask = inputs_mask[subset_indices]
        outputs_mask = outputs_mask[subset_indices]
        indices_tensor = indices_tensor[subset_indices]
        if subset_is_train_tensor is not None:
            subset_is_train_tensor = subset_is_train_tensor[subset_indices]

        print(f"  Filtered to {subset_name} subset: {len(subset_indices)} examples")

    # Apply data augmentation by repeating the dataset
    if data_augment_factor > 1:
        inputs_features = inputs_features.repeat(data_augment_factor, 1, 1, 1)
        outputs_features = outputs_features.repeat(data_augment_factor, 1, 1, 1)
        inputs_mask = inputs_mask.repeat(data_augment_factor, 1, 1)
        outputs_mask = outputs_mask.repeat(data_augment_factor, 1, 1)
        indices_tensor = indices_tensor.repeat(data_augment_factor)
        if subset_is_train_tensor is not None:
            subset_is_train_tensor = subset_is_train_tensor.repeat(data_augment_factor)
        print(
            f"  Data augmented {data_augment_factor}x: {len(inputs_features)} total examples"
        )

    print(f"  Input features shape: {inputs_features.shape}")
    print(f"  Output features shape: {outputs_features.shape}")
    print(f"  Examples: {len(inputs_features)}")

    # Create dataset
    if subset_is_train_tensor is not None:
        dataset = TensorDataset(
            inputs_features,
            outputs_features,
            inputs_mask,
            outputs_mask,
            indices_tensor,
            subset_is_train_tensor,
        )
    else:
        # Create a dummy tensor for consistency
        dummy_subset = torch.zeros(len(inputs_features), dtype=torch.bool)
        dataset = TensorDataset(
            inputs_features,
            outputs_features,
            inputs_mask,
            outputs_mask,
            indices_tensor,
            dummy_subset,
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load and display information about processed NPZ data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="processed_data/train.npz",
        help="Path to the processed NPZ file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="025d127b.json",
        help="Filter examples by filename",
    )
    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Load and use concatenated features",
    )

    args = parser.parse_args()

    # Load the data
    (
        filenames,
        indices,
        inputs,
        outputs,
        inputs_features,
        outputs_features,
        subset_is_train,
    ) = load_npz_data(args.data_path, use_features=args.use_features)[
        :7
    ]  # Take only first 7 elements

    # Print shape information
    print(f"Number of examples: {len(filenames)}")
    print(f"Filenames: {len(filenames)} items")
    print(f"Indices: {len(indices)} items")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")
    if inputs_features is not None:
        print(f"Inputs invariant shape: {inputs_features.shape}")
        if outputs_features is not None:
            print(f"Outputs invariant shape: {outputs_features.shape}")

    # Print subset information if available
    if subset_is_train is not None:
        train_count = subset_is_train.sum()
        test_count = len(subset_is_train) - train_count
        print(f"Examples from 'train' subset: {train_count}")
        print(f"Examples from 'test' subset: {test_count}")

    # Late import of terminal_imshow
    from utils.terminal_imshow import imshow

    # Filter examples by filename if provided
    if args.filename:
        filtered_indices: List[int] = [
            i for i, fname in enumerate(filenames) if fname == args.filename
        ]

        if not filtered_indices:
            print(f"No examples found with filename: {args.filename}")
            exit(1)

        # Get the first matching example
        idx: int = filtered_indices[0]
        print(f"\nShowing example for {args.filename} (index {idx}):")

        # Display original input and output
        print("\nOriginal Input:")
        imshow(one_hot_to_categorical(inputs[idx]))
        print("\nOriginal Output:")
        imshow(one_hot_to_categorical(outputs[idx]))

        # Display invariant features if available
        if inputs_features is not None:
            print(f"\nInvariant features for this example:")
            features = inputs_features[idx]
            print(f"Features shape: {features.shape}")
            print(
                f"Feature value range: {features.min().item():.2f} to {features.max().item():.2f}"
            )

            # Show some feature statistics for the center cell
            center_features = features[15, 15]
            print(f"Center cell (15,15) features:")
            print(f"  Pairwise features (first 5): {center_features[:5].numpy()}")  # type: ignore
            print(f"  Mask features (last 8): {center_features[36:44].numpy()}")  # type: ignore
