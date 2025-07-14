from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset


def load_npz_data(
    npz_path: str,
    use_features: bool = False,
) -> Tuple[
    List[str],
    List[int],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Loads data from an NPZ file created by preprocess_v2.py and creates PyTorch tensors.

    Args:
        npz_path: Path to the NPZ file
        use_features: Whether to load concatenated features (if available)

    Returns:
        filenames: List of filename strings
        indices: List of example indices within each file
        inputs: torch.Tensor of shape (n_examples, 30, 30, 11) with one-hot encoded colors
        outputs: torch.Tensor of shape (n_examples, 30, 30, 11) with one-hot encoded colors
        inputs_features: torch.Tensor of shape (n_examples, 30, 30, n_features) with binary features (if use_features=True and available)
        outputs_features: torch.Tensor of shape (n_examples, 30, 30, n_features) with binary features (if use_features=True and available)
    """
    # Load the NPZ file
    data = np.load(npz_path)

    # Extract basic data
    filenames: List[str] = data["filenames"].tolist()
    indices: List[int] = data["indices"].tolist()
    inputs_raw: NDArray[np.int32] = data["inputs"]  # Shape: (n_examples, 30, 30)
    outputs_raw: NDArray[np.int32] = data["outputs"]  # Shape: (n_examples, 30, 30)

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

    return filenames, indices, inputs, outputs, inputs_features, outputs_features


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


def prepare_dataset(
    npz_path: str,
    filter_filename: Optional[str] = None,
    limit_examples: Optional[int] = None,
    use_features: bool = False,
    dataset_name: str = "data",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load and prepare a dataset from an NPZ file with optional filtering.

    Args:
        npz_path: Path to the NPZ file
        filter_filename: Optional filename to filter examples
        limit_examples: Optional limit on number of examples
        use_features: Whether to load invariant features
        dataset_name: Name of dataset for logging

    Returns:
        Prepared inputs, outputs, and optionally invariant tensors
    """
    print(f"Loading {dataset_name} from {npz_path}...")
    filenames, _, inputs, outputs, inputs_features, outputs_features = load_npz_data(
        npz_path, use_features=use_features
    )

    # Apply filtering if requested
    if filter_filename is not None:
        inputs, outputs, inputs_features, outputs_features = filter_by_filename(
            filenames,
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
    filenames, indices, inputs, outputs, inputs_features, outputs_features = (
        load_npz_data(args.data_path, use_features=args.use_features)
    )

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
