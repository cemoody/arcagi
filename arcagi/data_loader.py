from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_parquet_data(
    parquet_path: str,
) -> Tuple[List[str], List[int], torch.Tensor, torch.Tensor]:
    """
    Loads data from a parquet file created by preprocess.py and creates PyTorch arrays.
    """
    # Load the parquet file
    df: pd.DataFrame = pd.read_parquet(parquet_path)

    # Extract filenames and indices
    filenames: List[str] = df["filename"].tolist()
    indices: List[int] = df["index"].tolist()

    # Get number of examples
    num_examples: int = len(df)

    # Initialize raw tensors
    inputs_raw: torch.Tensor = torch.full((num_examples, 30, 30), -1, dtype=torch.int)
    outputs_raw: torch.Tensor = torch.full((num_examples, 30, 30), -1, dtype=torch.int)

    # Fill tensors with values from DataFrame
    for i in range(30):
        for j in range(30):
            inputs_raw[:, i, j] = torch.tensor(
                df[f"input_{i}_{j}"].values, dtype=torch.int
            )
            outputs_raw[:, i, j] = torch.tensor(
                df[f"output_{i}_{j}"].values, dtype=torch.int
            )

    # Create one-hot encoded tensors for colors 0-10
    inputs: torch.Tensor = torch.zeros((num_examples, 30, 30, 11), dtype=torch.float)
    outputs: torch.Tensor = torch.zeros((num_examples, 30, 30, 11), dtype=torch.float)

    # Fill in one-hot encoded values
    for color in range(10):
        inputs[:, :, :, color] = (inputs_raw == color).float()
        outputs[:, :, :, color] = (outputs_raw == color).float()
    inputs[:, :, :, 10] = (inputs_raw == -1).float()
    outputs[:, :, :, 10] = (outputs_raw == -1).float()

    return filenames, indices, inputs, outputs


def generate_color_mapping(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random one-to-one mapping for colors 0-10 (excluding -1) and applies it to the input tensors.
    """
    assert inputs.shape == outputs.shape
    assert inputs.shape[0] == 30
    assert inputs.shape[1] == 30

    # Turn inputs into a one-hot 3 dimensional array

    # Generate a random permutation of colors 0 to 10
    colors: torch.Tensor = torch.arange(11)
    color_mapping: torch.Tensor = colors[torch.randperm(11)]

    # Initialize new tensors
    new_inputs: torch.Tensor = torch.zeros_like(inputs)
    new_outputs: torch.Tensor = torch.zeros_like(outputs)

    # Permute the color channels - efficient vectorized operation
    for old_color in range(11):
        new_color = int(color_mapping[old_color].item())
        new_inputs[:, :, new_color] = inputs[:, :, old_color]
        new_outputs[:, :, new_color] = outputs[:, :, old_color]
    return new_inputs, new_outputs


def one_hot_to_categorical(
    one_hot_tensor: torch.Tensor, last_value: int = 10
) -> torch.Tensor:
    # Get the indices of the maximum values along the last dimension
    cat: torch.Tensor = torch.argmax(one_hot_tensor, dim=-1)
    sum: torch.Tensor = one_hot_tensor.sum(dim=-1)
    cat[sum <= 1e-6] = last_value
    return cat


# ---------------------------------------------------------------------------
# Color permutation utilities
# ---------------------------------------------------------------------------


def batch_generate_color_mapping(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random permutation of the 11 colour channels (0-10)
    to *each* example in a batch.

    The permutation for a given sample is shared between its ``inputs`` and
    ``outputs`` so that the mapping is aligned.  All permutations are mutually
    independent across the batch.

    Args:
        inputs:  A tensor of shape ``(B, H, W, 11)`` containing one-hot encoded
            colours for *B* samples.
        outputs: Tensor with the same shape as ``inputs``.

    Returns:
        Tuple containing the permuted ``inputs`` and ``outputs`` (same shape as
        the originals).
    """

    # Basic validation – we expect one-hot encoded channels on the last axis.
    assert (
        inputs.shape == outputs.shape
    ), "inputs and outputs must have identical shapes"
    assert inputs.dim() == 4, "Expected inputs with shape (B, H, W, 11)"
    batch_size: int
    height: int
    width: int
    channels: int
    batch_size, height, width, channels = inputs.shape
    assert channels == 11, "Colour channel dimension (last dim) must be 11"

    # ---------------------------------------------------------------------
    # Generate a *vectorised* set of independent random permutations.
    # ---------------------------------------------------------------------
    # Draw random values and argsort – this produces a random permutation for
    # every sample without an explicit Python loop, keeping the operation on
    # the GPU when available.
    perm: torch.Tensor = torch.argsort(
        torch.rand(batch_size, channels, device=inputs.device), dim=1
    )

    # ---------------------------------------------------------------------
    # Re-index the colour channel using ``torch.gather``.
    # ---------------------------------------------------------------------
    # Move channel axis to dim=1 for easier gathering.
    inputs_bchw: torch.Tensor = inputs.permute(0, 3, 1, 2)  # (B, C, H, W)
    outputs_bchw: torch.Tensor = outputs.permute(0, 3, 1, 2)

    # Expand perm so it can index the (H, W) spatial dimensions.
    perm_expanded: torch.Tensor = perm[:, :, None, None].expand(-1, -1, height, width)

    # Gather along the channel dimension.
    permuted_inputs_bchw: torch.Tensor = torch.gather(inputs_bchw, 1, perm_expanded)
    permuted_outputs_bchw: torch.Tensor = torch.gather(outputs_bchw, 1, perm_expanded)

    # Restore original layout (B, H, W, C).
    permuted_inputs: torch.Tensor = permuted_inputs_bchw.permute(0, 2, 3, 1)
    permuted_outputs: torch.Tensor = permuted_outputs_bchw.permute(0, 2, 3, 1)

    return permuted_inputs, permuted_outputs


def repeat_and_permute(
    inputs: torch.Tensor, outputs: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Repeats the input and output tensors n times along the batch dimension and applies
    independent color permutations to each sample.

    Args:
        inputs: A tensor of shape (B, 30, 30, 11) containing one-hot encoded colors.
        outputs: A tensor with the shape (B * n, 30, 30, 11) containing permuted inputs
        n: Number of times to repeat each sample.

    Returns:
        Tuple containing the repeated and permuted inputs and outputs with shape (B*n, 30, 30, 11).
    """
    # Repeat each sample n times
    repeated_inputs: torch.Tensor = inputs.repeat(n, 1, 1, 1)
    repeated_outputs: torch.Tensor = outputs.repeat(n, 1, 1, 1)

    # Apply independent color permutations to each sample
    permuted_inputs, permuted_outputs = batch_generate_color_mapping(
        repeated_inputs, repeated_outputs
    )

    return permuted_inputs, permuted_outputs


def apply_mixing_steps(
    inputs: torch.Tensor, outputs: torch.Tensor, n_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply mixing steps to create intermediate states between inputs and outputs.

    For each sample, randomly picks a fraction f from {0, 1/n, 2/n, ..., (n-1)/n}.
    Creates new_input = input * f + output * (1 - f)
    Creates new_output = input * (f - 1/n) + output * (1 - f + 1/n)

    Args:
        inputs: A tensor of shape (B, 30, 30, 11) containing one-hot encoded colors.
        outputs: A tensor of shape (B, 30, 30, 11) containing one-hot encoded colors.
        n_steps: Number of mixing steps (n).

    Returns:
        Tuple containing the mixed inputs and outputs with the same shape.
    """
    batch_size = inputs.shape[0]

    # Generate random fractions for each sample
    # f can be 0, 1/n, 2/n, ..., (n-1)/n
    step_indices = torch.randint(0, n_steps, (batch_size,), device=inputs.device)
    f = step_indices.float() / n_steps  # Shape: (B,)

    # Reshape f for broadcasting
    f = f.view(batch_size, 1, 1, 1)  # Shape: (B, 1, 1, 1)

    # Calculate new inputs: input * f + output * (1 - f)
    new_inputs = inputs * f + outputs * (1 - f)

    # Calculate new outputs: input * (f - 1/n) + output * (1 - f + 1/n)
    # Note: When f = 0, this becomes input * (-1/n) + output * (1 + 1/n)
    # We need to clamp to ensure valid probabilities
    f_minus = f - 1.0 / n_steps
    new_outputs = inputs * f_minus + outputs * (1 - f_minus)

    # Ensure outputs remain valid probability distributions
    # (though the math should preserve this if inputs and outputs are valid)
    new_outputs = torch.clamp(new_outputs, min=0.0, max=1.0)

    return new_inputs, new_outputs


# ---------------------------------------------------------------------------
# DataLoader creation utilities
# ---------------------------------------------------------------------------


def filter_by_filename(
    filenames: List[str],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    target_filename: str,
    dataset_name: str = "data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter tensors to only include examples from a specific filename.

    Args:
        filenames: List of filenames for each example
        inputs: Input tensor to filter
        outputs: Output tensor to filter
        target_filename: Filename to filter by
        dataset_name: Name of dataset for logging (e.g., "training", "validation")

    Returns:
        Filtered inputs and outputs tensors

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
            return torch.empty(0, 30, 30, 11), torch.empty(0, 30, 30, 11)
        else:
            raise ValueError(
                f"No {dataset_name} examples found with filename: {target_filename}"
            )

    # Filter the tensors
    filtered_inputs = inputs[filtered_indices]
    filtered_outputs = outputs[filtered_indices]
    print(
        f"Found {len(filtered_indices)} {dataset_name} examples with filename {target_filename}"
    )

    return filtered_inputs, filtered_outputs


def prepare_dataset(
    parquet_path: str,
    filter_filename: Optional[str] = None,
    limit_examples: Optional[int] = None,
    augment_factor: int = 1,
    use_mixing_steps: Optional[int] = None,
    dataset_name: str = "data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and prepare a dataset from a parquet file with optional filtering and augmentation.

    Args:
        parquet_path: Path to the parquet file
        filter_filename: Optional filename to filter examples
        limit_examples: Optional limit on number of examples
        augment_factor: Factor for data augmentation via color permutation
        use_mixing_steps: Optional number of mixing steps for creating intermediate states
        dataset_name: Name of dataset for logging

    Returns:
        Prepared inputs and outputs tensors
    """
    print(f"Loading {dataset_name} from {parquet_path}...")
    filenames, _, inputs, outputs = load_parquet_data(parquet_path)

    # Apply filtering if requested
    if filter_filename is not None:
        inputs, outputs = filter_by_filename(
            filenames, inputs, outputs, filter_filename, dataset_name
        )

    # Apply limit if requested
    if limit_examples is not None:
        inputs = inputs[:limit_examples]
        outputs = outputs[:limit_examples]
        print(f"Limited to {limit_examples} {dataset_name} examples")

    # Apply augmentation if requested
    if augment_factor > 1:
        print(f"Applying {augment_factor}x augmentation to {dataset_name}...")
        inputs, outputs = repeat_and_permute(inputs, outputs, augment_factor)

    # Apply mixing steps if requested
    if use_mixing_steps is not None:
        print(f"Applying mixing steps with n={use_mixing_steps} to {dataset_name}...")
        inputs, outputs = apply_mixing_steps(inputs, outputs, use_mixing_steps)

    print(
        f"{dataset_name.capitalize()} shape: inputs={inputs.shape}, outputs={outputs.shape}"
    )

    return inputs, outputs


def create_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader[Any]:
    """Create a DataLoader from input and output tensors.

    Args:
        inputs: Input tensor
        outputs: Output tensor
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
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
        description="Load and display information about processed data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="processed_data/train.parquet",
        help="Path to the processed parquet file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="025d127b.json",
        help="Filter examples by filename",
    )
    args = parser.parse_args()

    # Load the data
    filenames, indices, inputs, outputs = load_parquet_data(args.data_path)

    binputs, boutputs = repeat_and_permute(inputs, outputs, 5)

    # Print shape information
    print(f"Number of examples: {len(filenames)}")
    print(f"Filenames: {len(filenames)} items")
    print(f"Indices: {len(indices)} items")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")

    # Late import of terminal_imshow
    from utils.terminal_imshow import imshow

    # Filter examples by filename if provided
    if args.filename:
        filtered_indices: list[int] = [
            i for i, fname in enumerate(filenames) if fname == args.filename
        ]

        # Get the first matching example
        idx: int = filtered_indices[0]
        print(f"\nShowing example for {args.filename} (index {idx}):")

        # Display original input and output
        print("\nOriginal Input:")

        imshow(one_hot_to_categorical(inputs[idx]))
        print("\nOriginal Output:")
        imshow(one_hot_to_categorical(outputs[idx]))

        # Find the corresponding indices in the permuted batch
        # Each original example is repeated n times (5 in this case)
        batch_indices: list[int] = [idx + i * len(filenames) for i in range(5)]

        # Display all permuted versions
        for i, batch_idx in enumerate(batch_indices):
            print(f"\nPermuted version {i+1}:")
            print("Input:")
            imshow(one_hot_to_categorical(binputs[batch_idx]))
            print("Output:")
            imshow(one_hot_to_categorical(boutputs[batch_idx]))
