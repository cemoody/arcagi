import pandas as pd
import torch
from typing import Tuple, List


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
    for color in range(11):
        inputs[:, :, :, color] = (inputs_raw == color).float()
        outputs[:, :, :, color] = (outputs_raw == color).float()

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


def one_hot_to_categorical(one_hot_tensor: torch.Tensor) -> torch.Tensor:
    # Get the indices of the maximum values along the last dimension
    cat: torch.Tensor = torch.argmax(one_hot_tensor, dim=-1)
    sum: torch.Tensor = one_hot_tensor.sum(dim=-1)
    cat[sum == 0] = -1
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
