#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Optional, Set

import torch


def imshow(
    matrix: torch.Tensor,
    title: Optional[str] = None,
    show_legend: bool = True,
    correct: Optional[torch.Tensor] = None,
) -> None:
    """
    Prints a 2D PyTorch tensor as a colored grid to the terminal.
    Each integer in the tensor is mapped to a distinct color.

    Args:
        matrix: A 2D PyTorch tensor of integers, where each integer represents a color.
        title: Optional title to display above the image.
        show_legend: Whether to show a legend mapping colors to values.
        correct: Optional 2D boolean tensor of same shape as matrix.
                If provided, incorrect predictions (False values) will be displayed in bold.
    """
    # Ensure matrix is 2D
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {matrix.dim()}D tensor")

    # Validate correct array if provided
    if correct is not None:
        if correct.shape != matrix.shape:
            raise ValueError(
                f"correct array shape {correct.shape} must match matrix shape {matrix.shape}"
            )
        if correct.dim() != 2:
            raise ValueError(f"Expected 2D correct tensor, got {correct.dim()}D tensor")
        # Convert to CPU and numpy for processing
        correct_np = correct.cpu().numpy()  # type: ignore
        correct_list = correct_np.tolist()
    else:
        correct_list = None

    # Convert to CPU numpy array and then to list
    matrix_np = matrix.cpu().numpy()  # type: ignore
    matrix_list = matrix_np.tolist()

    # ANSI color codes (background colors)
    colors: List[str] = [
        "\033[40m",  # Black background
        "\033[41m",  # Red background
        "\033[42m",  # Green background
        "\033[43m",  # Yellow background
        "\033[44m",  # Blue background
        "\033[45m",  # Magenta background
        "\033[46m",  # Cyan background
        "\033[47m",  # White background
        "\033[100m",  # Bright black background
        "\033[101m",  # Bright red background
        "\033[102m",  # Bright green background
        "\033[103m",  # Bright yellow background
        "\033[104m",  # Bright blue background
        "\033[105m",  # Bright magenta background
        "\033[106m",  # Bright cyan background
        "\033[107m",  # Bright white background
    ]

    # Reset code and formatting codes
    reset = "\033[0m"
    bold = "\033[1m"
    underline = "\033[4m"

    # Print title if provided
    if title:
        print(f"\n\033[1m{title}\033[0m")

    # Find all unique values in the matrix for legend
    unique_values: Set[int] = set()
    for row in matrix_list:
        for val in row:
            if isinstance(val, int) and val != -1:  # Ignore background
                unique_values.add(val)
    # Print each row of the matrix
    for row_idx, row in enumerate(matrix_list):
        row_str = ""
        for col_idx, val in enumerate(row):
            # Check if this prediction is incorrect (for special formatting)
            is_incorrect = (
                correct_list is not None and not correct_list[row_idx][col_idx]
            )

            if isinstance(val, int) and val == -1:
                # Background (use space with black background)
                # Using 3 spaces to match the width of " NN " format
                if is_incorrect:
                    row_str += f"{bold}{underline}\033[40m    {reset}"
                else:
                    row_str += f"\033[40m    {reset}"
            else:
                # Get the appropriate color based on value
                val_int = int(val)
                abs_val = abs(val_int)
                color_idx = abs_val % len(colors)

                # Create the display value with sign if negative
                if val_int < 0:
                    display_val = f"-{abs_val:>1}"
                else:
                    display_val = f" {abs_val:>1}"

                # Format to ensure consistent width (3 characters total)
                if is_incorrect:
                    # Use brackets around incorrect predictions for clear visibility
                    row_str += f"{colors[color_idx]}[{display_val:>2}]{reset}"
                else:
                    row_str += f"{colors[color_idx]} {display_val:>2} {reset}"
        print(row_str)

    # Display legend if requested
    if show_legend and unique_values:
        print("\n\033[1mLegend:\033[0m")
        legend_str = ""
        for val in sorted(unique_values):
            color_idx = val % len(colors)
            if val < 10:
                legend_str += f"{colors[color_idx]} {val} {reset} "
            else:
                legend_str += f"{colors[color_idx]}{val} {reset} "
        print(legend_str)


if __name__ == "__main__":
    # Test with random color grid
    random_grid = torch.randint(0, 10, (10, 10))
    imshow(random_grid, title="Random Color Grid (10x10)")

    # Test with a checkerboard pattern
    size: int = 8
    checkerboard = torch.zeros((size, size), dtype=torch.int)
    for i in range(size):
        for j in range(size):
            checkerboard[i, j] = (i + j) % 2
    imshow(checkerboard, title="Checkerboard Pattern (8x8)")

    # Test with a gradient pattern
    gradient = torch.zeros((15, 15), dtype=torch.int)
    for i in range(15):
        for j in range(15):
            gradient[i, j] = (i + j) % 16
    imshow(gradient, title="Color Gradient (15x15)")

    # Test with -1 values (should be rendered as empty space)
    with_background = torch.ones((10, 10), dtype=torch.int)
    with_background[3:7, 3:7] = 2
    with_background[0:10:3, 0:10:3] = -1
    imshow(with_background, title="Pattern with Background (-1 values)")

    # Test with correct array - some predictions are incorrect (bold)
    print("\n" + "=" * 60)
    print("Testing correct array functionality (bold = incorrect)")
    print("=" * 60)

    test_matrix = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1], [2, 3, 4, 5]], dtype=torch.int
    )

    # Mark some predictions as incorrect (these will be bold)
    correct_array = torch.tensor(
        [
            [True, False, True, False],  # Alternating correct/incorrect
            [False, True, False, True],  # Alternating incorrect/correct
            [True, True, False, False],  # Mixed pattern
            [False, False, True, True],  # Mixed pattern
        ]
    )

    imshow(
        test_matrix,
        title="With Correct Array (bold = incorrect predictions)",
        correct=correct_array,
    )

    # Show the same matrix without correct array for comparison
    imshow(test_matrix, title="Same Matrix Without Correct Array (normal display)")

    # Test with a larger 30x30 matrix (similar to what's used in the ARC dataset)
    large_matrix = torch.full((30, 30), -1, dtype=torch.int)
    # Draw a simple pattern in the center
    large_matrix[10:20, 10:20] = 3
    large_matrix[13:17, 13:17] = 5
    large_matrix[14:16, 14:16] = 7
    imshow(large_matrix, title="30x30 Matrix with Pattern in Center")

    # Try to load and display a real ARC example if the data_loader is available
    # Add parent directory to path so we can import data_loader
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

    try:
        from arcagi.data_loader import create_dataloader, load_npz_data

        data_loader_available = True
    except ImportError:
        print(
            "\nData loader functions not available. Skipping real ARC example visualization."
        )
        data_loader_available = False

    if data_loader_available:
        # Look for the data directory with processed NPZ files
        possible_data_dirs = [
            project_root / "processed_data",
            project_root / "processed_data_v2",
            project_root / "data",
        ]

        data_file = None
        for data_dir in possible_data_dirs:
            if data_dir.exists():
                # Look for NPZ files in the directory
                npz_files = list(data_dir.glob("*.npz"))
                if npz_files:
                    data_file = str(npz_files[0])  # Use the first NPZ file found
                    break

        # Check if we found processed data
        if data_file is not None:
            print(f"\nFound processed ARC data at: {data_file}")
            try:
                # Load the data
                (
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
                ) = load_npz_data(data_file)

                # Create a simple dataloader
                dataloader = create_dataloader(
                    inputs, outputs, batch_size=1, shuffle=False
                )

                # Get first batch
                batch_data = next(iter(dataloader))
                input_tensor = batch_data[0][0]  # First input from first batch
                output_tensor = batch_data[1][0]  # First output from first batch

                # Convert from one-hot back to color indices for display
                input_colors = torch.argmax(input_tensor, dim=-1)
                output_colors = torch.argmax(output_tensor, dim=-1)

                print("\n\033[1m========== Real ARC Example ==========\033[0m")
                print(f"\033[1mFrom file: {filenames[0]}, Example {indices[0]}\033[0m")

                imshow(input_colors, title="Input")
                imshow(output_colors, title="Expected Output")

            except Exception as e:
                print(f"\nError loading ARC data: {e}")
                print("Skipping real example visualization.")
        else:
            print(
                "\nProcessed ARC data not found. Skipping real example visualization."
            )
            print("Expected to find NPZ files in one of:")
            for path in possible_data_dirs:
                print(f"  - {path}")
    else:
        print(
            "Consider running preprocessing to create NPZ files for real ARC examples."
        )
