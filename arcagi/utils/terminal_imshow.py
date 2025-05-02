#!/usr/bin/env python3
import torch
from typing import Optional, List, Set
import sys
from pathlib import Path


def imshow(
    matrix: torch.Tensor, title: Optional[str] = None, show_legend: bool = True
) -> None:
    """
    Prints a 2D PyTorch tensor as a colored grid to the terminal.
    Each integer in the tensor is mapped to a distinct color.

    Args:
        matrix: A 2D PyTorch tensor of integers, where each integer represents a color.
        title: Optional title to display above the image.
        show_legend: Whether to show a legend mapping colors to values.
    """
    # Ensure matrix is 2D
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {matrix.dim()}D tensor")

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

    # Reset code
    reset = "\033[0m"

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
    for row in matrix_list:
        row_str = ""
        for val in row:
            if isinstance(val, int) and val == -1:
                # Background (use space with black background)
                # Using 3 spaces to match the width of " NN " format
                row_str += f"\033[40m   \033[0m"
            else:
                # Get the appropriate color based on value
                val_int = int(val)
                color_idx = val_int % len(colors)
                # Format to ensure consistent width with spaces on both sides
                # Single digit numbers: " N " (3 chars), double digits: "NN " (3 chars)
                if val_int < 10:
                    row_str += f"{colors[color_idx]} {val_int} {reset}"
                else:
                    row_str += f"{colors[color_idx]}{val_int} {reset}"

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
    from arcagi.data_loader import JSONDataModule

    # Look for the data directory
    possible_data_dirs = [
        project_root / "ARC-AGI" / "data" / "training",
        project_root / "ARC-AGI-2" / "data" / "training",
        project_root / "data" / "training",
    ]

    train_dir = None
    for data_dir in possible_data_dirs:
        if data_dir.exists():
            train_dir = str(data_dir)
            break

    # Check if directory exists
    if train_dir is not None:
        print(f"\nFound ARC dataset at: {train_dir}")
        data_module = JSONDataModule(train_dir=train_dir, batch_size=1)
        data_module.setup()

        # Get a batch from the dataloader
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        # Display input and output
        input_tensor = batch["input_colors_expanded"][0]
        output_tensor = batch["output_colors_expanded"][0]

        print("\n\033[1m========== Real ARC Example ==========\033[0m")
        print(
            f"\033[1mFrom file: {batch['filename'][0]}, Example {batch['example_index'][0]}\033[0m"
        )

        imshow(input_tensor, title="Input")
        imshow(output_tensor, title="Expected Output")
    else:
        print("\nARC dataset not found. Skipping real example visualization.")
        print("Expected to find the dataset in one of:")
        for path in possible_data_dirs:
            print(f"  - {path}")
