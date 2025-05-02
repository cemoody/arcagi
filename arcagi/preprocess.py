from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


def expand_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Expands a given 2D list (matrix) to a fixed 30x30 matrix.
    The original matrix is centered within the 30x30 grid.
    Positions beyond the size of the original matrix are filled with -1.
    """
    # Create empty 30x30 matrix filled with -1
    new_matrix: List[List[int]] = [[-1 for _ in range(30)] for _ in range(30)]

    # Calculate dimensions of input matrix
    height: int = len(matrix)
    width: int = len(matrix[0]) if height > 0 else 0

    # Calculate starting positions to center the matrix
    start_row: int = (30 - height) // 2
    start_col: int = (30 - width) // 2

    # Copy input matrix to centered position in the new matrix
    for i, row in enumerate(matrix):
        if i >= height:
            break
        for j, val in enumerate(row):
            if j >= width:
                break
            new_matrix[start_row + i][start_col + j] = val

    return new_matrix


def load_data_from_directory(
    directory: str, subset: str, filename_filter: Optional[str] = None
) -> Tuple[List[str], List[int], torch.Tensor, torch.Tensor]:
    """
    Loads JSON files from a given directory.
    For each JSON file, it reads the list of examples under the key `subset`
    (e.g., "train" or "test"), expands the input and output matrices,
    and aggregates the data.
    """
    filenames: List[str] = []
    indices: List[int] = []
    inputs_expanded: List[List[List[int]]] = []  # Each inner element is 30x30.
    outputs_expanded: List[List[List[int]]] = []

    path: Path = Path(directory)
    json_files: List[Path] = sorted(list(path.glob("*.json")))

    for json_file in tqdm(json_files):
        with json_file.open("r") as f:
            data: Dict[str, Any] = json.load(f)
        if subset not in data:
            raise ValueError(f"File {json_file.name} does not have the '{subset}' key.")
        examples: List[Dict[str, Any]] = data[subset]
        for idx, example in enumerate(examples):
            filenames.append(json_file.name)
            indices.append(idx)
            input_colors: List[List[int]] = example["input"]
            output_colors: List[List[int]] = example["output"]
            inputs_expanded.append(expand_matrix(input_colors))
            outputs_expanded.append(expand_matrix(output_colors))

    # Filter by filename if specified
    if filename_filter is not None:
        # Create mask for matching filenames
        mask: List[bool] = [filename == filename_filter for filename in filenames]

        # Apply mask to filter data
        filenames = [filenames[i] for i in range(len(filenames)) if mask[i]]
        indices = [indices[i] for i in range(len(indices)) if mask[i]]

        # Filter expanded inputs and outputs before converting to tensors
        inputs_expanded = [
            inputs_expanded[i] for i in range(len(inputs_expanded)) if mask[i]
        ]
        outputs_expanded = [
            outputs_expanded[i] for i in range(len(outputs_expanded)) if mask[i]
        ]

    inputs_tensor: torch.Tensor = torch.tensor(inputs_expanded, dtype=torch.int)
    outputs_tensor: torch.Tensor = torch.tensor(outputs_expanded, dtype=torch.int)

    return filenames, indices, inputs_tensor, outputs_tensor


def save_data_to_parquet(
    directory: str,
    output_path: str,
    subset: str = "train",
    filename_filter: Optional[str] = None,
) -> None:
    """
    Loads data using load_data_from_directory and saves it as a parquet file.
    All inputs and outputs tensors are guaranteed to be 30x30 2D arrays.

    Args:
        directory: Directory containing the JSON files
        output_path: Path to save the parquet file
        subset: Subset of data to load ("train", "test", etc.)
        filename_filter: Optional filter for specific filenames
    """
    import pandas as pd
    import numpy as np

    # Load the data
    filenames, indices, inputs_tensor, outputs_tensor = load_data_from_directory(
        directory, subset, filename_filter
    )

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(
        {
            "filename": filenames,
            "index": indices,
        }
    )

    # Convert tensors to numpy arrays
    inputs_np: np.ndarray = inputs_tensor.numpy()  # type: ignore
    outputs_np: np.ndarray = outputs_tensor.numpy()  # type: ignore

    # Create input columns dictionary
    input_dict: Dict[str, np.ndarray] = {}  # type: ignore
    for i in range(30):
        for j in range(30):
            col_name = f"input_{i}_{j}"
            input_dict[col_name] = inputs_np[:, i, j]

    # Create output columns dictionary
    output_dict: Dict[str, np.ndarray] = {}  # type: ignore
    for i in range(30):
        for j in range(30):
            col_name = f"output_{i}_{j}"
            output_dict[col_name] = outputs_np[:, i, j]

    # Create separate DataFrames
    input_df = pd.DataFrame(input_dict)
    output_df = pd.DataFrame(output_dict)

    # Concatenate all DataFrames at once
    df = pd.concat([metadata_df, input_df, output_df], axis=1)

    # Save as parquet
    df.to_parquet(output_path)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Preprocess ARC-AGI data and save to parquet files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ARC-AGI/data",
        help="Directory containing the ARC-AGI data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Directory to save the parquet files",
    )
    args = parser.parse_args()

    # Create paths
    data_dir: Path = Path(args.data_dir)
    output_dir: Path = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths for training and evaluation data
    train_dir: str = str(data_dir / "training")
    eval_dir: str = str(data_dir / "evaluation")

    train_output: str = str(output_dir / "train.parquet")
    eval_output: str = str(output_dir / "eval.parquet")

    # Check if directories exist
    print(f"Processing training data from {train_dir}")
    print(f"Saving to {train_output}")
    save_data_to_parquet(train_dir, train_output, subset="train")
    print("Training data saved successfully")

    print(f"Processing evaluation data from {eval_dir}")
    print(f"Saving to {eval_output}")
    save_data_to_parquet(eval_dir, eval_output, subset="test")
    print("Evaluation data saved successfully")
