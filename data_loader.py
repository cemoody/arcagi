#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, Sized

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm


def expand_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Expands a given 2D list (matrix) to a fixed 30x30 matrix.
    Positions beyond the size of the original matrix are filled with -1.
    """
    new_matrix: List[List[int]] = [[-1 for _ in range(30)] for _ in range(30)]
    for i, row in enumerate(matrix):
        if i >= 30:
            break
        for j, val in enumerate(row):
            if j >= 30:
                break
            new_matrix[i][j] = val
    return new_matrix


def load_data_from_directory(
    directory: str, subset: str
) -> Tuple[List[str], List[int], torch.Tensor, torch.Tensor]:
    """
    Loads JSON files from a given directory.
    For each JSON file, it reads the list of examples under the key `subset`
    (e.g., "train" or "test"), expands the input and output matrices,
    and aggregates the data.

    Returns:
        filenames: List of filenames (one per example).
        indices: List of example indices (per file).
        inputs_tensor: A 3D tensor of expanded input colors (N, 30, 30).
        outputs_tensor: A 3D tensor of expanded output colors (N, 30, 30).
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

    inputs_tensor: torch.Tensor = torch.tensor(inputs_expanded, dtype=torch.int)
    outputs_tensor: torch.Tensor = torch.tensor(outputs_expanded, dtype=torch.int)

    return filenames, indices, inputs_tensor, outputs_tensor


class JSONDataset(Dataset[Dict[str, Any]]):
    """
    A PyTorch Dataset that holds data (filenames, example indices,
    expanded input colors, and expanded output colors) loaded from JSON files.
    """

    def __init__(
        self,
        filenames: List[str],
        indices: List[int],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        self.filenames: List[str] = filenames
        self.indices: List[int] = indices
        self.inputs: torch.Tensor = inputs
        self.outputs: torch.Tensor = outputs

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "filename": self.filenames[idx],
            "example_index": self.indices[idx],
            "input_colors_expanded": self.inputs[idx],
            "output_colors_expanded": self.outputs[idx],
        }


class JSONDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that loads JSON data from directories.
    It supports separate directories for training and validation.
    """

    def __init__(
        self,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.train_dir: Optional[str] = train_dir
        self.val_dir: Optional[str] = val_dir
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_dataset: Optional[Dataset[Dict[str, Any]]] = None
        self.val_dataset: Optional[Dataset[Dict[str, Any]]] = None

        # Define which subset key to load for each dataset.
        self.train_subset: str = "train"
        self.val_subset: str = (
            "test"  # Change this if your validation key is named differently.
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dir is not None:
            filenames, indices, inputs, outputs = load_data_from_directory(
                self.train_dir, self.train_subset
            )
            self.train_dataset = JSONDataset(filenames, indices, inputs, outputs)
        if self.val_dir is not None:
            filenames, indices, inputs, outputs = load_data_from_directory(
                self.val_dir, self.val_subset
            )
            self.val_dataset = JSONDataset(filenames, indices, inputs, outputs)

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        if self.train_dataset is None:
            raise ValueError(
                "Train dataset is not loaded. Ensure 'train_dir' is provided and 'setup()' has been called."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset is not loaded. Ensure 'val_dir' is provided and 'setup()' has been called."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# Example usage
if __name__ == "__main__":

    # Set paths to data directories
    train_dir: str = "ARC-AGI/data/training"
    val_dir: str = "ARC-AGI/data/evaluation"

    # Check if directories exist
    if not Path(train_dir).exists():
        print(f"Warning: Training directory {train_dir} does not exist")
    if not Path(val_dir).exists():
        print(f"Warning: Evaluation directory {val_dir} does not exist")

    # Create data module
    data_module = JSONDataModule(
        train_dir=train_dir, val_dir=val_dir, batch_size=32, num_workers=0
    )

    # Setup data module
    data_module.setup()

    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Print dataset information
    assert data_module.train_dataset is not None, "Train dataset not loaded"
    assert data_module.val_dataset is not None, "Validation dataset not loaded"
    print(f"Train dataset size: {len(cast(Sized, data_module.train_dataset))}")
    print(f"Validation dataset size: {len(cast(Sized, data_module.val_dataset))}")

    # Get a batch from each loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    # Print batch information
    print("\nTrain batch:")
    print(f"  Filenames: {train_batch['filename'][:2]}...")
    print(f"  Indices: {train_batch['example_index'][:2]}...")
    print(f"  Inputs shape: {train_batch['input_colors_expanded'].shape}")
    print(f"  Outputs shape: {train_batch['output_colors_expanded'].shape}")

    print("\nValidation batch:")
    print(f"  Filenames: {val_batch['filename'][:2]}...")
    print(f"  Indices: {val_batch['example_index'][:2]}...")
    print(f"  Inputs shape: {val_batch['input_colors_expanded'].shape}")
    print(f"  Outputs shape: {val_batch['output_colors_expanded'].shape}")
