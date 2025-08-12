"""
PyTorch Dataset with D4 symmetry augmentation for both inputs and outputs.

This module provides a custom PyTorch Dataset that applies D4 symmetry transformations
(rotations and reflections) to both inputs and outputs using the same transformation.
"""

import torch
from torch.utils.data import Dataset, RandomSampler
import numpy as np
from typing import Tuple, Optional, List
import random


class D4AugmentedDataset(Dataset):
    """
    A PyTorch Dataset that applies D4 symmetry transformations to both inputs and outputs.
    
    The D4 dihedral group consists of 8 transformations:
    - Identity (no change)
    - 90° rotation
    - 180° rotation
    - 270° rotation
    - Horizontal flip
    - Horizontal flip + 90° rotation
    - Horizontal flip + 180° rotation
    - Horizontal flip + 270° rotation
    """
    
    def __init__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        inputs_features: Optional[torch.Tensor] = None,
        outputs_features: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        augment: bool = True,
        deterministic_augmentation: bool = False,
    ):
        """
        Initialize the D4 augmented dataset.
        
        Args:
            inputs: Input tensors of shape (N, H, W, C) or (N, H, W)
            outputs: Output tensors of shape (N, H, W, C) or (N, H, W)
            inputs_features: Optional feature tensors for inputs
            outputs_features: Optional feature tensors for outputs
            indices: Optional indices tensor
            augment: Whether to apply augmentation (set False for validation)
            deterministic_augmentation: If True, cycles through all 8 transformations
                                      If False, randomly selects one transformation
        """
        self.inputs = inputs
        self.outputs = outputs
        self.inputs_features = inputs_features
        self.outputs_features = outputs_features
        self.indices = indices
        self.augment = augment
        self.deterministic_augmentation = deterministic_augmentation
        self.num_transforms = 8
        
        # Store original length
        self.original_length = len(inputs)
        
    def __len__(self):
        if self.augment and self.deterministic_augmentation:
            # Return 8x the original length to cycle through all transformations
            return self.original_length * self.num_transforms
        return self.original_length
    
    def apply_d4_transform(
        self, 
        tensor: torch.Tensor, 
        transform_idx: int
    ) -> torch.Tensor:
        """
        Apply a specific D4 transformation to a tensor.
        
        Args:
            tensor: Input tensor of shape (..., H, W, ...) where H, W are spatial dims
            transform_idx: Index of transformation (0-7)
            
        Returns:
            Transformed tensor
        """
        # Determine which dimensions are spatial (assuming they're the 2nd and 3rd dims)
        # For tensors of shape (H, W, C) or (H, W)
        if tensor.ndim == 3:  # (H, W, C)
            spatial_dims = (0, 1)
        elif tensor.ndim == 2:  # (H, W)
            spatial_dims = (0, 1)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        
        # Apply transformation based on index
        if transform_idx == 0:
            # Identity
            return tensor
        elif transform_idx == 1:
            # Rotate 90° clockwise
            # Note: k=-1 for clockwise, k=1 for counter-clockwise
            return torch.rot90(tensor, k=-1, dims=spatial_dims)
        elif transform_idx == 2:
            # Rotate 180°
            return torch.rot90(tensor, k=2, dims=spatial_dims)
        elif transform_idx == 3:
            # Rotate 270° clockwise (90° counter-clockwise)
            return torch.rot90(tensor, k=-3, dims=spatial_dims)
        elif transform_idx == 4:
            # Horizontal flip
            return torch.flip(tensor, dims=[spatial_dims[1]])
        elif transform_idx == 5:
            # Horizontal flip + rotate 90°
            flipped = torch.flip(tensor, dims=[spatial_dims[1]])
            return torch.rot90(flipped, k=-1, dims=spatial_dims)
        elif transform_idx == 6:
            # Horizontal flip + rotate 180°
            flipped = torch.flip(tensor, dims=[spatial_dims[1]])
            return torch.rot90(flipped, k=2, dims=spatial_dims)
        elif transform_idx == 7:
            # Horizontal flip + rotate 270°
            flipped = torch.flip(tensor, dims=[spatial_dims[1]])
            return torch.rot90(flipped, k=-3, dims=spatial_dims)
        else:
            raise ValueError(f"Invalid transform index: {transform_idx}")
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get an item with D4 augmentation applied.
        
        Returns:
            Tuple of (input, output, [input_features], [output_features], [index])
        """
        if self.augment:
            if self.deterministic_augmentation:
                # Deterministic: cycle through all transformations
                original_idx = idx % self.original_length
                transform_idx = idx // self.original_length
            else:
                # Random: select a random transformation
                original_idx = idx
                transform_idx = random.randint(0, self.num_transforms - 1)
        else:
            # No augmentation
            original_idx = idx
            transform_idx = 0
        
        # Get the original data
        input_tensor = self.inputs[original_idx]
        output_tensor = self.outputs[original_idx]
        
        # Apply the same transformation to both input and output
        input_transformed = self.apply_d4_transform(input_tensor, transform_idx)
        output_transformed = self.apply_d4_transform(output_tensor, transform_idx)
        
        # Build return tuple
        return_items = [input_transformed, output_transformed]
        
        # Apply transformation to features if they exist
        if self.inputs_features is not None:
            input_features_transformed = self.apply_d4_transform(
                self.inputs_features[original_idx], transform_idx
            )
            return_items.append(input_features_transformed)
        
        if self.outputs_features is not None:
            output_features_transformed = self.apply_d4_transform(
                self.outputs_features[original_idx], transform_idx
            )
            return_items.append(output_features_transformed)
        
        # Add index if it exists
        if self.indices is not None:
            return_items.append(self.indices[original_idx])
        
        return tuple(return_items)


class D4CollateFunction:
    """
    Custom collate function that can handle D4 augmented batches.
    
    This is useful when you want to apply different transformations to different
    items in the batch but ensure input/output pairs get the same transformation.
    """
    
    def __init__(self, apply_random_d4: bool = True):
        self.apply_random_d4 = apply_random_d4
        self.num_transforms = 8
    
    def apply_d4_transform(self, tensor: torch.Tensor, transform_idx: int) -> torch.Tensor:
        """Apply D4 transformation (same as in dataset)."""
        # Handle different tensor dimensions
        if tensor.ndim == 4:  # Batched tensor (B, H, W, C)
            spatial_dims = (1, 2)  # H, W dimensions
        elif tensor.ndim == 3:  # Single tensor (H, W, C)
            spatial_dims = (0, 1)  # H, W dimensions
        elif tensor.ndim == 2:  # Single tensor (H, W)
            spatial_dims = (0, 1)  # H, W dimensions
        else:
            return tensor
        
        if transform_idx == 0:
            return tensor
        elif transform_idx == 1:
            # Rotate 90° clockwise
            return torch.rot90(tensor, k=-1, dims=spatial_dims)
        elif transform_idx == 2:
            # Rotate 180°
            return torch.rot90(tensor, k=2, dims=spatial_dims)
        elif transform_idx == 3:
            # Rotate 270° clockwise
            return torch.rot90(tensor, k=-3, dims=spatial_dims)
        elif transform_idx == 4:
            # Horizontal flip
            return torch.flip(tensor, dims=[spatial_dims[1]])
        elif transform_idx == 5:
            # Horizontal flip + rotate 90°
            flipped = torch.flip(tensor, dims=[spatial_dims[1]])
            return torch.rot90(flipped, k=-1, dims=spatial_dims)
        elif transform_idx == 6:
            # Horizontal flip + rotate 180°
            flipped = torch.flip(tensor, dims=[spatial_dims[1]])
            return torch.rot90(flipped, k=2, dims=spatial_dims)
        elif transform_idx == 7:
            # Horizontal flip + rotate 270°
            flipped = torch.flip(tensor, dims=[spatial_dims[1]])
            return torch.rot90(flipped, k=-3, dims=spatial_dims)
    
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        """
        Collate function that optionally applies D4 transformations.
        
        Each item in the batch gets the same transformation applied to its
        input/output pair, but different items can get different transformations.
        """
        # Transpose batch of tuples to tuple of lists
        batch_transposed = list(zip(*batch))
        
        # Stack each component
        tensors = [torch.stack(component) for component in batch_transposed]
        
        if self.apply_random_d4:
            # Apply random D4 transform to each item in batch
            batch_size = len(tensors[0])
            
            for i in range(batch_size):
                # Choose random transformation for this item
                transform_idx = random.randint(0, self.num_transforms - 1)
                
                # Apply same transform to input and output (first two tensors)
                tensors[0][i] = self.apply_d4_transform(tensors[0][i], transform_idx)
                tensors[1][i] = self.apply_d4_transform(tensors[1][i], transform_idx)
                
                # Apply to features if they exist (assuming they're tensors 2 and 3)
                if len(tensors) > 2 and tensors[2].ndim >= 3:
                    tensors[2][i] = self.apply_d4_transform(tensors[2][i], transform_idx)
                if len(tensors) > 3 and tensors[3].ndim >= 3:
                    tensors[3][i] = self.apply_d4_transform(tensors[3][i], transform_idx)
        
        return tuple(tensors)


def create_d4_augmented_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    inputs_features: Optional[torch.Tensor] = None,
    outputs_features: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = True,
    deterministic_augmentation: bool = False,
    use_custom_collate: bool = False,
    repeat_factor: int = 1,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with D4 augmentation.
    
    Args:
        inputs: Input tensors
        outputs: Output tensors
        inputs_features: Optional input features
        outputs_features: Optional output features
        indices: Optional indices
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        augment: Whether to apply augmentation
        deterministic_augmentation: If True, cycles through all transformations
        use_custom_collate: If True, uses custom collate function for batch-level augmentation
        repeat_factor: If > 1, repeats sampling within an epoch to see the same example
            multiple times (useful with stochastic transforms). Implemented via a sampler
            with replacement. Effective number of samples per epoch becomes
            len(dataset) * repeat_factor (or len(dataset) * 8 * repeat_factor when
            deterministic_augmentation=True).
        
    Returns:
        DataLoader with D4 augmentation
    """
    dataset = D4AugmentedDataset(
        inputs=inputs,
        outputs=outputs,
        inputs_features=inputs_features,
        outputs_features=outputs_features,
        indices=indices,
        augment=augment,
        deterministic_augmentation=deterministic_augmentation,
    )
    
    collate_fn = D4CollateFunction(apply_random_d4=False) if use_custom_collate else None

    sampler = None
    if repeat_factor is not None and repeat_factor > 1:
        # Sample with replacement to achieve repeated views per epoch
        total_samples = len(dataset) * repeat_factor
        sampler = RandomSampler(dataset, replacement=True, num_samples=total_samples)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )


# Example usage:
if __name__ == "__main__":
    # Create dummy data
    batch_size = 4
    n_samples = 16
    height, width = 30, 30
    n_colors = 11
    n_features = 44
    
    # Create tensors
    inputs = torch.rand(n_samples, height, width, n_colors)
    outputs = torch.rand(n_samples, height, width, n_colors)
    inputs_features = torch.rand(n_samples, height, width, n_features)
    outputs_features = torch.rand(n_samples, height, width, n_features)
    indices = torch.arange(n_samples)
    
    # Create augmented dataloader
    train_loader = create_d4_augmented_dataloader(
        inputs=inputs,
        outputs=outputs,
        inputs_features=inputs_features,
        outputs_features=outputs_features,
        indices=indices,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        deterministic_augmentation=False,  # Random augmentation
    )
    
    # Create validation dataloader (no augmentation)
    val_loader = create_d4_augmented_dataloader(
        inputs=inputs,
        outputs=outputs,
        inputs_features=inputs_features,
        outputs_features=outputs_features,
        indices=indices,
        batch_size=batch_size,
        shuffle=False,
        augment=False,  # No augmentation for validation
    )
    
    # Test the dataloader
    print("Testing augmented dataloader:")
    for i, batch in enumerate(train_loader):
        if len(batch) == 5:
            inp, out, inp_feat, out_feat, idx = batch
            print(f"Batch {i}: Input shape: {inp.shape}, Output shape: {out.shape}")
            print(f"  Feature shapes: {inp_feat.shape}, {out_feat.shape}")
            print(f"  Indices: {idx}")
        else:
            print(f"Batch {i}: {len(batch)} tensors")
        
        if i >= 2:  # Just show a few batches
            break
    
    print("\nDemonstrating that same transform is applied to input/output pairs:")
    # Create a simple test case
    test_input = torch.zeros(1, 4, 4, 1)
    test_input[0, 0, :, 0] = torch.tensor([1, 2, 3, 4])  # Distinctive pattern
    test_output = torch.zeros(1, 4, 4, 1)
    test_output[0, :, 0, 0] = torch.tensor([5, 6, 7, 8])  # Different pattern
    
    dataset = D4AugmentedDataset(
        inputs=test_input,
        outputs=test_output,
        augment=True,
        deterministic_augmentation=True
    )
    
    print("\nOriginal input (first row):", test_input[0, 0, :, 0])
    print("Original output (first column):", test_output[0, :, 0, 0])
    
    for i in range(8):
        inp, out = dataset[i]
        print(f"\nTransform {i}:")
        print("  Input shape:", inp.shape)
        print("  Output shape:", out.shape)
        # You can inspect the transformed tensors to verify they match