"""
Example of integrating D4 augmentation with existing data loaders.

This shows how to modify your existing data loading pipeline to include
on-the-fly D4 augmentation.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple


class D4Transform:
    """
    Callable transform class for D4 symmetry augmentations.
    Can be used with torchvision transforms or standalone.
    """
    
    def __init__(self, p: float = 1.0):
        """
        Args:
            p: Probability of applying augmentation (default 1.0)
        """
        self.p = p
        self.num_transforms = 8
    
    def __call__(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply the same random D4 transformation to all input tensors.
        
        Args:
            *tensors: Variable number of tensors to transform
            
        Returns:
            Tuple of transformed tensors
        """
        if torch.rand(1).item() > self.p:
            return tensors
        
        # Choose random transformation
        transform_idx = torch.randint(0, self.num_transforms, (1,)).item()
        
        # Apply same transformation to all tensors
        transformed = []
        for tensor in tensors:
            transformed.append(self.apply_transform(tensor, transform_idx))
        
        return tuple(transformed)
    
    def apply_transform(self, tensor: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply specific D4 transformation."""
        # Handle different tensor shapes
        if tensor.ndim == 4:  # Batch dimension
            spatial_dims = (-3, -2)  # H, W dimensions
        elif tensor.ndim == 3:  # Single sample with channels
            spatial_dims = (0, 1)  # H, W dimensions
        elif tensor.ndim == 2:  # Single sample without channels
            spatial_dims = (0, 1)
        else:
            return tensor  # Can't transform 1D or 5D+ tensors
        
        if idx == 0:  # Identity
            return tensor
        elif idx == 1:  # 90° rotation
            return torch.rot90(tensor, k=1, dims=spatial_dims)
        elif idx == 2:  # 180° rotation
            return torch.rot90(tensor, k=2, dims=spatial_dims)
        elif idx == 3:  # 270° rotation
            return torch.rot90(tensor, k=3, dims=spatial_dims)
        elif idx == 4:  # Horizontal flip
            return torch.flip(tensor, dims=[spatial_dims[1]])
        elif idx == 5:  # Horizontal flip + 90° rotation
            return torch.rot90(torch.flip(tensor, dims=[spatial_dims[1]]), k=1, dims=spatial_dims)
        elif idx == 6:  # Horizontal flip + 180° rotation
            return torch.rot90(torch.flip(tensor, dims=[spatial_dims[1]]), k=2, dims=spatial_dims)
        elif idx == 7:  # Horizontal flip + 270° rotation
            return torch.rot90(torch.flip(tensor, dims=[spatial_dims[1]]), k=3, dims=spatial_dims)


def create_dataloader_with_d4_augmentation(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    inputs_features: Optional[torch.Tensor] = None,
    outputs_features: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = True,
    augment_prob: float = 1.0,
) -> DataLoader:
    """
    Create a dataloader with on-the-fly D4 augmentation.
    
    This function modifies your existing create_dataloader pattern to include
    D4 augmentation.
    """
    
    # Create the base dataset
    tensors = [inputs, outputs]
    if inputs_features is not None:
        tensors.append(inputs_features)
    if outputs_features is not None:
        tensors.append(outputs_features)
    if indices is not None:
        tensors.append(indices)
    
    dataset = TensorDataset(*tensors)
    
    # Create transform
    d4_transform = D4Transform(p=augment_prob) if augment else None
    
    # Custom collate function that applies D4 augmentation
    def collate_with_d4(batch):
        # Default collate
        batch_tensors = torch.utils.data.default_collate(batch)
        
        if d4_transform is not None and augment:
            # Apply D4 augmentation to each sample in the batch
            transformed_batch = []
            batch_size = batch_tensors[0].size(0)
            
            for i in range(batch_size):
                # Get single sample from batch
                sample = [t[i] for t in batch_tensors]
                
                # Apply same transformation to input/output (and features if present)
                if len(sample) >= 2:
                    # Transform spatial data (not indices)
                    spatial_data = sample[:-1] if indices is not None else sample
                    transformed = d4_transform(*spatial_data)
                    
                    # Reconstruct sample
                    if indices is not None:
                        transformed = transformed + (sample[-1],)
                    
                    transformed_batch.append(transformed)
                else:
                    transformed_batch.append(sample)
            
            # Reconstruct batch
            return torch.utils.data.default_collate(transformed_batch)
        
        return batch_tensors
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_with_d4 if augment else None,
    )


# Alternative: Using Kornia for GPU-based augmentation
def create_kornia_d4_augmented_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = 'cuda',
) -> DataLoader:
    """
    Example using Kornia for GPU-accelerated D4 augmentation.
    
    Note: This requires kornia to be installed: pip install kornia
    """
    try:
        import kornia
        import kornia.augmentation as K
    except ImportError:
        print("Kornia not installed. Install with: pip install kornia")
        return None
    
    # Create base dataset
    dataset = TensorDataset(inputs, outputs)
    
    # Create Kornia augmentation pipeline
    # Note: Kornia expects (B, C, H, W) format
    augmentation = K.AugmentationSequential(
        K.RandomRotation(degrees=[0, 90, 180, 270], p=0.75),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        data_keys=["input", "output"],  # Apply same transform to both
        same_on_batch=False,  # Different transforms per sample
        keepdim=True,
    )
    
    def collate_with_kornia(batch):
        # Collate batch
        inputs_batch = torch.stack([item[0] for item in batch])
        outputs_batch = torch.stack([item[1] for item in batch])
        
        # Move to device and reshape if needed
        # Assuming inputs are (B, H, W, C), convert to (B, C, H, W)
        if inputs_batch.shape[-1] < inputs_batch.shape[-2]:
            inputs_batch = inputs_batch.permute(0, 3, 1, 2)
            outputs_batch = outputs_batch.permute(0, 3, 1, 2)
        
        # Apply augmentation on GPU
        if device == 'cuda' and torch.cuda.is_available():
            inputs_batch = inputs_batch.to(device)
            outputs_batch = outputs_batch.to(device)
            
            augmented = augmentation(inputs_batch, outputs_batch)
            inputs_aug = augmented[0]
            outputs_aug = augmented[1]
            
            # Convert back to original format if needed
            inputs_aug = inputs_aug.permute(0, 2, 3, 1).cpu()
            outputs_aug = outputs_aug.permute(0, 2, 3, 1).cpu()
        else:
            inputs_aug = inputs_batch
            outputs_aug = outputs_batch
        
        return inputs_aug, outputs_aug
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_with_kornia,
        pin_memory=True,
    )


# Integration with your existing code
def modify_existing_create_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    inputs_features: Optional[torch.Tensor] = None,
    outputs_features: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    apply_d4_augmentation: bool = False,
) -> DataLoader:
    """
    Modified version of your create_dataloader function with D4 augmentation option.
    
    This shows how to add D4 augmentation to your existing data loading pipeline
    with minimal changes.
    """
    # Handle empty features
    if inputs_features is not None and inputs_features.shape[-1] == 0:
        inputs_features = None
    if outputs_features is not None and outputs_features.shape[-1] == 0:
        outputs_features = None
    
    # Fill in empty tensors as in original
    inputs_features = (
        inputs_features
        if inputs_features is not None
        else torch.empty(len(inputs), 30, 30, 0)
    )
    outputs_features = (
        outputs_features
        if outputs_features is not None
        else torch.empty(len(outputs), 30, 30, 0)
    )
    
    if apply_d4_augmentation:
        # Use D4 augmented version
        return create_dataloader_with_d4_augmentation(
            inputs=inputs,
            outputs=outputs,
            inputs_features=inputs_features,
            outputs_features=outputs_features,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            augment=True,
        )
    else:
        # Original implementation
        dataset = TensorDataset(inputs, outputs, inputs_features, outputs_features, indices)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Example usage
    print("D4 Augmentation Example")
    print("-" * 50)
    
    # Create sample data
    batch_size = 4
    inputs = torch.randn(16, 30, 30, 11)
    outputs = torch.randn(16, 30, 30, 11)
    
    # Create distinctive pattern to track transformations
    inputs[0, :5, :5, 0] = 1.0  # Top-left corner marker
    outputs[0, :5, :5, 0] = 1.0  # Same marker in output
    
    # Create dataloader with D4 augmentation
    train_loader = create_dataloader_with_d4_augmentation(
        inputs=inputs,
        outputs=outputs,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle to see first sample
        augment=True,
    )
    
    print("Testing D4 augmented dataloader:")
    for i, batch in enumerate(train_loader):
        inp_batch, out_batch = batch[0], batch[1]
        print(f"\nBatch {i}:")
        print(f"  Input shape: {inp_batch.shape}")
        print(f"  Output shape: {out_batch.shape}")
        
        # Check if augmentation is applied (corner marker should move)
        print(f"  First sample input corner sum: {inp_batch[0, :5, :5, 0].sum():.2f}")
        print(f"  First sample output corner sum: {out_batch[0, :5, :5, 0].sum():.2f}")
        
        if i >= 2:
            break
    
    print("\n" + "-" * 50)
    print("Comparison with non-augmented dataloader:")
    
    # Create non-augmented version
    val_loader = create_dataloader_with_d4_augmentation(
        inputs=inputs,
        outputs=outputs,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    
    for i, batch in enumerate(val_loader):
        inp_batch, out_batch = batch[0], batch[1]
        print(f"\nBatch {i} (no augmentation):")
        print(f"  First sample input corner sum: {inp_batch[0, :5, :5, 0].sum():.2f}")
        print(f"  First sample output corner sum: {out_batch[0, :5, :5, 0].sum():.2f}")
        
        if i >= 0:
            break