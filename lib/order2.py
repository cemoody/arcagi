"""
PyTorch module for computing order-2 (pairwise) relational features.

This module provides a fast, vectorized implementation of the order-2 feature
computation used in the ARC-AGI preprocessing pipeline.

Example usage:
    # Create the transform
    transform = Order2Features()

    # Input: batch of color grids with shape [B, H, W]
    # Colors are integers 0-9, with -1 representing mask/empty cells
    color_grids = torch.randint(-1, 10, (32, 30, 30))

    # Compute features
    features = transform(color_grids)  # Output shape: [32, 30, 30, 45]

    # Features are binary (0 or 1):
    # - First 36 features: pairwise color comparisons in 3x3 neighborhood
    # - Next 8 features: mask detection for each neighbor
    # - Final feature: is_mask (whether the center cell is -1)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# No typing imports needed - using native type hints


class Order2Features(nn.Module):
    # Class-level annotations for registered buffers (helps static type checkers)
    pair_idx1: torch.Tensor
    pair_idx2: torch.Tensor
    neighbor_indices: torch.Tensor
    """
    Computes order-2 (pairwise) relational features for color grids.

    For each cell in the input grid, this module computes:
    1. Pairwise color matching in 3x3 neighborhood (36 features)
    2. Center-to-neighbor mask detection (8 features)
    3. is_mask: whether the center cell is a mask (-1) (1 feature)

    Total: 45 binary features per cell

    Input shape: [B, H, W] with integer color values (0-9) and -1 for mask
    Output shape: [B, H, W, 45] with binary features (0 or 1)
    """

    def __init__(self) -> None:
        super().__init__()

        # Type annotations for registered buffers (for type checkers)
        self.pair_idx1: torch.Tensor
        self.pair_idx2: torch.Tensor
        self.neighbor_indices: torch.Tensor

        # Pre-compute pair indices for the 36 pairwise comparisons
        # These are all unique pairs from a 3x3 neighborhood (9 positions)
        pairs: List[Tuple[int, int]] = []
        for i in range(9):
            for j in range(i + 1, 9):
                pairs.append((i, j))

        # Convert to tensors for efficient indexing
        pair_idx1: torch.Tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        pair_idx2: torch.Tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        self.register_buffer("pair_idx1", pair_idx1)
        self.register_buffer("pair_idx2", pair_idx2)

        # Neighbor indices (all except center which is at position 4)
        neighbor_indices: torch.Tensor = torch.tensor(
            [0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.long
        )
        self.register_buffer("neighbor_indices", neighbor_indices)

    def extract_neighborhoods(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 3x3 neighborhoods for all positions in the grid.
        """
        B, H, W = x.shape

        # Pad with -1 (mask value) to handle boundaries
        padded = F.pad(x, (1, 1, 1, 1), mode="constant", value=-1)

        # Use unfold to extract 3x3 patches
        # unfold dimension 1 (height): kernel_size=3, stride=1
        unfolded = padded.unfold(1, 3, 1)  # [B, H, W+2, 3]
        # unfold dimension 2 (width): kernel_size=3, stride=1
        unfolded = unfolded.unfold(2, 3, 1)  # [B, H, W, 3, 3]

        # Reshape to [B, H, W, 9]
        neighborhoods = unfolded.reshape(B, H, W, 9)

        return neighborhoods

    def compute_pairwise_features(self, neighborhoods: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise color matching features.
        """
        # Extract values for all pairs
        values1 = neighborhoods[..., self.pair_idx1]  # [B, H, W, 36]
        values2 = neighborhoods[..., self.pair_idx2]  # [B, H, W, 36]

        # Compare: 0 if same color, 1 if different
        pairwise_features = (values1 != values2).float()

        return pairwise_features

    def compute_mask_features(self, neighborhoods: torch.Tensor) -> torch.Tensor:
        """
        Compute center-to-neighbor mask detection features.
        """
        # Extract neighbor values (excluding center)
        neighbor_values = neighborhoods[..., self.neighbor_indices]  # [B, H, W, 8]

        # Check if neighbor is not mask: 0 if neighbor is -1, 1 otherwise
        mask_features = (neighbor_values != -1).float()

        return mask_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute order-2 features for the input color grid.
        """
        # Ensure input is integer type for comparisons
        x = x.long()

        # Extract 3x3 neighborhoods for all positions
        neighborhoods = self.extract_neighborhoods(x)

        # Compute pairwise features (36 features)
        pairwise_features = self.compute_pairwise_features(neighborhoods)

        # Compute mask features (8 features)
        mask_features = self.compute_mask_features(neighborhoods)

        # Compute is_mask for the center cell (1 if -1, else 0)
        center_values = neighborhoods[..., 4]
        is_mask_feature = (center_values == -1).float().unsqueeze(-1)

        # Concatenate all features: 36 + 8 + 1 = 45
        features = torch.cat(
            [pairwise_features, mask_features, is_mask_feature], dim=-1
        )

        return features


def create_order2_transform():
    """
    Create an instance of the Order2Features transform.

    This is a convenience function for creating the transform.
    """
    return Order2Features()


if __name__ == "__main__":
    # Simple test
    transform = create_order2_transform()

    # Create a simple test input
    test_input = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, -1]]])  # Shape: [1, 3, 3]

    # Compute features
    features = transform(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Features are binary: {torch.all((features == 0) | (features == 1))}")
