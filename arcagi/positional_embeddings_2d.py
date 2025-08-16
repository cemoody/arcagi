"""
2D Rotary Position Embeddings (RoPE) for Vision Transformers and Image Processing.

This module provides 2D RoPE implementation for encoding relative spatial positions
in image data, particularly for use with attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embeddings (RoPE) for images.

    Applies rotary embeddings separately to row and column dimensions.
    """

    # Type annotations for registered buffers
    inv_freq: torch.Tensor
    pos_y: torch.Tensor
    pos_x: torch.Tensor

    def __init__(self, embed_dim: int, height: int, width: int, base: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4"

        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.base = base

        # Compute frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, embed_dim // 4, 2).float() / (embed_dim // 4))
        )
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute position indices
        y = torch.arange(height).unsqueeze(1)
        x = torch.arange(width).unsqueeze(0)
        self.register_buffer("pos_y", y)
        self.register_buffer("pos_x", x)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE to query and key tensors.

        Args:
            q: Query tensor of shape [B, H, W, num_heads, head_dim] or [B, H*W, num_heads, head_dim]
            k: Key tensor of same shape as q

        Returns:
            Tuple of (q_rotated, k_rotated)
        """
        # Handle both 5D and 4D inputs
        reshape_output = False
        if q.dim() == 5:  # [B, H, W, num_heads, head_dim]
            batch_size, height, width, num_heads, head_dim = q.shape
            q = q.view(batch_size, height * width, num_heads, head_dim)
            k = k.view(batch_size, height * width, num_heads, head_dim)
            reshape_output = True
        else:  # [B, H*W, num_heads, head_dim]
            batch_size, seq_len, num_heads, head_dim = q.shape
            height = self.height
            width = self.width
            assert (
                seq_len == height * width
            ), f"Sequence length {seq_len} doesn't match H*W ({height}*{width}={height*width})"
            reshape_output = False

        # Split dimensions for row and column
        q_row, q_col = q.chunk(2, dim=-1)
        k_row, k_col = k.chunk(2, dim=-1)

        # Apply rotary embeddings
        q_row = self._apply_rope_1d(q_row, self.pos_y, self.height, self.width)
        q_col = self._apply_rope_1d(q_col, self.pos_x.T, self.height, self.width)
        k_row = self._apply_rope_1d(k_row, self.pos_y, self.height, self.width)
        k_col = self._apply_rope_1d(k_col, self.pos_x.T, self.height, self.width)

        # Concatenate back
        q_rotated = torch.cat([q_row, q_col], dim=-1)
        k_rotated = torch.cat([k_row, k_col], dim=-1)

        if reshape_output:
            q_rotated = q_rotated.view(batch_size, height, width, num_heads, head_dim)
            k_rotated = k_rotated.view(batch_size, height, width, num_heads, head_dim)

        return q_rotated, k_rotated

    def _apply_rope_1d(
        self, x: torch.Tensor, pos: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Apply 1D rotary embeddings."""
        # Create position encodings
        pos_flat = pos.flatten().unsqueeze(1)
        sincos = torch.einsum("i,j->ij", pos_flat, self.inv_freq)
        sin, cos = sincos.sin(), sincos.cos()

        # Apply rotation
        x_r, x_i = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_r * cos.unsqueeze(0).unsqueeze(
            2
        ) - x_i * sin.unsqueeze(0).unsqueeze(2)
        x_rotated[..., 1::2] = x_r * sin.unsqueeze(0).unsqueeze(
            2
        ) + x_i * cos.unsqueeze(0).unsqueeze(2)

        return x_rotated
