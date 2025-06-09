"""
TemporalGridAttention (TGA) Module Suite
========================================

A collection of building-blocks for learning on 2-D spatial grids that have
per-cell temporal histories.

Main public classes
-------------------
* **TemporalGridAttention** – one FlashAttention-2 layer that attends from the
  *latest* 30×30 grid frame (queries) to:
    * a *learnable* t = 0 frame (shared across all cells),
    * each cell's own history (t = 1…T),
    * the latest grid itself (for self-attention).
  Spatial co-ordinates use **2-D rotary embeddings**; time uses **1-D RoPE**.
  Group-query attention (more Q heads than KV heads) and FP16 FlashAttention
  make the layer extremely lightweight.

* **TGALayer** – wraps a TemporalGridAttention with a wider **SwiGLU** feed-
  forward and a residual connection.

* **TGABlock** – stacks *depth* TGALayers (default = 6) to increase representational
  power without touching the embedding dimension.

* **RepeatedTGA** – iteratively applies a TGABlock up to *max_timesteps*.
  The previous grid frame is appended to history after every step.
  Optional **local 3×3 depth-wise mixing** lets each cell pre-mix with its eight
  neighbours before the attention pass.

All tensors follow the « (H, W, T, D) » convention:
* `H, W` : spatial grid = 30 × 30 (hard-coded but easy to change)
* `T`    : history length (0 allowed)
* `D`    : embedding dimension (default 48, must be multiple of 8 / heads)

FlashAttention-2 is required (`pip install flash-attn`).  Everything runs in
FP16/BF16 for speed; outputs are promoted back to FP32 before returning.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# Utility layers
######################################################################


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: (x · W₁) ⊗ swish(x · W₂) → proj → dropout."""

    def __init__(self, d_model: int, factor: int = 8, dropout: float = 0.01) -> None:
        super().__init__()  # type: ignore
        hidden = factor * d_model
        self.w12 = nn.Linear(d_model, hidden * 2, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (… , D)
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.proj(self.act(a) * b))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes using the root mean square of the input features,
    which is more efficient than LayerNorm as it doesn't compute mean and variance.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()  # type: ignore
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS normalization
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


######################################################################
# Rotary helpers
######################################################################


def rope_pairwise(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    xr, xi = x[..., 0], x[..., 1]
    return torch.stack((xr * cos - xi * sin, xr * sin + xi * cos), dim=-1)


def apply_rope_1d(
    x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    """Apply 1D Rotary Position Embedding (RoPE) to tensor along temporal axis.

    RoPE rotates pairs of adjacent features in the embedding space based on their position,
    enabling the model to understand relative positions in sequences. This is particularly
    effective for temporal sequences where position encoding needs to capture time relationships.

    The rotation is applied element-wise to pairs of features using:
    [x_even, x_odd] → [x_even * cos(θ) - x_odd * sin(θ), x_even * sin(θ) + x_odd * cos(θ)]
    where θ = pos * inv_freq for each frequency component.

    Args:
        x (torch.Tensor): Input tensor of shape (L, H, d) where:
            - L: sequence length (number of positions)
            - H: number of attention heads
            - d: embedding dimension (must be even for pairing)
        pos (torch.Tensor): Position indices of shape (L,). Each element represents
            the position/time step for the corresponding sequence element.
        inv_freq (torch.Tensor): Inverse frequency tensor of shape (d//2,). Contains
            1 / (base^(2i/d)) for i in range(d//2), typically with base=10000.

    Returns:
        torch.Tensor: Rotated tensor of same shape (L, H, d) with position information
            encoded through rotational transformations. Preserves the magnitude of input
            vectors while encoding positional relationships.

    Example:
        >>> L, H, d = 10, 6, 8  # 10 timesteps, 6 heads, 8 dimensions
        >>> x = torch.randn(L, H, d)
        >>> pos = torch.arange(L).float()  # [0, 1, 2, ..., 9]
        >>> inv_freq = 1.0 / (10000 ** (torch.arange(d // 2) / (d // 2)))
        >>> rotated = apply_rope_1d(x, pos, inv_freq)
        >>> print(rotated.shape)  # torch.Size([10, 6, 8])

    Note:
        - The embedding dimension d must be even to allow pairing
        - Different frequency components create different rotation rates
        - Nearby positions will have more similar embeddings than distant ones
        - This function preserves the L2 norm of the input vectors
    """
    L, H, d = x.shape
    theta = torch.einsum("l,f->lf", pos, inv_freq)  # (L, d//2)
    cos, sin = theta.cos(), theta.sin()  # (L, d//2)
    x2 = x.view(L, H, -1, 2)  # (L, H, d//2, 2)
    # Reshape cos/sin to broadcast correctly: (L, d//2) -> (L, 1, d//2)
    cos = cos[:, None, :]  # (L, 1, d//2)
    sin = sin[:, None, :]  # (L, 1, d//2)
    return rope_pairwise(x2, cos, sin).view(L, H, d)


def apply_rope_2d(
    x: torch.Tensor, coords_xy: torch.Tensor, inv_x: torch.Tensor, inv_y: torch.Tensor
) -> torch.Tensor:
    """Apply 2D Rotary Position Embedding (RoPE) to tensor along spatial axes.

    Extends RoPE to 2D spatial coordinates by applying separate rotations for X and Y
    dimensions. The embedding is split into two halves: first half gets X-axis rotations,
    second half gets Y-axis rotations. This enables the model to understand spatial
    relationships in 2D grids, images, or other spatially-structured data.

    The rotation is applied as:
    - First d/2 dimensions: rotated based on X coordinates
    - Last d/2 dimensions: rotated based on Y coordinates
    Each rotation follows the same pairwise pattern as 1D RoPE.

    Args:
        x (torch.Tensor): Input tensor of shape (L, H, d) where:
            - L: number of spatial positions
            - H: number of attention heads
            - d: embedding dimension (must be divisible by 4 for X/Y splitting)
        coords_xy (torch.Tensor): 2D coordinates of shape (L, 2) where coords_xy[i]
            contains [x_coord, y_coord] for position i. Coordinates can be integers
            (grid indices) or continuous values.
        inv_x (torch.Tensor): Inverse frequency tensor for X-axis of shape (d//4,).
            Controls rotation rates along the X dimension.
        inv_y (torch.Tensor): Inverse frequency tensor for Y-axis of shape (d//4,).
            Controls rotation rates along the Y dimension.

    Returns:
        torch.Tensor: Rotated tensor of same shape (L, H, d) with 2D spatial position
            information encoded. First d/2 dimensions contain X-rotated features,
            last d/2 dimensions contain Y-rotated features.

    Example:
        >>> L, H, d = 9, 4, 8  # 3x3 grid, 4 heads, 8 dimensions
        >>> x = torch.randn(L, H, d)
        >>> # Create 3x3 grid coordinates: (0,0), (0,1), (0,2), (1,0), ..., (2,2)
        >>> coords = torch.stack(torch.meshgrid(
        ...     torch.arange(3), torch.arange(3), indexing='ij'
        ... ), dim=-1).reshape(-1, 2).float()
        >>> inv_freq = 1.0 / (10000 ** (torch.arange(d // 4) / (d // 4)))
        >>> inv_x = inv_y = inv_freq
        >>> rotated = apply_rope_2d(x, coords, inv_x, inv_y)
        >>> print(rotated.shape)  # torch.Size([9, 4, 8])

    Note:
        - The embedding dimension d must be divisible by 4 (d/2 for each axis)
        - Nearby coordinates will produce more similar embeddings than distant ones
        - Each axis (X, Y) gets independent rotational encoding
        - This function preserves the L2 norm of the input vectors
        - Commonly used in vision transformers and 2D attention mechanisms
    """
    L, H, d = x.shape
    d_axis = d // 2  # half for X, half for Y

    # X axis
    xb = x[..., :d_axis].view(L, H, -1, 2)  # (L, H, d_axis//2, 2)
    theta_x = torch.einsum("l,f->lf", coords_xy[:, 0], inv_x)  # (L, d_axis//2)
    cos_x, sin_x = theta_x.cos(), theta_x.sin()  # (L, d_axis//2)
    cos_x = cos_x[:, None, :]  # (L, 1, d_axis//2)
    sin_x = sin_x[:, None, :]  # (L, 1, d_axis//2)
    x_part = rope_pairwise(xb, cos_x, sin_x).view(L, H, d_axis)

    # Y axis
    yb = x[..., d_axis:].view(L, H, -1, 2)  # (L, H, d_axis//2, 2)
    theta_y = torch.einsum("l,f->lf", coords_xy[:, 1], inv_y)  # (L, d_axis//2)
    cos_y, sin_y = theta_y.cos(), theta_y.sin()  # (L, d_axis//2)
    cos_y = cos_y[:, None, :]  # (L, 1, d_axis//2)
    sin_y = sin_y[:, None, :]  # (L, 1, d_axis//2)
    y_part = rope_pairwise(yb, cos_y, sin_y).view(L, H, d_axis)

    return torch.cat((x_part, y_part), dim=-1)


######################################################################
# TemporalGridAttention (single layer)
######################################################################


class TemporalGridAttention(nn.Module):
    """Single FlashAttention-2 layer over a temporal 30×30 grid.

    Args:
        embed_dim: Hidden size **D** (must be divisible by 8 · num_q_heads).
        num_q_heads: Number of *query* heads (Hq).
        num_kv_heads: Number of KV heads for grouped-query (Hkv ≤ Hq).
        attn_dropout: Drop-out inside FlashAttention softmax.
    """

    def __init__(
        self,
        embed_dim: int = 48,
        num_q_heads: int = 6,
        num_kv_heads: int = 2,
        attn_dropout: float = 0.01,
        grid_size: int = 30,
    ) -> None:
        super().__init__()  # type: ignore
        assert embed_dim % num_q_heads == 0, "embed_dim % num_q_heads != 0"
        head_dim = embed_dim // num_q_heads
        assert head_dim % 8 == 0, "FlashAttention needs head_dim multiple of 8"
        assert embed_dim % num_kv_heads == 0, "embed_dim % num_kv_heads != 0"

        # For grouped-query attention, we need num_q_heads to be divisible by num_kv_heads
        assert (
            num_q_heads % num_kv_heads == 0
        ), f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for grouped-query attention"

        self.D = embed_dim
        self.Hq = num_q_heads
        self.Hkv = num_kv_heads
        self.dh = head_dim
        self.S = grid_size * grid_size  # 30×30 = 900
        self.dropout_p = attn_dropout

        # learnable t = 0 embedding (shared across cells)
        self.init_embed = nn.Parameter(torch.randn(embed_dim))

        # 2-D coords buffer
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
            ),
            dim=-1,
        )  # (H, W, 2)
        self.register_buffer("coords", coords.reshape(-1, 2).float(), persistent=False)

        # inverse freq tables - fix for 2D RoPE
        # For 2D RoPE: d_axis = head_dim // 2, and we need d_axis//2 = head_dim//4 frequencies
        quarter = head_dim // 4
        assert quarter > 0, f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE"
        inv_2d = 1.0 / (10000 ** (torch.arange(quarter) / quarter))
        # For 1D RoPE: we need head_dim//2 frequencies
        half = head_dim // 2
        inv_1d = 1.0 / (10000 ** (torch.arange(half) / half))

        self.register_buffer("inv_x", inv_2d, persistent=False)
        self.register_buffer("inv_y", inv_2d.clone(), persistent=False)
        self.register_buffer("inv_t", inv_1d, persistent=False)

        # Type annotations for buffers
        self.coords: torch.Tensor
        self.inv_x: torch.Tensor
        self.inv_y: torch.Tensor
        self.inv_t: torch.Tensor

        # projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.Hkv * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.Hkv * head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        static_grid: torch.Tensor,  # (H, W, D)
        history_seq: torch.Tensor,  # (H, W, T, D)
    ) -> torch.Tensor:  # (H, W, D)
        """Apply pedestal-shaped temporal grid attention to temporally evolving 2D grids of embeddings.

        This method implements a specialized attention mechanism over a 4D tensor representing
        the temporal evolution of a 2D spatial grid. Each cell has an embedding, creating
        a (H, W, T, D) structure where the time dimension captures the grid's history.

        **Pedestal-Shaped Attention Pattern:**
        For each cell (h, w) in the latest timestep, that cell attends to:
        1. **Spatial**: All other cells at the current timestep → full 2D spatial attention
        2. **Temporal**: Its own history (h, w, t) for t = 0, 1, ..., T-1 → 1D column through time
        3. **Exclusion**: It does NOT attend to other cells' past states

        This creates a "pedestal" shape where the top is the full current 2D grid and the
        column extends backwards in time only for each cell's own spatial location.

        **Implementation Details:**
        - Tokens are assembled as: [learnable_init, cell_histories, current_grid]
        - Uses 2D RoPE for spatial coordinates and 1D RoPE for temporal positions
        - FlashAttention-2 computes the attention efficiently in FP16
        - Only the updated current grid frame is returned

        Args:
            static_grid (torch.Tensor): Current 2D grid frame of shape (H, W, D) where:
                - H, W: Spatial dimensions of the grid (default 30×30)
                - D: Embedding dimension per cell (default 48)
                This represents the queries in the attention mechanism.

            history_seq (torch.Tensor): Historical evolution of shape (H, W, T, D) where:
                - T: Number of historical timesteps (can be 0 for no history)
                - Each (h, w, t) contains the embedding for cell (h,w) at time t
                These provide the temporal context for each spatial location.

        Returns:
            torch.Tensor: Updated current grid of shape (H, W, D) where each cell's
                embedding now incorporates:
                - Information from ALL cells in the current timestep (spatial context)
                - Information from its OWN temporal history (temporal continuity)
                - The learnable initial embedding (global context)

        Note:
            - Attention complexity scales as O(HW × (HW + T)) per cell
            - Uses grouped-query attention (fewer KV heads than Q heads) for efficiency
            - Processes in FP16 internally, returns FP32 for numerical stability
            - The pedestal shape ensures temporal locality while allowing spatial mixing

        Example:
            >>> # 3×3 grid evolving over 4 timesteps with 8-dim embeddings
            >>> tga = TemporalGridAttention(embed_dim=8, grid_size=3)
            >>> current = torch.randn(3, 3, 8)      # Current 2D grid state
            >>> history = torch.randn(3, 3, 4, 8)   # 4 historical timesteps
            >>> updated = tga(current, history)      # Apply pedestal attention
            >>> print(updated.shape)                # torch.Size([3, 3, 8])
        """
        device = static_grid.device
        S = self.S
        T = history_seq.shape[2] if history_seq.numel() else 0

        # Check if module is in half precision
        is_half = next(self.parameters()).dtype == torch.float16

        # build token list ------------------------------------------------
        init_tok = self.init_embed.unsqueeze(0).expand(S, -1)  # (S, D)
        hist_tok = (
            history_seq.reshape(S * T, self.D)
            if T > 0
            else history_seq.new_empty((0, self.D))
        )
        grid_tok = static_grid.reshape(S, self.D)
        tokens = torch.cat((init_tok, hist_tok, grid_tok), dim=0)

        # Convert to half precision only if module is in half precision
        if is_half:
            tokens = tokens.half()

        L_total = tokens.shape[0]

        # projections -----------------------------------------------------
        q = self.q_proj(tokens).view(L_total, self.Hq, self.dh)
        k = self.k_proj(tokens).view(L_total, self.Hkv, self.dh)
        v = self.v_proj(tokens).view(L_total, self.Hkv, self.dh)

        # 2-D RoPE for everyone ------------------------------------------
        coords_all = torch.cat(
            (
                self.coords,  # t = 0   (S)
                self.coords.repeat(T, 1),  # t = 1…T (S·T)
                self.coords,  # latest  (S)
            ),
            dim=0,
        ).to(device)
        q = apply_rope_2d(q, coords_all, self.inv_x, self.inv_y)
        k = apply_rope_2d(k, coords_all, self.inv_x, self.inv_y)

        # 1-D RoPE for history (shifted by +1) ---------------------------
        if T > 0:
            start, end = S, S + S * T
            pos = torch.arange(1, T + 1, device=device).repeat_interleave(S)
            q[start:end] = apply_rope_1d(q[start:end], pos, self.inv_t)
            k[start:end] = apply_rope_1d(k[start:end], pos, self.inv_t)

        # FlashAttention --------------------------------------------------
        q_b = q.unsqueeze(0)  # (1, L, Hq, dh)
        k_b = k.unsqueeze(0)  # (1, L, Hkv, dh)
        v_b = v.unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            dropout_p=self.dropout_p,
            is_causal=False,
        )[
            0
        ]  # remove batch -> (L, Hq, dh)

        # extract latest frame -------------------------------------------
        latest_start = S + S * T
        grid_out = out[latest_start : latest_start + S].reshape(S, self.D)

        # Apply output projection in the same dtype as the module
        grid_out = self.out_proj(grid_out)

        return grid_out.view(int(math.sqrt(S)), int(math.sqrt(S)), self.D)


######################################################################
# High-level stacks
######################################################################


class TGALayer(nn.Module):
    """One TemporalGridAttention → SwiGLU residual layer."""

    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        attn_dropout: float,
        ff_dropout: float,
        swiglu_factor: int,
        grid_size: int = 30,
    ) -> None:
        super().__init__()  # type: ignore
        self.norm_grid = RMSNorm(embed_dim)
        self.norm_hist = RMSNorm(embed_dim)
        self.norm_ff = RMSNorm(embed_dim)
        self.tga = TemporalGridAttention(
            embed_dim,
            num_q_heads,
            num_kv_heads,
            attn_dropout,
            grid_size,
        )
        self.ff = SwiGLU(embed_dim, factor=swiglu_factor, dropout=ff_dropout)

    def forward(self, grid: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        # Normalize both grid and history inputs
        normed_grid = self.norm_grid(grid)
        normed_hist = self.norm_hist(hist) if hist.numel() > 0 else hist

        # Apply attention with residual connection
        x = grid + self.tga(normed_grid, normed_hist)

        # Apply feed-forward with normalization and residual connection
        x = x + self.ff(self.norm_ff(x))

        return x


class TGABlock(nn.Module):
    """Stack *depth* TGALayers for greater capacity."""

    def __init__(
        self,
        depth: int = 1,
        embed_dim: int = 48,
        num_q_heads: int = 6,
        num_kv_heads: int = 2,
        attn_dropout: float = 0.01,
        ff_dropout: float = 0.01,
        swiglu_factor: int = 8,
        grid_size: int = 30,
    ) -> None:
        super().__init__()  # type: ignore
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TGALayer(
                    embed_dim,
                    num_q_heads,
                    num_kv_heads,
                    attn_dropout,
                    ff_dropout,
                    swiglu_factor,
                    grid_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, grid: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            grid = layer(grid, hist)
        return grid


######################################################################
# RepeatedTGA – iterative rollout
######################################################################


class RepeatedTGA(nn.Module):
    """Iteratively updates a 30×30 grid using a TGABlock.

    Each iteration:
      • (optional) local 3×3 depthwise Conv mixes each cell with its eight
        spatial neighbours;
      • run TGABlock (default 6 TGALayers);
      • append previous grid frame to history.

    Args mirror the lower-level components.  Outputs the new grid and the
    extended history tensor.
    """

    def __init__(
        self,
        embed_dim: int = 48,
        num_q_heads: int = 6,
        num_kv_heads: int = 2,
        attn_dropout: float = 0.01,
        ff_dropout: float = 0.01,
        swiglu_factor: int = 8,
    ) -> None:
        super().__init__()  # type: ignore
        self.tga_block = TGABlock(
            embed_dim=embed_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            swiglu_factor=swiglu_factor,
        )

    def forward(
        self,
        input: torch.Tensor,
        history: torch.Tensor,
        max_timesteps: int,
        local_mix: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = input
        for _ in range(max_timesteps):
            if local_mix:
                grid = self.local_mix(grid)
            grid = self.tga_block(grid, history)
            history = torch.cat((history, grid.unsqueeze(2)), dim=2)
        return grid, history

    def local_mix(self, grid: torch.Tensor) -> torch.Tensor:
        # Implement local mixing logic here
        return grid  # placeholder return
