#!/usr/bin/env python3
"""
Terminal-based visualizer for order2 features (edges between cells).
Uses Unicode block elements to display edges in 8 directions from each cell.

Example Output:
    Single cell with all 8 edges:

      ▛▜
      ▙▟


    Grid pattern:
     ▐▌▐▌
    ▄▟▙▟▙▄
    ▀▜▛▜▛▀
     ▐▌▐▌
"""
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch

# Unicode block elements for edge visualization
GLYPH: Dict[FrozenSet[str], str] = {
    frozenset(): " ",
    frozenset({"N"}): "▀",
    frozenset({"S"}): "▄",
    frozenset({"E"}): "▐",
    frozenset({"W"}): "▌",
    frozenset({"NW"}): "▘",
    frozenset({"NE"}): "▝",
    frozenset({"SW"}): "▖",
    frozenset({"SE"}): "▗",
    # Multi-edge combinations for smoother visuals
    frozenset({"N", "W", "NW"}): "▛",
    frozenset({"N", "E", "NE"}): "▜",
    frozenset({"S", "W", "SW"}): "▙",
    frozenset({"S", "E", "SE"}): "▟",
    frozenset({"N", "W"}): "▛",
    frozenset({"N", "E"}): "▜",
    frozenset({"S", "W"}): "▙",
    frozenset({"S", "E"}): "▟",
    frozenset({"N", "S"}): "█",
    frozenset({"E", "W"}): "█",
    frozenset({"N", "S", "E", "W"}): "█",
}


def order2_to_edges(
    order2_features: torch.Tensor, height: int, width: int
) -> List[List[Set[str]]]:
    """
    Convert order2 features tensor to edge representation.

    Args:
        order2_features: Tensor of shape (H, W, 8) where the last dimension contains
                        binary values for edges to 8 neighbors in order:
                        [N, NE, E, SE, S, SW, W, NW]
        height: Height of the grid
        width: Width of the grid

    Returns:
        Grid of sets where each cell contains its active edge directions
    """
    # Direction mapping - indices in the order2 features
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    # Initialize grid
    grid_edges = [[set() for _ in range(width)] for _ in range(height)]

    # Convert tensor to edge sets
    for y in range(height):
        for x in range(width):
            for i, direction in enumerate(directions):
                if order2_features[y, x, i] > 0:
                    grid_edges[y][x].add(direction)

    return grid_edges


def render_edges(grid_edges: List[List[Set[str]]]) -> str:
    """
    Render edge grid to terminal using Unicode block elements.

    Args:
        grid_edges: Grid where each cell contains a set of edge directions

    Returns:
        String representation using Unicode block elements
    """
    H = len(grid_edges)
    W = len(grid_edges[0]) if H > 0 else 0

    # Create terminal grid (2x size of logical grid)
    term = [[" "] * (2 * W) for _ in range(2 * H)]

    for y in range(H):
        for x in range(W):
            dirs = grid_edges[y][x]

            # Top-left quadrant: N, NW, W
            tl_dirs = dirs & {"N", "NW", "W"}
            term[2 * y][2 * x] = GLYPH.get(frozenset(tl_dirs), " ")

            # Top-right quadrant: N, NE, E
            tr_dirs = dirs & {"N", "NE", "E"}
            term[2 * y][2 * x + 1] = GLYPH.get(frozenset(tr_dirs), " ")

            # Bottom-left quadrant: S, SW, W
            bl_dirs = dirs & {"S", "SW", "W"}
            term[2 * y + 1][2 * x] = GLYPH.get(frozenset(bl_dirs), " ")

            # Bottom-right quadrant: S, SE, E
            br_dirs = dirs & {"S", "SE", "E"}
            term[2 * y + 1][2 * x + 1] = GLYPH.get(frozenset(br_dirs), " ")

    return "\n".join("".join(row) for row in term)


def relshow(
    order2_features: torch.Tensor, title: Optional[str] = None, show_grid: bool = False
) -> None:
    """
    Display order2 features (edges between cells) in the terminal.

    Args:
        order2_features: Tensor of shape (H, W, 8) containing binary edge features
                        for 8 directions: [N, NE, E, SE, S, SW, W, NW]
        title: Optional title to display above the visualization
        show_grid: If True, overlay a grid to show cell boundaries
    """
    # Validate input
    if order2_features.dim() != 3:
        raise ValueError(
            f"Expected 3D tensor (H, W, 8), got {order2_features.dim()}D tensor"
        )

    if order2_features.shape[2] != 8:
        raise ValueError(f"Expected 8 edge features, got {order2_features.shape[2]}")

    height, width = order2_features.shape[:2]

    # Convert to edge representation
    grid_edges = order2_to_edges(order2_features, height, width)

    # Print title if provided
    if title:
        print(f"\n\033[1m{title}\033[0m")

    # Render the edges
    rendered = render_edges(grid_edges)

    # Add grid overlay if requested
    if show_grid:
        lines = rendered.split("\n")
        # Add horizontal separators
        for i in range(0, len(lines), 2):
            if i > 0:
                print("─" * (width * 2))
            print(lines[i] if i < len(lines) else "")
            if i + 1 < len(lines):
                print(lines[i + 1])
    else:
        print(rendered)


def create_test_pattern(height: int, width: int, pattern: str = "grid") -> torch.Tensor:
    """
    Create test patterns for edge visualization.

    Args:
        height: Height of the grid
        width: Width of the grid
        pattern: Type of pattern ("grid", "diagonal", "spiral", "random")

    Returns:
        Tensor of shape (H, W, 8) with edge features
    """
    features = torch.zeros((height, width, 8), dtype=torch.float32)

    if pattern == "grid":
        # Horizontal edges (E and W)
        for y in range(height):
            for x in range(width - 1):
                features[y, x, 2] = 1  # E
                features[y, x + 1, 6] = 1  # W

        # Vertical edges (S and N)
        for y in range(height - 1):
            for x in range(width):
                features[y, x, 4] = 1  # S
                features[y + 1, x, 0] = 1  # N

    elif pattern == "diagonal":
        # Main diagonal (NW to SE)
        for i in range(min(height - 1, width - 1)):
            features[i, i, 3] = 1  # SE
            features[i + 1, i + 1, 7] = 1  # NW

    elif pattern == "spiral":
        # Create a spiral pattern
        y, x = height // 2, width // 2
        dy, dx = 0, 1

        for _ in range(min(height, width) // 2):
            for _ in range(2):
                for _ in range(max(1, min(height, width) // 4)):
                    if 0 <= y < height and 0 <= x < width:
                        # Add edge in current direction
                        if dx == 1:  # Moving East
                            features[y, x, 2] = 1
                        elif dx == -1:  # Moving West
                            features[y, x, 6] = 1
                        elif dy == 1:  # Moving South
                            features[y, x, 4] = 1
                        elif dy == -1:  # Moving North
                            features[y, x, 0] = 1

                    y += dy
                    x += dx

                # Turn right
                dy, dx = dx, -dy

    elif pattern == "random":
        # Random edges
        features = torch.rand((height, width, 8)) > 0.7
        features = features.float()

    return features


def compute_order2_features(
    color_grid: torch.Tensor, target_color: Optional[int] = None
) -> torch.Tensor:
    """
    Compute order2 features from a color grid.

    This function identifies edges between cells based on color differences.
    If target_color is specified, it creates edges where the target color
    connects to different colors. Otherwise, it creates edges wherever
    adjacent cells have different colors.

    Args:
        color_grid: Tensor of shape (H, W) containing color values
        target_color: Optional specific color to track connections from

    Returns:
        Tensor of shape (H, W, 8) with binary edge features
    """
    height, width = color_grid.shape
    features = torch.zeros((height, width, 8), dtype=torch.float32)

    # Direction offsets: N, NE, E, SE, S, SW, W, NW
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    dx = [0, 1, 1, 1, 0, -1, -1, -1]

    for y in range(height):
        for x in range(width):
            current_color = color_grid[y, x]

            # Check if we should process this cell
            if target_color is not None and current_color != target_color:
                continue

            # Check each neighbor
            for i in range(8):
                ny, nx = y + dy[i], x + dx[i]

                # Check bounds
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_color = color_grid[ny, nx]

                    # Create edge if colors differ
                    if current_color != neighbor_color:
                        features[y, x, i] = 1

    return features


if __name__ == "__main__":
    # Test basic patterns
    print("\033[1m========== Edge Visualization Tests ==========\033[0m")

    # Test 1: Grid pattern
    grid_features = create_test_pattern(5, 5, "grid")
    relshow(grid_features, title="Grid Pattern (5x5)")

    # Test 2: Diagonal pattern
    diag_features = create_test_pattern(6, 6, "diagonal")
    relshow(diag_features, title="Diagonal Pattern (6x6)")

    # Test 3: Random edges
    random_features = create_test_pattern(4, 4, "random")
    relshow(random_features, title="Random Edges (4x4)", show_grid=True)

    # Test 4: Manual pattern - square with diagonals
    manual_features = torch.zeros((3, 3, 8), dtype=torch.float32)
    # Create a square
    manual_features[0, 0, 2] = 1  # E from top-left
    manual_features[0, 0, 4] = 1  # S from top-left
    manual_features[0, 1, 2] = 1  # E from top-middle
    manual_features[0, 1, 4] = 1  # S from top-middle
    manual_features[0, 2, 4] = 1  # S from top-right
    manual_features[0, 2, 6] = 1  # W from top-right
    manual_features[1, 0, 4] = 1  # S from middle-left
    manual_features[1, 0, 2] = 1  # E from middle-left
    manual_features[1, 2, 4] = 1  # S from middle-right
    manual_features[1, 2, 6] = 1  # W from middle-right
    manual_features[2, 0, 2] = 1  # E from bottom-left
    manual_features[2, 0, 0] = 1  # N from bottom-left
    manual_features[2, 1, 2] = 1  # E from bottom-middle
    manual_features[2, 1, 0] = 1  # N from bottom-middle
    manual_features[2, 2, 0] = 1  # N from bottom-right
    manual_features[2, 2, 6] = 1  # W from bottom-right
    # Add diagonals
    manual_features[0, 0, 3] = 1  # SE from top-left
    manual_features[0, 2, 5] = 1  # SW from top-right
    manual_features[2, 0, 1] = 1  # NE from bottom-left
    manual_features[2, 2, 7] = 1  # NW from bottom-right

    relshow(manual_features, title="Square with Diagonals (3x3)")

    # Test 5: All edges from center cell
    center_features = torch.zeros((3, 3, 8), dtype=torch.float32)
    # Set all edges from center cell
    for i in range(8):
        center_features[1, 1, i] = 1

    relshow(center_features, title="All Edges from Center (3x3)")

    # Test 6: Color-based edges
    print("\n\033[1m========== Color-Based Edge Detection ==========\033[0m")

    # Create a simple color pattern
    color_grid = torch.tensor([[1, 1, 2, 2], [1, 3, 3, 2], [4, 3, 3, 5], [4, 4, 5, 5]])

    # Show edges for all color boundaries
    all_edges = compute_order2_features(color_grid)
    relshow(all_edges, title="All Color Boundaries (4x4)")

    # Show edges only from color 3
    color3_edges = compute_order2_features(color_grid, target_color=3)
    relshow(color3_edges, title="Edges from Color 3 Only (4x4)")

    # Test 7: Integration with terminal_imshow
    try:
        from terminal_imshow import imshow

        print("\n\033[1m========== Combined Color and Edge Display ==========\033[0m")
        print("\nOriginal Color Grid:")
        imshow(color_grid, show_legend=True)

        print("\nCorresponding Edge Visualization:")
        relshow(all_edges, title="Edge Structure")
    except ImportError:
        print("\n(Skipping terminal_imshow integration test)")

    # Test 8: Larger pattern with mixed edges
    large_features = torch.zeros((8, 8, 8), dtype=torch.float32)
    # Create some interesting patterns
    for y in range(8):
        for x in range(8):
            # Checkerboard of different edge patterns
            if (x + y) % 2 == 0:
                # Even cells: cardinal directions
                if x < 7:
                    large_features[y, x, 2] = 1  # E
                if y < 7:
                    large_features[y, x, 4] = 1  # S
                if x > 0:
                    large_features[y, x, 6] = 1  # W
                if y > 0:
                    large_features[y, x, 0] = 1  # N
            else:
                # Odd cells: diagonal directions
                if x < 7 and y > 0:
                    large_features[y, x, 1] = 1  # NE
                if x < 7 and y < 7:
                    large_features[y, x, 3] = 1  # SE
                if x > 0 and y < 7:
                    large_features[y, x, 5] = 1  # SW
                if x > 0 and y > 0:
                    large_features[y, x, 7] = 1  # NW

    relshow(large_features, title="Checkerboard Edge Pattern (8x8)")
