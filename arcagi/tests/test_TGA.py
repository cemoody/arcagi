"""
Test suite for Rotary Position Embeddings (RoPE) functions from TGA.py

Tests both apply_rope_1d and apply_rope_2d functions with various scenarios
including edge cases, mathematical properties, and integration with the TGA module.
"""

import pytest
import torch

# Import the functions under test
from arcagi.TGA import _rope_pairwise, apply_rope_1d, apply_rope_2d


def test__rope_pairwise__applies_basic_rotation_correctly():
    """Test that _rope_pairwise applies basic rotation correctly"""
    x = torch.tensor([[[[1.0, 0.0]]]])  # Shape: (1, 1, 1, 2)
    cos = torch.tensor([[[1.0]]])  # cos(0) = 1 - Shape: (1, 1, 1)
    sin = torch.tensor([[[0.0]]])  # sin(0) = 0 - Shape: (1, 1, 1)

    result = _rope_pairwise(x, cos, sin)
    expected = torch.tensor([[[[1.0, 0.0]]]])

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test__rope_pairwise__applies_ninety_degree_rotation():
    """Test that _rope_pairwise correctly applies 90 degree rotation"""
    x = torch.tensor([[[[1.0, 0.0]]]])  # Shape: (1, 1, 1, 2)
    cos = torch.tensor([[[0.0]]])  # cos(π/2) = 0
    sin = torch.tensor([[[1.0]]])  # sin(π/2) = 1

    result = _rope_pairwise(x, cos, sin)
    expected = torch.tensor([[[[0.0, 1.0]]]])  # Should rotate to (0, 1)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test__apply_rope_1d__works_with_basic_inputs():
    """Test that apply_rope_1d works with basic inputs and returns correct shape"""
    L, H, d = 4, 2, 8  # d must be even for pairing
    x = torch.randn(L, H, d)
    pos = torch.arange(L).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(d // 2) / (d // 2)))

    result = apply_rope_1d(x, pos, inv_freq)

    assert result.shape == (L, H, d)
    assert torch.isfinite(result).all()


def test__apply_rope_1d__works_with_minimal_inputs():
    """Test that apply_rope_1d works with minimal tensor sizes"""
    L, H, d = 1, 1, 2  # Minimal case
    x = torch.tensor([[[1.0, 0.0]]])
    pos = torch.tensor([0.0])
    inv_freq = torch.tensor([1.0])

    result = apply_rope_1d(x, pos, inv_freq)

    assert result.shape == (1, 1, 2)
    assert torch.isfinite(result).all()


def test__apply_rope_1d__preserves_identity_at_zero_position():
    """Test that apply_rope_1d preserves input when position is zero"""
    L, H, d = 1, 1, 2
    x = torch.tensor([[[1.0, 2.0]]])
    pos = torch.zeros(L)
    inv_freq = torch.ones(d // 2)

    result = apply_rope_1d(x, pos, inv_freq)

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    # With zero positions, cos=1, sin=0, so should be close to identity
    torch.testing.assert_close(result, x, rtol=1e-5, atol=1e-6)


def test__apply_rope_2d__works_with_basic_inputs():
    """Test that apply_rope_2d works with basic inputs and returns correct shape"""
    L, H, d = 2, 1, 8  # d must be divisible by 4 for X/Y splitting
    x = torch.randn(L, H, d)
    coords_xy = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    inv_x = torch.ones(d // 4)  # d//4 = 2
    inv_y = torch.ones(d // 4)

    result = apply_rope_2d(x, coords_xy, inv_x, inv_y)

    assert result.shape == (L, H, d)
    assert torch.isfinite(result).all()


def test__apply_rope_2d__works_with_zero_coordinates():
    """Test that apply_rope_2d works correctly with zero coordinates"""
    L, H, d = 1, 1, 4  # Minimal case
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    coords_xy = torch.zeros(L, 2)
    inv_x = torch.ones(d // 4)  # d//4 = 1
    inv_y = torch.ones(d // 4)

    result = apply_rope_2d(x, coords_xy, inv_x, inv_y)

    assert result.shape == x.shape
    assert torch.isfinite(result).all()


def test__apply_rope_2d__requires_dimension_divisible_by_four():
    """Test that apply_rope_2d works with dimensions divisible by 4"""
    L, H, d = 1, 1, 8
    x = torch.randn(L, H, d)
    coords_xy = torch.zeros(L, 2)
    inv_x = torch.ones(d // 4)
    inv_y = torch.ones(d // 4)

    result = apply_rope_2d(x, coords_xy, inv_x, inv_y)

    assert result.shape == x.shape


def test__rope_functions__have_correct_signatures():
    """Test that rope functions can be called with expected signatures"""
    # Test 1D rope signature
    L, H, d = 1, 1, 2
    x = torch.randn(L, H, d)
    pos = torch.arange(L).float()
    inv_freq = torch.ones(d // 2)
    result = apply_rope_1d(x, pos, inv_freq)
    assert result.shape == x.shape

    # Test 2D rope signature
    L, H, d = 1, 1, 4
    x = torch.randn(L, H, d)
    coords = torch.zeros(L, 2)
    inv_x = torch.ones(d // 4)
    inv_y = torch.ones(d // 4)
    result = apply_rope_2d(x, coords, inv_x, inv_y)
    assert result.shape == x.shape


def test__rope_functions__work_with_tga_like_dimensions():
    """Test rope functions with dimensions similar to those used in TGA"""
    # Based on TGA usage: embed_dim=48, num_heads=6, head_dim=8
    L, H, d = 100, 6, 8  # 100 tokens, 6 heads, 8 head_dim

    # Test 1D rope
    x = torch.randn(L, H, d)
    pos = torch.arange(L).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(d // 2) / (d // 2)))
    result = apply_rope_1d(x, pos, inv_freq)
    assert result.shape == (L, H, d)

    # Test 2D rope
    coords = torch.randn(L, 2)  # Random coordinates
    inv_x = 1.0 / (10000 ** (torch.arange(d // 4) / (d // 4)))
    inv_y = inv_x.clone()
    result = apply_rope_2d(x, coords, inv_x, inv_y)
    assert result.shape == (L, H, d)


def test__rope_pairwise__handles_broadcasting_correctly():
    """Test that _rope_pairwise handles tensor broadcasting correctly"""
    # Create tensors that verify correct broadcasting behavior
    x = torch.randn(2, 1, 2, 2)  # (L, H, freq_pairs, 2)
    cos = torch.randn(2, 1, 2)  # (L, 1, freq_pairs)
    sin = torch.randn(2, 1, 2)  # (L, 1, freq_pairs)

    result = _rope_pairwise(x, cos, sin)

    # Should maintain the same shape as input x
    assert result.shape == x.shape


def test__rope_functions__handle_empty_tensors():
    """Test how rope functions handle edge cases like empty tensors"""
    # Test with empty tensor - this is expected to fail due to reshape ambiguity
    x_empty = torch.empty(0, 1, 4)
    pos_empty = torch.empty(0)
    inv_freq = torch.ones(2)

    # Empty tensors cause reshape ambiguity, which is expected behavior
    with pytest.raises(RuntimeError, match="cannot reshape tensor of 0 elements"):
        apply_rope_1d(x_empty, pos_empty, inv_freq)


def test__apply_rope_1d__preserves_temporal_locality():
    """Test that apply_rope_1d preserves temporal locality - nearby positions more similar than distant"""
    l, h, d = 5, 1, 8
    # Use consistent input to see positional encoding effects clearly
    x = torch.ones(l, h, d)

    # Create positions: 0, 1, 2, 3, 4
    pos = torch.arange(l).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(d // 2) / (d // 2)))

    result = apply_rope_1d(x, pos, inv_freq)

    # Test locality: pos[0] vs pos[1] should be more similar than pos[0] vs pos[4]
    emb_0 = result[0]  # Position 0
    emb_1 = result[1]  # Position 1 (nearby)
    emb_4 = result[4]  # Position 4 (distant)

    # Verify all embeddings have the same magnitude (RoPE should preserve magnitude)
    mag_0 = torch.norm(emb_0)
    mag_1 = torch.norm(emb_1)
    mag_4 = torch.norm(emb_4)

    assert torch.allclose(
        mag_0, mag_1, rtol=1e-6
    ), f"Magnitudes should be equal: {mag_0:.6f} vs {mag_1:.6f}"
    assert torch.allclose(
        mag_1, mag_4, rtol=1e-6
    ), f"Magnitudes should be equal: {mag_1:.6f} vs {mag_4:.6f}"

    # Calculate cosine similarities (now guaranteed to be fair comparison)
    sim_nearby = torch.cosine_similarity(emb_0.flatten(), emb_1.flatten(), dim=0)
    sim_distant = torch.cosine_similarity(emb_0.flatten(), emb_4.flatten(), dim=0)

    # Nearby positions should be more similar
    assert (
        sim_nearby > sim_distant
    ), f"Nearby similarity {sim_nearby} should be > distant similarity {sim_distant}"


def test__apply_rope_2d__preserves_spatial_locality():
    """Test that apply_rope_2d preserves spatial locality - nearby coordinates more similar than distant"""
    l, h, d = 9, 1, 8  # 3x3 grid
    # Use consistent input to see positional encoding effects clearly
    x = torch.ones(l, h, d)

    # Create 3x3 grid coordinates: (0,0), (0,1), (0,2), (1,0), ..., (2,2)
    coords = (
        torch.stack(
            torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij"), dim=-1
        )
        .reshape(-1, 2)
        .float()
    )

    inv_x = 1.0 / (10000 ** (torch.arange(d // 4) / (d // 4)))
    inv_y = inv_x.clone()

    result = apply_rope_2d(x, coords, inv_x, inv_y)

    # Get embeddings for specific positions
    # (0,0) is index 0, (0,1) is index 1, (2,2) is index 8
    emb_00 = result[0]  # (0,0)
    emb_01 = result[1]  # (0,1) - nearby
    emb_22 = result[8]  # (2,2) - distant

    # Verify all embeddings have the same magnitude (RoPE should preserve magnitude)
    mag_00 = torch.norm(emb_00)
    mag_01 = torch.norm(emb_01)
    mag_22 = torch.norm(emb_22)

    assert torch.allclose(
        mag_00, mag_01, rtol=1e-6
    ), f"Magnitudes should be equal: {mag_00:.6f} vs {mag_01:.6f}"
    assert torch.allclose(
        mag_01, mag_22, rtol=1e-6
    ), f"Magnitudes should be equal: {mag_01:.6f} vs {mag_22:.6f}"

    # Calculate cosine similarities (now guaranteed to be fair comparison)
    sim_nearby = torch.cosine_similarity(emb_00.flatten(), emb_01.flatten(), dim=0)
    sim_distant = torch.cosine_similarity(emb_00.flatten(), emb_22.flatten(), dim=0)

    # Nearby coordinates should be more similar
    assert (
        sim_nearby > sim_distant
    ), f"Nearby similarity {sim_nearby} should be > distant similarity {sim_distant}"


def test__apply_rope_1d__preserves_rotation_consistency():
    """Test that apply_rope_1d rotations are consistent across different input magnitudes"""
    l, h, d = 3, 1, 4

    # Test with two different input vectors
    x1 = torch.ones(l, h, d)
    x2 = torch.ones(l, h, d) * 2.0  # Different magnitude

    pos = torch.arange(l).float()
    inv_freq = torch.ones(d // 2)

    result1 = apply_rope_1d(x1, pos, inv_freq)
    result2 = apply_rope_1d(x2, pos, inv_freq)

    # The relative differences between positions should be similar
    # even though the input magnitudes are different
    diff1_01 = result1[1] - result1[0]
    diff1_12 = result1[2] - result1[1]

    diff2_01 = result2[1] - result2[0]
    diff2_12 = result2[2] - result2[1]

    # The ratios should be approximately the same (accounting for the 2x scale)
    ratio1 = torch.norm(diff1_01) / torch.norm(diff1_12)
    ratio2 = torch.norm(diff2_01) / torch.norm(diff2_12)

    torch.testing.assert_close(ratio1, ratio2, rtol=1e-3, atol=1e-4)


def test__apply_rope_1d__preserves_orthogonality():
    """Test that apply_rope_1d preserves orthogonality of orthogonal input vectors"""
    l, h, d = 2, 1, 4

    # Create orthogonal input vectors
    x = torch.zeros(l, h, d)
    x[0, 0, 0] = 1.0  # First vector: [1, 0, 0, 0]
    x[1, 0, 1] = 1.0  # Second vector: [0, 1, 0, 0]

    pos = torch.zeros(l)  # Same position for both
    inv_freq = torch.ones(d // 2)

    result = apply_rope_1d(x, pos, inv_freq)

    # At the same position, orthogonal vectors should remain orthogonal after rotation
    vec1 = result[0].flatten()
    vec2 = result[1].flatten()

    dot_product = torch.dot(vec1, vec2)

    # Should be very close to zero (allowing for numerical precision)
    assert (
        abs(dot_product) < 1e-5
    ), f"Orthogonal vectors should remain orthogonal after RoPE, got dot product: {dot_product}"
