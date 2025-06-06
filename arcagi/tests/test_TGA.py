"""
Test suite for Rotary Position Embeddings (RoPE) functions from TGA.py

Tests both apply_rope_1d and apply_rope_2d functions with various scenarios
including edge cases, mathematical properties, and integration with the TGA module.
"""

from typing import Any

import pytest
import torch

# Import the functions under test
from arcagi.TGA import apply_rope_1d, apply_rope_2d, rope_pairwise


def test__rope_pairwise__applies_basic_rotation_correctly():
    """Test that _rope_pairwise applies basic rotation correctly"""
    x = torch.tensor([[[[1.0, 0.0]]]])  # Shape: (1, 1, 1, 2)
    cos = torch.tensor([[[1.0]]])  # cos(0) = 1 - Shape: (1, 1, 1)
    sin = torch.tensor([[[0.0]]])  # sin(0) = 0 - Shape: (1, 1, 1)

    result = rope_pairwise(x, cos, sin)
    expected = torch.tensor([[[[1.0, 0.0]]]])

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test__rope_pairwise__applies_ninety_degree_rotation():
    """Test that _rope_pairwise correctly applies 90 degree rotation"""
    x = torch.tensor([[[[1.0, 0.0]]]])  # Shape: (1, 1, 1, 2)
    cos = torch.tensor([[[0.0]]])  # cos(π/2) = 0
    sin = torch.tensor([[[1.0]]])  # sin(π/2) = 1

    result = rope_pairwise(x, cos, sin)
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
    x = torch.tensor([[[1.0, 0.0]]])
    pos = torch.tensor([0.0])
    inv_freq = torch.tensor([1.0])

    result = apply_rope_1d(x, pos, inv_freq)

    assert result.shape == (1, 1, 2)
    assert torch.isfinite(result).all()


def test__apply_rope_1d__preserves_identity_at_zero_position():
    """Test that apply_rope_1d preserves input when position is zero"""
    L, d = 1, 2
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
    L, d = 1, 4
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
    l, h, d = 1, 1, 2
    x = torch.randn(l, h, d)
    pos = torch.arange(l).float()
    inv_freq = torch.ones(d // 2)
    result = apply_rope_1d(x, pos, inv_freq)
    assert result.shape == x.shape

    # Test 2D rope signature
    l, h, d = 1, 1, 4
    x = torch.randn(l, h, d)
    coords = torch.zeros(l, 2)
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

    result = rope_pairwise(x, cos, sin)

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
    mag_0 = torch.norm(emb_0)  # type: ignore
    mag_1 = torch.norm(emb_1)  # type: ignore
    mag_4 = torch.norm(emb_4)  # type: ignore

    assert torch.allclose(  # type: ignore
        mag_0, mag_1, rtol=1e-6  # type: ignore
    ), f"Magnitudes should be equal: {mag_0:.6f} vs {mag_1:.6f}"
    assert torch.allclose(  # type: ignore
        mag_1, mag_4, rtol=1e-6  # type: ignore
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
    mag_00 = torch.norm(emb_00)  # type: ignore
    mag_01 = torch.norm(emb_01)  # type: ignore
    mag_22 = torch.norm(emb_22)  # type: ignore

    assert torch.allclose(  # type: ignore
        mag_00, mag_01, rtol=1e-6  # type: ignore
    ), f"Magnitudes should be equal: {mag_00:.6f} vs {mag_01:.6f}"
    assert torch.allclose(  # type: ignore
        mag_01, mag_22, rtol=1e-6  # type: ignore
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
    ratio1 = torch.norm(diff1_01) / torch.norm(diff1_12)  # type: ignore
    ratio2 = torch.norm(diff2_01) / torch.norm(diff2_12)  # type: ignore

    torch.testing.assert_close(ratio1, ratio2, rtol=1e-3, atol=1e-4)  # type: ignore


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


######################################################################
# TemporalGridAttention Tests
######################################################################


@pytest.fixture
def tga_module() -> Any:
    """Fixture providing a TemporalGridAttention module with working parameters"""
    from arcagi.TGA import TemporalGridAttention

    # Use parameters that satisfy the constraints: same head dim for Q and KV
    module = TemporalGridAttention(
        embed_dim=24,  # 24/3=8 and 24/3=8, so head_dim=8 for both Q and KV
        num_q_heads=3,  # 24/3 = 8 head_dim, divisible by 8 and 4
        num_kv_heads=3,  # Same head dim as Q heads
        grid_size=3,  # Small grid for testing
        attn_dropout=0.0,  # Disable dropout for deterministic tests
    )
    return module


@pytest.fixture
def static_grid() -> torch.Tensor:
    """Fixture providing a static grid tensor"""
    return torch.randn(3, 3, 24)


@pytest.mark.parametrize("temporal_length", [0, 1, 7])
def test__temporal_grid_attention__module_creates_correctly(temporal_length: int):
    """Test that TemporalGridAttention module can be created with different temporal lengths"""
    from arcagi.TGA import TemporalGridAttention

    # Test that the module can be created
    module = TemporalGridAttention(
        embed_dim=24, num_q_heads=3, num_kv_heads=3, grid_size=3
    )

    # Test basic properties
    assert module.D == 24
    assert module.Hq == 3
    assert module.Hkv == 3
    assert module.S == 9  # 3x3 grid

    # Test that tensors have correct shapes
    static_grid = torch.randn(3, 3, 24)
    if temporal_length == 0:
        history_seq = torch.empty(3, 3, 0, 24)
    else:
        history_seq = torch.randn(3, 3, temporal_length, 24)

    # Test basic tensor properties without calling forward
    assert static_grid.shape == (3, 3, 24)
    assert history_seq.shape == (3, 3, temporal_length, 24)


@pytest.mark.parametrize("temporal_length", [0, 1, 7])
def test__temporal_grid_attention__forward_pass_shapes(
    tga_module: Any, temporal_length: int
):
    """Test that TemporalGridAttention forward pass returns correct shapes"""
    # Create inputs with the specified temporal length
    static_grid = torch.randn(3, 3, 24)
    if temporal_length == 0:
        history_seq = torch.empty(3, 3, 0, 24)
    else:
        history_seq = torch.randn(3, 3, temporal_length, 24)

    # Forward pass
    with torch.no_grad():  # Disable gradients for testing
        output = tga_module(static_grid, history_seq)  # type: ignore

    # Check output shape matches expected (3, 3, 24)
    assert output.shape == (3, 3, 24), f"Expected shape (3, 3, 24), got {output.shape}"  # type: ignore

    # Check output is finite (no NaN or inf values)
    assert torch.isfinite(output).all(), "Output contains non-finite values"  # type: ignore

    # Check output dtype matches input
    assert (
        output.dtype == static_grid.dtype  # type: ignore
    ), f"Output dtype {output.dtype} != input dtype {static_grid.dtype}"  # type: ignore


def test__temporal_grid_attention__handles_empty_history(tga_module: Any):
    """Test that TemporalGridAttention works correctly with empty history"""
    static_grid = torch.randn(3, 3, 24)
    history_seq = torch.empty(3, 3, 0, 24)

    with torch.no_grad():
        output = tga_module(static_grid, history_seq)  # type: ignore

    assert output.shape == static_grid.shape  # type: ignore
    assert torch.isfinite(output).all()  # type: ignore


def test__temporal_grid_attention__handles_different_history_lengths(tga_module: Any):
    """Test that TGA can handle different history lengths without errors"""
    static_grid = torch.randn(3, 3, 24)

    # Test with multiple history lengths
    for hist_len in [0, 1, 3, 5]:
        if hist_len == 0:
            history_seq = torch.empty(3, 3, 0, 24)
        else:
            history_seq = torch.randn(3, 3, hist_len, 24)

        with torch.no_grad():
            output = tga_module(static_grid, history_seq)  # type: ignore

        # Just check that it runs without error and produces valid output
        assert output.shape == (3, 3, 24), f"Failed for history length {hist_len}"  # type: ignore
        assert torch.isfinite(  # type: ignore
            output  # type: ignore
        ).all(), f"Non-finite outputs for history length {hist_len}"


def test__temporal_grid_attention__has_correct_parameters(tga_module: Any):
    """Test that TemporalGridAttention has the expected parameters and buffers"""
    # Check that the module has the expected attributes
    assert hasattr(tga_module, "init_embed")  # type: ignore
    assert hasattr(tga_module, "coords")  # type: ignore
    assert hasattr(tga_module, "inv_x")  # type: ignore
    assert hasattr(tga_module, "inv_y")  # type: ignore
    assert hasattr(tga_module, "inv_t")  # type: ignore

    # Check parameter shapes
    assert tga_module.init_embed.shape == (24,)  # type: ignore
    assert tga_module.coords.shape == (9, 2)  # 3x3 grid coordinates  # type: ignore

    # Check projection layers exist
    assert hasattr(tga_module, "q_proj")  # type: ignore
    assert hasattr(tga_module, "k_proj")  # type: ignore
    assert hasattr(tga_module, "v_proj")  # type: ignore
    assert hasattr(tga_module, "out_proj")  # type: ignore


def test__temporal_grid_attention__projection_shapes_correct(tga_module: Any):
    """Test that the projection layers have correct input/output dimensions"""
    # Q projection: embed_dim -> embed_dim
    assert tga_module.q_proj.in_features == 24  # type: ignore
    assert tga_module.q_proj.out_features == 24  # type: ignore

    # KV projections: embed_dim -> num_kv_heads * head_dim = 3 * 8 = 24
    assert tga_module.k_proj.in_features == 24  # type: ignore
    assert tga_module.k_proj.out_features == 24  # 3 heads * 8 head_dim  # type: ignore
    assert tga_module.v_proj.in_features == 24  # type: ignore
    assert tga_module.v_proj.out_features == 24  # type: ignore

    # Output projection: embed_dim -> embed_dim
    assert tga_module.out_proj.in_features == 24  # type: ignore
    assert tga_module.out_proj.out_features == 24  # type: ignore


def test__temporal_grid_attention__works_with_half_precision():
    """Test that TGA works correctly when converted to half precision"""
    from arcagi.TGA import TemporalGridAttention

    module = TemporalGridAttention(
        embed_dim=24, num_q_heads=3, num_kv_heads=3, grid_size=3, attn_dropout=0.0
    ).half()  # Convert to half precision

    # Test with half precision inputs
    static_grid = torch.randn(3, 3, 24, dtype=torch.float16)
    history_seq = torch.randn(3, 3, 2, 24, dtype=torch.float16)

    with torch.no_grad():
        output = module(static_grid, history_seq)

    # Output should remain in half precision
    assert output.dtype == torch.float16, f"Expected float16 output, got {output.dtype}"  # type: ignore
    assert output.shape == (3, 3, 24)  # type: ignore
    assert torch.isfinite(output).all()  # type: ignore


def test__tga_layer__forward_pass_shapes_and_dtype():
    """Test that TGALayer forward pass returns correct shapes and dtype"""
    from arcagi.TGA import TGALayer

    # Create TGALayer with compatible parameters
    layer = TGALayer(
        embed_dim=24,
        num_q_heads=3,
        num_kv_heads=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        swiglu_factor=4,
        grid_size=3,  # Match the test grid size
    )

    # Create input tensors
    grid = torch.randn(3, 3, 24)  # (H, W, D)
    history = torch.randn(3, 3, 5, 24)  # (H, W, T, D)

    # Forward pass
    with torch.no_grad():
        output = layer(grid, history)

    # Check output shape matches input grid shape
    assert (
        output.shape == grid.shape
    ), f"Expected shape {grid.shape}, got {output.shape}"

    # Check output dtype matches input dtype
    assert (
        output.dtype == grid.dtype
    ), f"Expected dtype {grid.dtype}, got {output.dtype}"

    # Check output is finite (no NaN or inf values)
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # Test with empty history
    empty_history = torch.empty(3, 3, 0, 24)
    with torch.no_grad():
        output_empty = layer(grid, empty_history)

    assert (
        output_empty.shape == grid.shape
    ), f"Expected shape {grid.shape} with empty history, got {output_empty.shape}"
    assert torch.isfinite(
        output_empty
    ).all(), "Output with empty history contains non-finite values"
