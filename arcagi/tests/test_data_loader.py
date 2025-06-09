import torch

from arcagi.data_loader import repeat_and_permute

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot_from_int(mat: torch.Tensor, num_classes: int = 11) -> torch.Tensor:  # type: ignore
    """Convert an integer tensor of shape (H, W) (or (B, H, W)) to one-hot with
    channels on the last axis.
    """

    assert mat.dim() in (2, 3)
    if mat.dim() == 2:
        mat = mat.unsqueeze(0)
    batch: int = mat.shape[0]
    height: int = mat.shape[1]
    width: int = mat.shape[2]

    one_hot: torch.Tensor = torch.zeros(
        (batch, height, width, num_classes), dtype=torch.float
    )
    valid_mask: torch.Tensor = mat >= 0
    # Only scatter where colour is in [0, num_classes-1]
    indices: torch.Tensor = mat.clone()
    indices[~valid_mask] = 0  # placeholder index for invalid positions
    one_hot[valid_mask.unsqueeze(-1).expand_as(one_hot)] = (
        0.0  # ensure zero (already zero)
    )
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return one_hot


def _categorical_from_one_hot(one_hot: torch.Tensor) -> torch.Tensor:  # type: ignore
    cat: torch.Tensor = torch.argmax(one_hot, dim=-1)
    sum_: torch.Tensor = one_hot.sum(dim=-1)
    cat[sum_ == 0] = -1
    return cat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_batch_generate_color_mapping_independent_per_sample() -> None:
    """Verify that *batch_generate_color_mapping* applies
    sample-wise-independent permutations while keeping input/output aligned."""

    from arcagi.data_loader import batch_generate_color_mapping

    # Construct a random batch of categorical colour grids (B, H, W) with
    # values in [0, 10]. Use -1 for ~20% of cells.
    torch.manual_seed(123)  # type: ignore[arg-type]
    batch_size: int = 5
    height: int = 30
    width: int = 30

    rand_vals: torch.Tensor = torch.randint(0, 11, (batch_size, height, width))
    mask_invalid: torch.Tensor = torch.rand(batch_size, height, width) < 0.2
    rand_vals = rand_vals.masked_fill(mask_invalid, -1)

    inputs_one_hot: torch.Tensor = _one_hot_from_int(rand_vals)
    # For outputs, just copy inputs so colours are aligned before permutation.
    outputs_one_hot: torch.Tensor = inputs_one_hot.clone()

    perm_inputs, perm_outputs = batch_generate_color_mapping(
        inputs_one_hot, outputs_one_hot
    )

    # Shapes must be preserved.
    assert perm_inputs.shape == inputs_one_hot.shape
    assert perm_outputs.shape == outputs_one_hot.shape

    # Convert back to categorical for easier reasoning.
    orig_cat: torch.Tensor = rand_vals  # (B, H, W)
    perm_inp_cat: torch.Tensor = _categorical_from_one_hot(perm_inputs)
    perm_out_cat: torch.Tensor = _categorical_from_one_hot(perm_outputs)

    # For each sample check:
    # 1) There exists a one-to-one mapping between orig_cat and perm_inp_cat.
    # 2) The same mapping applies to perm_out_cat.
    # 3) The mapping is a permutation of 0-10.
    # Additionally, at least two samples should have distinct permutations.

    mappings: list[dict[int, int]] = []  # type: ignore

    for b in range(batch_size):
        mapping: dict[int, int] = {}
        for h in range(height):
            for w in range(width):
                old_col: int = int(orig_cat[b, h, w].item())
                if old_col == -1:
                    # Ignore padding cells.
                    continue
                new_col: int = int(perm_inp_cat[b, h, w].item())
                # Consistency: if we've already seen old_col, mapping must match.
                if old_col in mapping:
                    assert mapping[old_col] == new_col
                else:
                    mapping[old_col] = new_col
        # Ensure it is a permutation (11 distinct colours).
        assert set(mapping.keys()) == set(range(11))
        assert set(mapping.values()) == set(range(11))

        # Verify outputs follow the same mapping.
        for h in range(height):
            for w in range(width):
                old_col: int = int(orig_cat[b, h, w].item())
                if old_col == -1:
                    continue
                expected_new: int = mapping[old_col]
                actual_new: int = int(perm_out_cat[b, h, w].item())
                assert expected_new == actual_new

        mappings.append(mapping)

    # Check that at least two mappings differ (highly likely with random perms).
    unique_mappings: set[tuple[int, ...]] = {
        tuple(m[k] for k in sorted(m)) for m in mappings
    }
    assert len(unique_mappings) > 1, "Expected independent permutations across samples"


def test_repeat_and_permute():
    """Test that repeat_and_permute correctly repeats and permutes the input and output tensors."""
    # Create a small test dataset
    batch_size: int = 3
    height: int = 30
    width: int = 30
    channels: int = 11
    n_repeats: int = 4

    # Create random one-hot encoded inputs and outputs
    rand_vals: torch.Tensor = torch.randint(0, 11, (batch_size, height, width))
    inputs: torch.Tensor = torch.zeros((batch_size, height, width, channels))
    outputs: torch.Tensor = torch.zeros((batch_size, height, width, channels))

    # Fill in one-hot encoded values
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                color: int = int(rand_vals[b, h, w].item())
                inputs[b, h, w, color] = 1.0
                outputs[b, h, w, color] = 1.0

    # Apply repeat_and_permute
    repeated_inputs, repeated_outputs = repeat_and_permute(inputs, outputs, n_repeats)

    # Check shapes
    assert repeated_inputs.shape == (batch_size * n_repeats, height, width, channels)
    assert repeated_outputs.shape == (batch_size * n_repeats, height, width, channels)

    # Convert to categorical for easier verification
    repeated_inputs_cat: torch.Tensor = _categorical_from_one_hot(repeated_inputs)
    repeated_outputs_cat: torch.Tensor = _categorical_from_one_hot(repeated_outputs)

    # Verify that each original sample is repeated n_repeats times with different permutations
    for orig_idx in range(batch_size):
        # Get the indices in the repeated tensor for this original sample
        repeat_indices: list[int] = [
            orig_idx + i * batch_size for i in range(n_repeats)
        ]

        # For each repeated instance, verify it's a valid permutation of the original
        permutation_mappings: list[dict[int, int]] = []

        for repeat_idx in repeat_indices:
            # Build the color mapping for this repeated instance
            mapping: dict[int, int] = {}

            # Sample a few points to build the mapping
            for h in range(0, height, 5):  # Sample every 5th point to save time
                for w in range(0, width, 5):
                    orig_color: int = int(rand_vals[orig_idx, h, w].item())
                    if orig_color == -1:
                        continue
                    new_color: int = int(repeated_inputs_cat[repeat_idx, h, w].item())

                    if orig_color in mapping:
                        assert (
                            mapping[orig_color] == new_color
                        ), "Inconsistent mapping within a sample"
                    else:
                        mapping[orig_color] = new_color

            # Verify it's a valid permutation
            if len(mapping) == 11:  # Only check if we found all colors
                assert set(mapping.keys()) == set(range(11))
                assert set(mapping.values()) == set(range(11))

            # Verify inputs and outputs have the same mapping
            for h in range(0, height, 5):
                for w in range(0, width, 5):
                    orig_color: int = int(rand_vals[orig_idx, h, w].item())
                    if orig_color == -1 or orig_color not in mapping:
                        continue
                    expected_color: int = mapping[orig_color]
                    actual_color_input: int = int(
                        repeated_inputs_cat[repeat_idx, h, w].item()
                    )
                    actual_color_output: int = int(
                        repeated_outputs_cat[repeat_idx, h, w].item()
                    )

                    assert expected_color == actual_color_input
                    assert expected_color == actual_color_output

            permutation_mappings.append(mapping)

        # Check that the permutations are different for each repeat
        if all(len(m) == 11 for m in permutation_mappings):
            unique_mappings: set[tuple[int, ...]] = {
                tuple(m[k] for k in sorted(m)) for m in permutation_mappings
            }
            # With random permutations, it's highly unlikely all would be the same
            assert (
                len(unique_mappings) > 1
            ), "Expected different permutations for repeated samples"


def test_apply_mixing_steps():
    """Test that apply_mixing_steps correctly creates intermediate states."""
    from arcagi.data_loader import apply_mixing_steps

    # Create a simple test case with known values
    batch_size: int = 4
    height: int = 30
    width: int = 30
    channels: int = 11
    n_steps: int = 4

    # Create simple inputs and outputs where we can verify the mixing
    # Input: all pixels are color 0 (one-hot)
    # Output: all pixels are color 1 (one-hot)
    inputs: torch.Tensor = torch.zeros((batch_size, height, width, channels))
    outputs: torch.Tensor = torch.zeros((batch_size, height, width, channels))

    inputs[:, :, :, 0] = 1.0  # All pixels are color 0
    outputs[:, :, :, 1] = 1.0  # All pixels are color 1

    # Apply mixing steps
    mixed_inputs, mixed_outputs = apply_mixing_steps(inputs, outputs, n_steps)

    # Check shapes are preserved
    assert mixed_inputs.shape == inputs.shape
    assert mixed_outputs.shape == outputs.shape

    # For each sample, verify the mixing is correct
    for b in range(batch_size):
        # Get the value at channel 0 and 1 for a sample pixel
        input_ch0: float = mixed_inputs[b, 0, 0, 0].item()
        input_ch1: float = mixed_inputs[b, 0, 0, 1].item()
        output_ch0: float = mixed_outputs[b, 0, 0, 0].item()
        output_ch1: float = mixed_outputs[b, 0, 0, 1].item()

        # The fraction f should be one of {0, 0.25, 0.5, 0.75}
        # For mixed_input: ch0 = f, ch1 = (1-f)
        # For mixed_output: ch0 = (f - 0.25), ch1 = (1 - f + 0.25)

        # Determine which f was used based on input_ch0
        f: float = input_ch0

        # Verify f is one of the expected values
        assert f in [0.0, 0.25, 0.5, 0.75] or abs(f - round(f * 4) / 4) < 1e-6

        # Verify the mixing formulas
        assert abs(input_ch1 - (1 - f)) < 1e-6

        # For output, when f = 0, we expect ch0 = -0.25, ch1 = 1.25
        # But it should be clamped to [0, 1]
        expected_output_ch0: float = max(0.0, f - 0.25)
        expected_output_ch1: float = min(1.0, 1 - f + 0.25)

        assert abs(output_ch0 - expected_output_ch0) < 1e-6
        assert abs(output_ch1 - expected_output_ch1) < 1e-6

        # Verify all channels sum to 1 (valid probability distribution)
        input_sum: float = mixed_inputs[b, 0, 0, :].sum().item()
        output_sum: float = mixed_outputs[b, 0, 0, :].sum().item()
        assert abs(input_sum - 1.0) < 1e-6
        assert abs(output_sum - 1.0) < 1e-6

    # Test with more complex case - different colors
    torch.manual_seed(42)  # type: ignore[arg-type]
    rand_input_colors: torch.Tensor = torch.randint(0, 11, (batch_size, height, width))
    rand_output_colors: torch.Tensor = torch.randint(0, 11, (batch_size, height, width))

    complex_inputs: torch.Tensor = _one_hot_from_int(rand_input_colors)
    complex_outputs: torch.Tensor = _one_hot_from_int(rand_output_colors)

    mixed_inputs, mixed_outputs = apply_mixing_steps(
        complex_inputs, complex_outputs, n_steps
    )

    # Verify probability distributions are valid
    assert torch.all(mixed_inputs >= 0)
    assert torch.all(mixed_inputs <= 1)
    assert torch.all(mixed_outputs >= 0)
    assert torch.all(mixed_outputs <= 1)

    # Verify sums are approximately 1
    input_sums: torch.Tensor = mixed_inputs.sum(dim=-1)
    output_sums: torch.Tensor = mixed_outputs.sum(dim=-1)
    assert torch.allclose(input_sums, torch.ones_like(input_sums), atol=1e-6)
    assert torch.allclose(output_sums, torch.ones_like(output_sums), atol=1e-6)
