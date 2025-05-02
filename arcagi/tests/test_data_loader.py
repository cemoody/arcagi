import torch


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
