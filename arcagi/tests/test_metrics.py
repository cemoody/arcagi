"""Tests for metrics utilities."""

from typing import Dict

import pytest
import torch

from arcagi.models import Batch, BatchData
from arcagi.utils.metrics import training_index_metrics


class TestTrainingIndexMetrics:
    """Tests for training_index_metrics function."""

    def test_basic_functionality(self):
        """Test basic metric computation."""
        # Create mock data
        inp_col = torch.tensor(
            [[[0, 1], [2, 3]], [[1, 2], [3, 0]]]  # First example  # Second example
        )  # [B=2, H=2, W=2]

        inp_msk = torch.tensor(
            [
                [[True, True], [True, False]],  # First example has 3 active pixels
                [[True, True], [False, True]],  # Second example has 3 active pixels
            ]
        )  # [B=2, H=2, W=2]

        # Create predictions (some correct, some incorrect)
        col_logits = torch.zeros(2, 2, 2, 10)  # [B, H, W, num_colors]
        # Set predictions for first example
        col_logits[0, 0, 0, 0] = 1.0  # Correct (0)
        col_logits[0, 0, 1, 2] = 1.0  # Incorrect (should be 1)
        col_logits[0, 1, 0, 2] = 1.0  # Correct (2)
        col_logits[0, 1, 1, 3] = 1.0  # Correct (3) but masked
        # Set predictions for second example
        col_logits[1, 0, 0, 1] = 1.0  # Correct (1)
        col_logits[1, 0, 1, 2] = 1.0  # Correct (2)
        col_logits[1, 1, 0, 0] = 1.0  # Incorrect (should be 3) and masked
        col_logits[1, 1, 1, 0] = 1.0  # Correct (0)

        msk_logits = torch.zeros(2, 2, 2, 1)  # [B, H, W, 1]
        # Set mask predictions
        msk_logits[0, :, :, 0] = torch.tensor([[1.0, 1.0], [1.0, -1.0]])  # All correct
        msk_logits[1, :, :, 0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # One incorrect

        # Create batch data
        inp = Batch(
            one=torch.zeros(2, 2, 2, 10),
            fea=torch.zeros(2, 2, 2, 147),
            col=inp_col,
            msk=inp_msk,
            colf=inp_col.reshape(2, -1),
            mskf=inp_msk.reshape(2, -1).float(),
            idx=torch.tensor([5, 10]),  # Example indices
        )

        out = Batch(
            one=torch.zeros(2, 2, 2, 10),
            fea=torch.zeros(2, 2, 2, 147),
            col=torch.zeros(2, 2, 2).long(),
            msk=torch.zeros(2, 2, 2).bool(),
            colf=torch.zeros(2, 4).long(),
            mskf=torch.zeros(2, 4),
            idx=torch.tensor([5, 10]),
        )

        batch_data = BatchData(inp=inp, out=out)

        # Run the function
        metrics: Dict[int, Dict[str, float]] = {}
        result = training_index_metrics(
            batch_data, col_logits, msk_logits, "input", metrics
        )

        # Check results
        assert 5 in result
        assert 10 in result

        # First example (idx=5): 3 active pixels, 1 color incorrect within mask, 0 mask incorrect
        assert result[5]["input_n_masked_pixels"] == 3.0
        assert result[5]["input_n_incorrect_num_color"] == 1.0  # Only the pixel at [0,0,1]
        assert result[5]["input_n_incorrect_num_mask"] == 0.0

        # Second example (idx=10): 3 active pixels, 0 color incorrect within mask, 1 mask incorrect
        assert result[10]["input_n_masked_pixels"] == 3.0
        assert result[10]["input_n_incorrect_num_color"] == 0.0  # Incorrect pixel at [1,1,0] is masked
        assert result[10]["input_n_incorrect_num_mask"] == 1.0  # One incorrect mask prediction

    def test_duplicate_indices(self):
        """Test handling of duplicate indices in a batch."""
        # Create data with duplicate indices
        inp_col = torch.tensor(
            [[[0, 1], [2, 3]], [[1, 2], [3, 0]], [[2, 3], [0, 1]]]
        )  # [B=3, H=2, W=2]

        inp_msk = torch.ones(3, 2, 2).bool()  # All pixels active

        col_logits = torch.zeros(3, 2, 2, 10)
        # Make all predictions correct
        for b in range(3):
            for h in range(2):
                for w in range(2):
                    col_logits[b, h, w, inp_col[b, h, w]] = 1.0

        msk_logits = torch.ones(3, 2, 2, 1)  # All mask predictions correct

        # Create batch with duplicate indices
        inp = Batch(
            one=torch.zeros(3, 2, 2, 10),
            fea=torch.zeros(3, 2, 2, 147),
            col=inp_col,
            msk=inp_msk,
            colf=inp_col.reshape(3, -1),
            mskf=inp_msk.reshape(3, -1).float(),
            idx=torch.tensor([5, 10, 5]),  # idx 5 appears twice
        )

        out = Batch(
            one=torch.zeros(3, 2, 2, 10),
            fea=torch.zeros(3, 2, 2, 147),
            col=torch.zeros(3, 2, 2).long(),
            msk=torch.zeros(3, 2, 2).bool(),
            colf=torch.zeros(3, 4).long(),
            mskf=torch.zeros(3, 4),
            idx=torch.tensor([5, 10, 5]),
        )

        batch_data = BatchData(inp=inp, out=out)

        metrics: Dict[int, Dict[str, float]] = {}
        result = training_index_metrics(
            batch_data, col_logits, msk_logits, "input", metrics
        )

        # Check that metrics are averaged for duplicate indices
        assert result[5]["input_n_masked_pixels"] == 4.0  # Average: (4 + 4) / 2
        assert result[5]["input_n_incorrect_num_color"] == 0.0  # All correct
        assert result[5]["input_n_incorrect_num_mask"] == 0.0  # All correct
        assert result[10]["input_n_masked_pixels"] == 4.0  # Only one instance
        assert result[10]["input_n_incorrect_num_color"] == 0.0
        assert result[10]["input_n_incorrect_num_mask"] == 0.0

    def test_accumulation_across_batches(self):
        """Test that metrics accumulate correctly across multiple calls."""
        # Simple 1x1 examples for clarity
        inp_col = torch.tensor([[[0]], [[1]]])  # [B=2, H=1, W=1]
        inp_msk = torch.ones(2, 1, 1).bool()

        col_logits = torch.zeros(2, 1, 1, 10)
        col_logits[0, 0, 0, 0] = 1.0  # Correct
        col_logits[1, 0, 0, 0] = 1.0  # Incorrect (should be 1)

        msk_logits = torch.ones(2, 1, 1, 1)  # All correct

        inp = Batch(
            one=torch.zeros(2, 1, 1, 10),
            fea=torch.zeros(2, 1, 1, 147),
            col=inp_col,
            msk=inp_msk,
            colf=inp_col.reshape(2, -1),
            mskf=inp_msk.reshape(2, -1).float(),
            idx=torch.tensor([0, 1]),
        )

        out = Batch(
            one=torch.zeros(2, 1, 1, 10),
            fea=torch.zeros(2, 1, 1, 147),
            col=torch.zeros(2, 1, 1).long(),
            msk=torch.zeros(2, 1, 1).bool(),
            colf=torch.zeros(2, 1).long(),
            mskf=torch.zeros(2, 1),
            idx=torch.tensor([0, 1]),
        )

        batch_data = BatchData(inp=inp, out=out)

        # First call
        metrics: Dict[int, Dict[str, float]] = {}
        result = training_index_metrics(
            batch_data, col_logits, msk_logits, "input", metrics
        )

        assert result[0]["input_n_masked_pixels"] == 1.0
        assert result[0]["input_n_incorrect_num_color"] == 0.0
        assert result[0]["input_n_incorrect_num_mask"] == 0.0
        assert result[1]["input_n_masked_pixels"] == 1.0
        assert result[1]["input_n_incorrect_num_color"] == 1.0
        assert result[1]["input_n_incorrect_num_mask"] == 0.0

        # Second call with same indices should update with new averages
        result = training_index_metrics(
            batch_data, col_logits, msk_logits, "input", result
        )

        assert result[0]["input_n_masked_pixels"] == 1.0  # Average: (1 + 1) / 2
        assert result[0]["input_n_incorrect_num_color"] == 0.0
        assert result[0]["input_n_incorrect_num_mask"] == 0.0
        assert result[1]["input_n_masked_pixels"] == 1.0  # Average: (1 + 1) / 2
        assert result[1]["input_n_incorrect_num_color"] == 1.0  # Average: (1 + 1) / 2
        assert result[1]["input_n_incorrect_num_mask"] == 0.0

    def test_edge_cases(self):
        """Test edge cases like empty masks and single examples."""
        # Test with no active pixels
        inp_col = torch.zeros(1, 2, 2).long()
        inp_msk = torch.zeros(1, 2, 2).bool()  # No active pixels

        col_logits = torch.zeros(1, 2, 2, 10)
        msk_logits = torch.zeros(1, 2, 2, 1)

        inp = Batch(
            one=torch.zeros(1, 2, 2, 10),
            fea=torch.zeros(1, 2, 2, 147),
            col=inp_col,
            msk=inp_msk,
            colf=inp_col.reshape(1, -1),
            mskf=inp_msk.reshape(1, -1).float(),
            idx=torch.tensor([42]),
        )

        out = Batch(
            one=torch.zeros(1, 2, 2, 10),
            fea=torch.zeros(1, 2, 2, 147),
            col=torch.zeros(1, 2, 2).long(),
            msk=torch.zeros(1, 2, 2).bool(),
            colf=torch.zeros(1, 4).long(),
            mskf=torch.zeros(1, 4),
            idx=torch.tensor([42]),
        )

        batch_data = BatchData(inp=inp, out=out)

        metrics: Dict[int, Dict[str, float]] = {}
        result = training_index_metrics(
            batch_data, col_logits, msk_logits, "input", metrics
        )

        assert result[42]["input_n_masked_pixels"] == 0.0
        assert result[42]["input_n_incorrect_num_color"] == 0.0
        assert result[42]["input_n_incorrect_num_mask"] == 0.0

    def test_mask_aware_color_counting(self):
        """Test that color errors are only counted within the mask."""
        # Create a simple example with errors both inside and outside the mask
        inp_col = torch.tensor([[[0, 1], [2, 3]]])  # [B=1, H=2, W=2]
        inp_msk = torch.tensor([[[True, True], [False, False]]])  # Top row active

        # Create predictions with errors both inside and outside mask
        col_logits = torch.zeros(1, 2, 2, 10)
        col_logits[0, 0, 0, 0] = 1.0  # Correct (0) inside mask
        col_logits[0, 0, 1, 9] = 1.0  # Incorrect (should be 1) inside mask
        col_logits[0, 1, 0, 9] = 1.0  # Incorrect (should be 2) but OUTSIDE mask
        col_logits[0, 1, 1, 9] = 1.0  # Incorrect (should be 3) but OUTSIDE mask

        msk_logits = torch.ones(1, 2, 2, 1)  # All mask predictions correct
        msk_logits[0, 1, :, 0] = -1.0  # Correctly predict False for bottom row

        inp = Batch(
            one=torch.zeros(1, 2, 2, 10),
            fea=torch.zeros(1, 2, 2, 147),
            col=inp_col,
            msk=inp_msk,
            colf=inp_col.reshape(1, -1),
            mskf=inp_msk.reshape(1, -1).float(),
            idx=torch.tensor([99]),
        )

        out = Batch(
            one=torch.zeros(1, 2, 2, 10),
            fea=torch.zeros(1, 2, 2, 147),
            col=torch.zeros(1, 2, 2).long(),
            msk=torch.zeros(1, 2, 2).bool(),
            colf=torch.zeros(1, 4).long(),
            mskf=torch.zeros(1, 4),
            idx=torch.tensor([99]),
        )

        batch_data = BatchData(inp=inp, out=out)

        metrics: Dict[int, Dict[str, float]] = {}
        result = training_index_metrics(
            batch_data, col_logits, msk_logits, "input", metrics
        )

        # Should only count the 1 error inside the mask, not the 2 errors outside
        assert result[99]["input_n_masked_pixels"] == 2.0  # Two active pixels
        assert result[99]["input_n_incorrect_num_color"] == 1.0  # Only 1 error inside mask
        assert result[99]["input_n_incorrect_num_mask"] == 0.0  # All mask predictions correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
