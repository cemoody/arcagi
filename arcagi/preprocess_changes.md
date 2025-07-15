# Preprocessing Changes Summary

## Overview
Modified `preprocess.py` to reorganize data loading to be clearer about the data structure and avoid confusion with overloaded "train" terminology.

## Key Changes

### 1. Always Load Both Subsets
- Previously: Only loaded either "train" or "test" subset from each JSON file based on parameter
- Now: Always loads BOTH "train" and "test" subsets from each JSON file
- The `subset` parameter has been removed since it's no longer needed

### 2. New Array: `subset_example_index_is_train`
- Added a new boolean array that tracks which subset each example comes from
- `True` = example is from the "train" subset within the JSON file
- `False` = example is from the "test" subset within the JSON file
- Same length as `filenames`, `indices`, and first axis of `inputs_mask`

### 3. Index Numbering
- Indices now continue sequentially across both subsets within a file
- Example: If a file has 5 train examples and 1 test example:
  - Train examples get indices 0, 1, 2, 3, 4
  - Test example gets index 5

## Data Organization Clarification

The data hierarchy is now clearer:
1. **Top level**: `training/` or `evaluation/` directories (train vs eval split)
2. **File level**: Individual JSON files
3. **Within files**: "train" and "test" subsets (now both loaded together)

## NPZ File Structure

The saved NPZ files now contain:
- `inputs`: (n_examples, 30, 30) int32 array
- `outputs`: (n_examples, 30, 30) int32 array
- `filenames`: array of source JSON filenames
- `indices`: array of example indices within each file
- **`subset_example_index_is_train`**: NEW - boolean array indicating train/test subset
- `filename_colors_keys`: unique filenames
- `filename_colors_values`: colors used by each filename
- `inputs_mask`: boolean mask for valid pixels
- `outputs_mask`: boolean mask for valid pixels
- `inputs_features`: concatenated features (if enabled)
- `outputs_features`: concatenated features (if enabled)
- `feature_names`: names of feature types (if features enabled)

## Usage Example

```python
import numpy as np

# Load data
data = np.load('processed_data/train_all.npz')

# Access the new array
is_train = data['subset_example_index_is_train']

# Filter examples by subset
train_mask = is_train  # Examples from "train" subset
test_mask = ~is_train  # Examples from "test" subset

# Count examples
print(f"Train subset examples: {train_mask.sum()}")
print(f"Test subset examples: {test_mask.sum()}")
```

## Benefits

1. **No more confusion**: Clear distinction between directory-level train/eval split and within-file train/test subsets
2. **Complete data**: All examples from both subsets are available in one place
3. **Flexible filtering**: Easy to filter by subset when needed using the boolean array
4. **Backward compatible**: Existing code can ignore the new array if not needed 