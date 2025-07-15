# Changes Log

## Per-Index Loss Reporting (Latest)

Added per-index metrics reporting to track model performance on individual examples.

### Changes:
1. **Updated data loading**:
   - Now loads and tracks `indices` from NPZ files
   - Passes indices through the data pipeline

2. **Added per-example metrics in validation**:
   - Tracks loss and accuracy for each unique index
   - Counts non-mask pixels per example
   - Counts correctly predicted pixels for order2 features

3. **Enhanced epoch-end reporting**:
   - Prints a table with metrics for each index in ascending order
   - Shows: Index, Non-mask Pixels, Correct Pixels, Accuracy
   - Logs per-index metrics to wandb

### Output Example:
```
Epoch 0 - Per-Index Metrics:
Index      Non-mask Pixels      Correct Pixels       Accuracy  
----------------------------------------------------------------------
0          97320                78759.7              80.93%    
1          1195                 861.5                72.09%    
```

This helps identify:
- Which examples are harder/easier for the model
- Whether the model struggles with larger or smaller examples
- Progress on individual examples across epochs

## Filename Filtering Feature (Latest)

Added the ability to filter training and validation data to examples from a specific file.

### Changes:
1. **Added `--filename` argument**:
   - Default: `28e73c20`
   - Use `--filename all` to train on all files
   - Specify any filename without .json extension

2. **Updated `load_feature_data` function**:
   - Added `filename_filter` parameter
   - Filters examples based on filename in the NPZ file
   - Raises error if no examples found for specified filename

3. **Updated wandb config**:
   - Now includes the filename in the logged configuration

### Usage Examples:
```bash
# Train on default file (28e73c20)
python arcagi/main_mapping/ex01.py

# Train on specific file
python arcagi/main_mapping/ex01.py --filename 3345333e

# Train on all files
python arcagi/main_mapping/ex01.py --filename all
```

## Wandb Integration Changes

## Summary

Added Weights & Biases (wandb) integration to `ex01.py` for detailed experiment tracking.

## Changes Made

1. **Import wandb**:
   - Added `import wandb` and `from pytorch_lightning.loggers import WandbLogger`

2. **Enhanced Logging**:
   - Training losses are now logged on every step (`on_step=True`)
   - Added `prog_bar=True` for key metrics to show in progress bar
   - Validation metrics logged per epoch with progress bar display

3. **Command Line Arguments**:
   - `--wandb_project`: Set wandb project name (default: "arc-main-mapping")
   - `--wandb_name`: Set custom run name (default: auto-generated)
   - `--no_wandb`: Disable wandb logging entirely

4. **Wandb Logger Configuration**:
   - Logs model hyperparameters and architecture details
   - Enables model checkpoint logging (`log_model=True`)
   - Tracks all training and validation metrics

## Latest Updates for Per-Iteration Logging

### Additional Changes:
1. **Added `sync_dist=False`** to all training logs for faster logging
2. **Added `log_every_n_steps=1`** to trainer configuration
3. **Added per-step tracking**:
   - `batch_idx`: Current batch index
   - `global_step`: Global step counter
   - `learning_rate`: Current learning rate
4. **Configured wandb for immediate logging**:
   - Set `WANDB_MODE=online`
   - Added `_log_every_n_steps: 1` to wandb settings
   - Specified `save_dir` for wandb logs

## Metrics Tracked

### Training (per step):
- `train_loss`: Total training loss
- `train_feature_loss`: Total feature prediction loss
- `train_order2_loss`: Loss on order2 features (heavily weighted)
- `train_other_loss`: Loss on other features
- `train_mask_loss`: Binary mask prediction loss
- `batch_idx`: Current batch index within epoch
- `global_step`: Global step counter across all epochs
- `learning_rate`: Current learning rate value

### Validation (per epoch):
- `val_loss`: Total validation loss
- `val_feature_loss`: Total feature prediction loss
- `val_order2_loss`: Loss on order2 features
- `val_other_loss`: Loss on other features
- `val_mask_loss`: Mask prediction loss
- `val_mask_acc`: Mask prediction accuracy
- `val_order2_acc`: Accuracy on order2 binary features

### Epoch-level aggregates:
- `val_epoch_loss`: Average validation loss across epoch
- `val_epoch_order2_acc`: Average order2 accuracy across epoch
- `val_epoch_mask_acc`: Average mask accuracy across epoch

## Usage

```bash
# Run with wandb (default)
python arcagi/main_mapping/ex01.py

# Run without wandb
python arcagi/main_mapping/ex01.py --no_wandb

# Custom project and run name
python arcagi/main_mapping/ex01.py \
    --wandb_project "my-project" \
    --wandb_name "experiment-1"
``` 

## v1.2.1 - Fixed Per-Index Metrics Calculation

### Fixed
- **Correct pixel counting**: Fixed the per-index metrics to properly count pixels instead of feature predictions
  - Previously was counting total correct feature predictions across all 36 order2 features  
  - Now counts pixels where ALL 36 order2 features match correctly
- **Average metrics per index**: Fixed metrics aggregation to show averages when indices appear multiple times
  - Previously summed up metrics across all appearances of an index in validation set
  - Now shows average metrics per example and the count of how many times each index appears
  - This explains why we saw 97,320 "non-mask pixels" for index 0 - it appeared ~108 times with ~900 pixels each

### Display Changes
- Added "Count" column showing how many times each index appears in validation set
- Changed "Non-mask Pixels" to "Avg Non-mask Pixels" 
- Changed "Correct Pixels" to "Avg Correct Pixels"
- Updated header to clarify metrics are averaged over validation examples 

## v1.2.2 - Enhanced Per-Index Wandb Logging

### Added
- **Total correct pixels logging**: Now logs `val_idx_{idx}_total_correct_pixels` to wandb
  - Shows the total number of correctly predicted pixels across all validation examples for each index
  - Complements the existing average metrics for better tracking of overall performance
- Keeps all existing metrics:
  - `val_idx_{idx}_count`: Number of times index appears in validation set
  - `val_idx_{idx}_avg_valid_pixels`: Average non-mask pixels per example
  - `val_idx_{idx}_avg_correct_pixels`: Average correct pixels per example  
  - `val_idx_{idx}_accuracy`: Average accuracy (correct/valid pixels)

## v1.3.0 - Data Augmentation for Small Datasets

### Added
- **Data augmentation factor**: New `--data_augment_factor` argument to repeat dataset multiple times per epoch
  - Helps reduce epoch overhead when training on small datasets (e.g., single filename)
  - Default value is 1 (no augmentation)
  - Example: `--data_augment_factor 100` repeats the training data 100x per epoch
  - Only applies to training data; validation data is never augmented
  - Significantly improves wallclock training time for small datasets by reducing epoch transitions

### Implementation
- Modified `load_feature_data` function to accept `data_augment_factor` parameter
- Uses `torch.Tensor.repeat()` to efficiently duplicate all tensors in memory
- Preserves original indices for tracking purposes

## v1.4.0 - Per-Example Metrics for Single-File Training

### Added
- **Per-example tracking**: When training on a single file (≤10 unique indices), shows individual performance
  - Detects single-file mode automatically based on number of unique indices seen
  - Shows metrics for each example (0, 1, 2, 3, 4) separately
  - Tracks how many times each example was seen (useful with data augmentation)
  - Reports average loss and order2 accuracy per example

### Features
- **Training metrics**: Shows per-example performance at end of each training epoch
- **Validation metrics**: Shows per-example performance when validation set has few unique indices
- Helps identify which specific examples the model struggles with
- Particularly useful when combined with data augmentation to see averaged performance

### Example Output
```
Training Epoch 0 - Per-Example Performance:
(Note: These are averaged across all augmented copies)
Example    Seen       Avg Loss        Avg Order2 Acc     
------------------------------------------------------------
0          100        2.456           45.23%             
1          100        1.892           62.45%             
2          100        3.124           38.91%             
3          100        2.001           55.67%             
4          100        1.756           68.34%             
``` 