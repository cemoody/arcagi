# Main Mapping Experiments

This directory contains experiments for learning to map from input features to output features on the ARC-AGI dataset.

## Overview

The main mapping task involves:
- **Input**: `input_features` (147-dimensional feature vector per grid cell)
- **Output**: `output_features` (147-dimensional feature vector per grid cell) and `output_mask` (binary mask)
- **Focus**: Heavily weight the prediction of `order2` features (first 36 dimensions) which represent pairwise relational features
- **Architecture**: Message passing networks for spatial consistency

## Key Differences from Color Mapping

1. **Feature-to-Feature Mapping**: Instead of mapping features to colors, we map features to features
2. **Mask Prediction**: Also predict which cells are masked in the output
3. **Weighted Loss**: Order2 features (pairwise relations) are weighted 10x more than other features
4. **No Color Information**: We explicitly ignore the color arrays (`inputs` and `outputs`)

## Feature Breakdown

The 147-dimensional feature vector consists of:
- **Dimensions 0-35**: Order2 features (36 dims) - pairwise color matching in 3x3 neighborhood
- **Dimensions 36-43**: Center-to-neighbor mask detection (8 dims)
- **Dimensions 44-127**: Order3 features (84 dims) - triplet relations
- **Dimensions 128-136**: Number of colors features (9 dims)
- **Dimension 137**: Is mask feature (1 dim)
- **Dimensions 138-146**: Number of cells matching center (9 dims)

## Experiments

### ex01.py - Message Passing Feature Mapper
- Based on the successful ex15.py architecture from color_mapping
- Uses weight-tied message passing (single layer reused 12 times)
- Predicts both output features and output mask
- Loss function heavily weights order2 features (10x weight)
- Architecture:
  - Feature extractor: 147 → 512 → 512
  - Position embeddings: learnable 30×30×512
  - Message passing: 12 rounds with shared 3×3 convolution
  - Feature head: 512 → 512 → 147
  - Mask head: 512 → 256 → 1
- Model size: ~1.6M parameters
- Features:
  - Weights & Biases (wandb) integration for detailed logging
  - Logs training losses on every step
  - Separate tracking of order2 loss vs other feature losses
  - Mask prediction accuracy tracking
  - **Filename filtering**: Can train on examples from a single file (default: `28e73c20`)

## Running Experiments

To run the first experiment:

```bash
# Train on all files
python arcagi/main_mapping/ex01.py --filename all

# Train on a specific file (default: 28e73c20)
python arcagi/main_mapping/ex01.py

# Train on a different file
python arcagi/main_mapping/ex01.py --filename 3345333e
```

### With Weights & Biases (wandb) Integration

The experiment now includes wandb integration for tracking losses on every iteration:

```bash
# Run with wandb logging (default)
python arcagi/main_mapping/ex01.py

# Disable wandb logging
python arcagi/main_mapping/ex01.py --no_wandb

# Custom wandb project and run name
python arcagi/main_mapping/ex01.py \
    --wandb_project "my-arc-project" \
    --wandb_name "experiment-1"
```

Wandb will log:
- Training losses on every step
- Validation metrics on every epoch
- Model architecture and hyperparameters
- Model checkpoints

## Metrics

- **val_epoch_order2_acc**: Accuracy on order2 features (primary metric)
- **val_epoch_mask_acc**: Accuracy on mask prediction
- **val_epoch_loss**: Total validation loss

## Data Requirements

The experiments expect preprocessed data files:
- `processed_data/train_all.npz`: Training data with all features
- `processed_data/eval_all.npz`: Evaluation data with all features

These files should contain:
- `inputs_features`: Input feature arrays
- `outputs_features`: Output feature arrays
- `inputs_mask`: Input mask arrays
- `outputs_mask`: Output mask arrays 