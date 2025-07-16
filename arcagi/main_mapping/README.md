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

### ex05.py - Message Passing Feature Mapper with Training Noise & BCE Loss (Latest)
- Enhanced version of ex04.py with binary noise injection and proper binary loss functions
- **Binary Cross Entropy Loss**: Uses BCE loss instead of MSE for all binary features
  - More appropriate for binary classification tasks (0/1 features)
  - Better gradients and training dynamics for binary data
  - Applied to both order2 features (heavily weighted) and other features
  - Uses `binary_cross_entropy_with_logits` for numerical stability
- **Training Noise**: Adds controlled binary noise to input features during training only
  - Default `noise_prob=0.05` (5% chance to flip each binary feature value)
  - Only applied during training, not validation/inference
  - Configurable via `--noise_prob` command line argument
  - For binary features: flips 0→1 and 1→0 with specified probability
- **Robustness Benefits**:
  - Helps prevent overfitting to exact feature patterns
  - Improves generalization to slightly corrupted inputs
  - Forces the model to learn more robust feature representations
  - Particularly useful for small datasets where overfitting is common
- **Architecture**: Same message passing with sinusoidal position embeddings and 48 message rounds
- **Integration**: Includes ex17 color model integration for enhanced visualization
- **Accuracy Calculation**: Apply sigmoid to model logits, then threshold at 0.5 for binary predictions

### ex05.py Results & Findings
- **Severe Overfitting**: Achieves 100% training accuracy with near-zero loss (1.12e-6) by epoch 14
- **Poor Generalization**: Test accuracy only 86.11% despite perfect training performance
- **Key Issues Identified**:
  1. **48 Message Rounds**: Doubled from ex04's 24 rounds, providing too much capacity
  2. **BCE Loss Convergence**: BCE converges much faster than MSE for binary features
  3. **Small Dataset**: Training on just 5 examples from one file (28e73c20) makes memorization easy
- **Overfitting Mechanism**: 
  - With 48 rounds, information can traverse the entire 30x30 grid multiple times
  - Combined with BCE's efficient gradients, the model quickly memorizes training examples
  - The deeper computation graph (48 layers of message passing) enables learning complex, example-specific patterns
- **Lessons Learned**:
  - More message rounds can hurt generalization even without adding parameters
  - BCE loss requires more regularization than MSE for binary features
  - Need to balance global information propagation with generalization ability

### Potential Solutions for ex05 Overfitting
To maintain global information propagation while reducing overfitting:
1. **Stochastic Message Passing**: Randomly vary number of rounds during training (e.g., 12-48)
2. **Message Dropout**: Skip random message passing rounds during training
3. **Progressive Training**: Start with fewer rounds, gradually increase during training
4. **DropPath/Stochastic Depth**: Skip rounds with increasing probability
5. **Reduce Message Rounds**: Simply use fewer rounds (e.g., back to 24 or even 12-16)
6. **Add Regularization**: Increase dropout, weight decay, or noise probability
7. **Early Stopping**: Stop training based on validation loss instead of continuing to overfit

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

### ex03.py - Sinusoidal Position Embeddings
- Same architecture as ex01.py but with parameter-free position encodings
- **Sinusoidal Position Embeddings**: Uses the Transformer-style sinusoidal encoding:
  - No learnable parameters for positions (completely parameter-free)
  - Applies sin/cos functions with different frequencies to x and y coordinates
  - First half of dimensions encode x-position, second half encode y-position
  - Uses the formula: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d))
- Benefits:
  - Zero position-related parameters (down from 460,800 in ex01.py)
  - Smooth interpolation between positions
  - Can theoretically generalize to unseen positions
  - Proven effective in Transformers and other architectures

## Model Conclusions

### ex01.py Results
- **Overfitting on Training Set**: The model effectively overfits on the 5 training examples, achieving near-perfect accuracy
- **Test Set Performance**: On the test example, the model almost gets it perfect initially
- **Performance Degradation**: However, after a few epochs, test performance actually degrades
- **Suspected Cause**: The position embeddings (one for every cell in the 30×30 grid) are likely leading to overfitting
- **Next Steps**: The next experiment will remove or replace the position embeddings with a more generalizable approach


## ex03.py Results
- Works pretty well, even without pos embeddings
- However, I notice that the *expected* color image is itself pretty sloppy. Is the color mapping really perfect for the eval set?
- Also for 28e73c20 i am noticing that the outer color should have been 3, but in our validation data we see 0.

### ex04.py - Enhanced Message Passing with ex17 Color Models
- Updated version of ex03.py using ex17 dual prediction color models for visualization
- **Architecture**: Same message passing with sinusoidal position embeddings but with 24 message rounds (doubled from 12)
- **Key Features**:
  - Integration with ex17 color models for enhanced visualization
  - Dual prediction capability (colors + masks) from loaded models
  - Accuracy metrics calculation for both colors and masks after visualization
  - Updated checkpoint directory support for ex17 models
- **Visualization Enhancements**:
  - Uses ex17 trained models for color prediction from features
  - Shows both expected and predicted outputs with color visualization
  - Displays accuracy metrics after each validation example
  - Backward compatible with ex15 model interface
- **Results on `28e73c20` (Single-file mode, 29 epochs before interruption)**:
  - **Training Performance**: 100% order2 accuracy on all 5 training examples
  - **Validation Performance**: 
    - Test Example 5: 97-99% color accuracy within predicted regions
    - Mask accuracy: 100% (perfect mask prediction)
    - Order2 accuracy: 99.6-99.7% 
  - **Progressive Improvement**: Color accuracy improved from 97.22% to 98.15% over epochs
  - **Training Stability**: Consistent 100% training accuracy with decreasing loss (0.03-0.09)
  - **Visualization Quality**: Clear color-coded terminal output showing expected vs predicted patterns
- **Model Integration Success**: Successfully loads and uses ex17 dual-prediction models for inference
- **Enhanced Metrics**: Now shows separate accuracy for colors within predicted mask regions vs target mask regions 



## Running Experiments

To run the latest experiment (ex05.py):

```bash
# Train with default 5% noise
python arcagi/main_mapping/ex05.py --filename 28e73c20

# Train with custom noise level
python arcagi/main_mapping/ex05.py --filename 28e73c20 --noise_prob 0.1  # 10% noise

# Train without noise
python arcagi/main_mapping/ex05.py --filename 28e73c20 --noise_prob 0.0

# Train on all files with noise
python arcagi/main_mapping/ex05.py --filename all --noise_prob 0.05
```

To run earlier experiments:

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

## Binary Cross Entropy Loss (ex05.py)

Since all features in the ARC-AGI dataset are binary (0 or 1), ex05.py uses Binary Cross Entropy (BCE) loss instead of Mean Squared Error (MSE):

1. **Why BCE over MSE**:
   - MSE treats the problem as regression, but our features are discrete binary values
   - BCE is specifically designed for binary classification/prediction tasks
   - Provides better gradients: steeper gradients when predictions are wrong, gentler when close
   - More stable training for binary targets

2. **Implementation Details**:
   - Uses `F.binary_cross_entropy_with_logits()` for numerical stability
   - **Model Architecture**: Feature head ends with raw `Linear` layer (no activation)
   - **Perfect Logits**: Model outputs are unbounded (-∞ to +∞), ideal for BCE loss
   - Applied to both order2 features (36 dims) and other features (111 dims)
   - **Accuracy Calculation**: Apply sigmoid to logits, then threshold at 0.5
   - Order2 features still weighted 10x higher in the total loss

3. **Accuracy Calculation**:
   - Applies `torch.sigmoid()` to logits to get probabilities
   - Thresholds at 0.5 to get binary predictions
   - Compares binary predictions with binary targets

## Training Noise Mechanism (ex05.py)

The training noise feature adds robustness by randomly flipping binary feature values during training:

1. **When Applied**: Only during training phase (`model.training=True`), never during validation/inference
2. **How It Works**: 
   - For each input feature value, generate a random number between 0 and 1
   - If the random number < `noise_prob`, flip the feature value (0→1, 1→0)
   - Otherwise, keep the original value unchanged
3. **Benefits**:
   - **Prevents Overfitting**: Forces the model to be robust to small perturbations
   - **Improves Generalization**: Model learns to handle slightly corrupted inputs
   - **Better Feature Learning**: Model must rely on combinations of features rather than exact patterns
4. **Implementation Details**:
   - Uses `torch.rand_like()` for efficient vectorized noise generation
   - Creates a copy of input features to avoid modifying original data
   - Configurable probability parameter with sensible default (5%)

## Data Requirements

The experiments expect preprocessed data files:
- `processed_data/train_all.npz`: Training data with all features
- `processed_data/eval_all.npz`: Evaluation data with all features

These files should contain:
- `inputs_features`: Input feature arrays
- `outputs_features`: Output feature arrays
- `inputs_mask`: Input mask arrays
- `outputs_mask`: Output mask arrays 