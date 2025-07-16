# Color Mapping Experiments

This directory contains experiments for learning color mappings from ARC-AGI data.

## Experiments

### ex16.py - Optimized Memorization with Sinusoidal Position Embeddings
- Same architecture as ex15.py but with parameter-free position encodings
- **Sinusoidal Position Embeddings**: Uses Transformer-style sinusoidal encoding:
  - No learnable parameters for positions (saves 460,800 parameters)
  - Applies sin/cos functions with different frequencies to x and y coordinates
  - First half of dimensions encode x-position, second half encode y-position
- **Binary Noise Regularization**: Adds training-time noise for robustness:
  - 5% probability of flipping binary features (0↔1) during training
  - Small Gaussian noise (σ=0.01) for continuous features
  - Only active during training, disabled during evaluation
  - Controllable via `--noise_prob` argument (default: 0.05)
- **Per-Index Accuracy Tracking**: Detailed performance monitoring:
  - Tracks accuracy separately for each example index within files
  - Shows input vs output accuracy for each index
  - Reports whether both input and output achieve perfect accuracy
  - Helps identify which specific examples are challenging
- **Colored Terminal Visualization**: Uses `terminal_imshow` for better readability:
  - Color-coded grid display instead of plain numbers
  - Shows background (-1) as empty spaces
  - Includes colored legend for easy interpretation
  - Much easier to spot differences between predicted and ground truth
- **Single-File Mode**: Train/test split within same file:
  - Train on 'train' subset, evaluate on 'test' subset
  - Useful for understanding generalization within file patterns
  - Separate test evaluation with detailed metrics
- Benefits:
  - Reduces overfitting by removing learnable position parameters
  - Provides smooth interpolation between positions
  - Should generalize better to unseen spatial patterns
  - More robust training with noise regularization
  - Better visibility into model performance per example
- **Results**: Near-perfect input accuracy (~99.7%) 
  - Sinusoidal embeddings may cause less stable training compared to learned embeddings
  - While parameter-free, they may not adapt as well to spatial patterns in this task

**Training Progress on `28e73c20` (single-file mode, 100 epochs):**
- **Input accuracy**: ~99.7% - Model learns input patterns very well
- **Training behavior**: Unstable, oscillates between predicting all zeros or all threes
- **Parameter reduction**: ~460k fewer parameters compared to ex15.py (learned embeddings)
- **Key insight**: Sinusoidal embeddings, while parameter-free and theoretically better for generalization, may not be optimal for this specific memorization task where learned position embeddings can better adapt to the spatial patterns in the training data

### ex01.py - Basic Color Mapping
- Simple linear model to predict colors from features
- Achieved 37% validation accuracy with 10 classes

### ex02.py - Improved Color Mapping with MLP
- Multi-layer perceptron with hidden layers
- Achieved 46% validation accuracy
- Better feature extraction but still struggles with complex patterns

### ex03.py - Color Mapping with Residual Connections
- Added residual connections and deeper architecture
- Achieved 48% validation accuracy
- Marginal improvement over basic MLP

### ex04.py - Color Mapping with Attention
- Added self-attention mechanism for global context
- Achieved 49% validation accuracy
- Attention helps but not significantly

### ex05.py - Color Mapping with Spatial Features
- Added spatial position encodings and local feature aggregation
- Achieved 51% validation accuracy
- Spatial awareness provides modest improvements

### ex06.py - Color Mapping with Color Constraints
- Added hard constraints to only predict colors present in input
- Achieved 58% validation accuracy
- Significant improvement by constraining output space

### ex07.py - Advanced Color Mapping with Multiple Strategies
- Combined multiple approaches: attention, spatial features, color constraints
- Added example-specific adaptation
- Best validation accuracy: ~52% (without full color constraints)

### ex08.py - Deep MLP for Single File Memorization
- Target: Perfect memorization of filename `3345333e` (2 train examples)
- Deep MLP with 6 hidden layers, 512 units each
- 2.8M parameters
- Achieved 95.3% accuracy after 100 epochs
- Strong on dominant colors (0, 6), struggled with rare colors (1, 2, 3)

### ex09.py - Improved Single File Model
- Enhanced architecture with residual connections and position embeddings
- 8 hidden layers, 768 units, color-specific sub-networks
- Achieved 97.9% accuracy after 200 epochs
- Better handling of rare colors but still not perfect

### ex10.py - Perfect Memorization Model (Failed)
- 342M parameters with example-specific sub-networks
- Spatial attention mechanisms, custom loss weighting
- Encountered technical issues with implementation

### ex11.py - Lookup-Based Memorization
- Simple approach with learnable embeddings per example
- 279M parameters, direct mapping approach
- Achieved 100% accuracy after ~200 epochs
- Successfully memorized both training examples

### ex12.py - Message-Passing Network
- Spatial message passing for consistency
- 14.7M parameters, 6 rounds of message passing
- Achieved 100% accuracy after 62 epochs
- Demonstrated that spatial consistency is key to perfect memorization

### ex13.py - Streamlined Message-Passing Model
- Optimized version of ex12 for faster training
- 1.2M parameters with example-aware features
- Achieved 99.71% accuracy in just 40 epochs
- Much more efficient than previous models

### ex14.py - High-Performance Message-Passing Model
- Further optimized with better initialization and learning schedule
- 1.7M parameters with 8 message passing rounds
- Achieved 99.9% accuracy in just 35 epochs
- Best balance of model size, training time, and accuracy
- **UPDATE**: With 12 message passing rounds and 100 epochs:
  - Achieved **100% accuracy in 44 epochs**
  - 2.0M parameters
  - Perfect memorization with efficient training time
- **Multi-file testing results** (all with 12 message rounds, 100 max epochs):
  - `3345333e` (2 examples): 100% accuracy in 44 epochs
  - `e76a88a6` (3 examples): 100% accuracy in 43 epochs
  - `c0f76784` (2 examples): 100% accuracy in 30 epochs
  - `e21d9049` (2 examples): 99.12% accuracy after 100 epochs → **100% accuracy after 183 epochs with 200 max epochs**
  - `ef135b50` (3 examples): 100% accuracy in 47 epochs
  - `3ac3eb23` (2 examples): 100% accuracy in 51 epochs
  - **Success rate**: 6/6 files (100%) achieved perfect accuracy
  - Average epochs to 100%: 43 epochs (for files trained with 100 epochs)
  - Note: One file required 183 epochs to reach 100% accuracy
- **Final configuration** (500 max epochs, 12 message rounds, 0.1 learning rate):
  - Default patience: 5 epochs (requires 5 consecutive epochs of 100% accuracy before stopping)
  - Custom early stopping callback: `PerfectAccuracyEarlyStopping`
    - Monitors both INPUT and OUTPUT accuracy separately
    - Only stops when **BOTH** achieve exactly 100% accuracy for 5 consecutive epochs
    - Resets counter if either drops below 100%
    - Provides clear progress messages during training
  - Tracks separate metrics: `val_input_acc`, `val_output_acc`, and `val_both_perfect`
  - Ensures stable perfect memorization on both prediction tasks before terminating training

## Key Insights

1. **Color constraints are crucial**: Limiting predictions to colors present in the input significantly improves accuracy
2. **Spatial consistency matters**: Message passing between neighboring pixels helps maintain coherent predictions
3. **Example-specific features help**: Adding learnable embeddings per training example aids memorization
4. **More message passing rounds improve accuracy**: 12 rounds (ex14) performed better than 2-6 rounds
5. **Aggressive learning rates work well**: Higher learning rates (0.05-0.1) with proper scheduling achieve faster convergence
6. **Some files need more epochs**: While most files converge in ~40-50 epochs, some require 150+ epochs for perfect accuracy

## Best Models

For perfect memorization on single file:
- **Fastest training**: ex14.py (99.9% accuracy in 35 epochs, 1.7M params)
- **Smallest accurate model**: ex13.py (99.71% accuracy in 40 epochs, 1.2M params)
- **Perfect accuracy**: ex11.py and ex12.py (100% accuracy but slower training)
- **Best overall**: ex14.py with 12 message rounds (100% accuracy in 44 epochs, 2.0M params) - combines perfect accuracy with fast training
  - **Multi-file robustness**: ex14.py achieved 100% accuracy on all 6 tested files (100% success rate) with sufficient epochs

### ex15.py - Weight-Tied Message Passing Model
- Experiment with weight tying: single message passing layer reused multiple times
- Based on ex14.py architecture but with shared weights across all message passing rounds
- **Model size**: 1.1M parameters (vs 2.0M in ex14.py - 45% reduction!)
- **Performance results**:
  - `3345333e`: 100% accuracy in 74 epochs (vs 44 epochs in ex14.py)
  - `e76a88a6`: 100% accuracy in 54 epochs (vs 43 epochs in ex14.py)
- Key innovation: Instead of having separate layers for each message round, uses the same layer repeatedly
- This forces the model to learn a more general message passing function that works across all rounds
- **Trade-off**: Slightly slower convergence but 45% smaller model with better parameter efficiency
- The weight tying constraint makes the model learn a universal message passing operation
- **Conclusion**: Weight tying is effective - achieves perfect accuracy with significantly fewer parameters

### ex17.py - Dual Prediction: Colors and Masks
- **Dual prediction heads**: Predicts both colors and masks (is_mask field) simultaneously
- **Separate losses and accuracies**: 
  - Color loss: Cross-entropy for 10-class color classification
  - Mask loss: Binary cross-entropy for mask prediction
  - Tracks separate accuracies for color and mask predictions
- **Enhanced metrics tracking**:
  - Input/Output color accuracy
  - Input/Output mask accuracy  
  - Overall color and mask accuracy
  - Per-index breakdown for both color and mask performance
- **Architecture**: Based on ex16.py with additional mask prediction head
  - Same sinusoidal position embeddings and message passing architecture
  - Added `mask_head` Linear layer for binary mask prediction
  - Combined loss: `total_loss = color_loss + mask_loss`
- **Data loading**: Utilizes `inputs_mask` and `outputs_mask` from processed data
- **Model outputs**: Returns tuple `(color_logits, mask_logits)` from forward pass
- **Results on `28e73c20` (6 examples, 200 epochs)**:
  - **Perfect dual prediction**: 100% accuracy for both colors AND masks
  - **Training completion**: Achieved perfect accuracy and maintained for 77 consecutive epochs (stopped after 200 total epochs)
  - **Per-component accuracy**:
    - Input color accuracy: 100.00% (918 pixels)
    - Output color accuracy: 100.00% (918 pixels) 
    - Input mask accuracy: 100.00% (5,400 pixels)
    - Output mask accuracy: 100.00% (5,400 pixels)
  - **Per-index performance**: All 6 examples achieved 100% accuracy on both colors and masks
  - **Best checkpoint**: Epoch 43 with color_acc=1.0000, mask_acc=0.9965
  - **Final validation loss**: 2.9e-6 (extremely low, indicating perfect memorization)
- **Key achievement**: First model to successfully predict both color classification and binary mask with perfect accuracy
- **Robustness**: Maintained perfect accuracy across all validation examples and both prediction tasks 