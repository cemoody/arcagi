# Color Mapping Experiments

This directory contains experiments for learning color mappings from ARC-AGI data.

## Experiments

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