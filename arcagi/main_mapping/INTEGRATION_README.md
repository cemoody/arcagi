# Color Model Integration with Feature Mapping

This document explains the integration between the ex15.py color memorization model and the ex01.py feature mapping model for improved visualization.

## Overview

The integration allows ex01.py to use trained color models from ex15.py to visualize feature predictions as actual colors instead of using the terminal-based `relshow` function. This provides much clearer and more intuitive visualizations.

## Changes Made

### 1. ex15.py Modifications

- **Added `predict_colors()` method**: Converts features to color predictions
- **Added model saving**: Automatically saves trained models with filename-specific names
- **Added `load_color_model_for_inference()` function**: Utility to load saved models for inference

#### Key additions:
```python
def predict_colors(self, features: torch.Tensor) -> torch.Tensor:
    """Predict colors from features. Returns color indices."""
    
def load_color_model_for_inference(filename: str, checkpoint_dir: str = "color_mapping_outputs") -> OptimizedMemorizationModel:
    """Load a trained color model for inference."""
```

### 2. ex01.py Modifications

- **Added color model import**: Imports the ex15 color model loader
- **Added `set_filename()` method**: Allows setting the current filename for visualization
- **Added `visualize_with_color_model()` method**: Uses color model for visualization instead of relshow
- **Modified validation visualization**: Automatically uses color model when available, falls back to relshow

#### Key additions:
```python
def set_filename(self, filename: str) -> None:
    """Set the current filename for visualization."""
    
def visualize_with_color_model(self, features: torch.Tensor, filename: str, title: str = "Predicted Colors") -> None:
    """Visualize features using the trained color model."""
```

## Usage Instructions

### Step 1: Train a Color Model

First, train a color model for your specific filename using ex15.py:

```bash
python arcagi/color_mapping/ex15.py --filename YOUR_FILENAME --max_epochs 100
```

This will create a saved model file: `color_mapping_outputs/color_model_YOUR_FILENAME.pt`

### Step 2: Run Feature Mapping with Visualization

Run the feature mapping model with the same filename:

```bash
python arcagi/main_mapping/ex01.py --filename YOUR_FILENAME --single_file_mode
```

The visualization will automatically:
1. Detect that a color model exists for the filename
2. Load the color model for inference
3. Use it to convert features to colors
4. Save PNG images with color visualizations instead of using terminal output

### Step 3: View Generated Visualizations

The system will generate PNG files with names like:
- `visualization_YOUR_FILENAME_expected_edges_ex0_region5-25_5-25.png`
- `visualization_YOUR_FILENAME_predicted_edges_ex0_region5-25_5-25.png`

## File Structure

```
arcagi/
├── color_mapping/
│   ├── ex15.py                     # Color memorization model (modified)
│   └── color_model_*.pt           # Saved color models (generated)
├── main_mapping/
│   ├── ex01.py                     # Feature mapping model (modified)
│   ├── test_color_integration.py   # Integration test script (new)
│   └── INTEGRATION_README.md       # This file (new)
└── visualization_*.png             # Generated visualizations (new)
```

## Testing the Integration

Run the test script to verify everything works:

```bash
python arcagi/main_mapping/test_color_integration.py
```

This will test:
- Import functionality
- Model creation and configuration
- Visualization function availability

## Fallback Behavior

If no color model is found for the specified filename, the system automatically falls back to the original `relshow` terminal visualization. This ensures backward compatibility.

## Benefits

1. **Better Visualization**: Color images are much clearer than terminal ASCII art
2. **Saved Outputs**: PNG files can be shared and analyzed later
3. **Filename-Specific**: Each model learns the color patterns for specific ARC tasks
4. **Automatic Integration**: No manual steps needed once models are trained
5. **Fallback Safety**: Always works even without trained color models

## Troubleshooting

### "No trained model found" error
Make sure you've trained a color model for the filename first using ex15.py.

### Import errors
Ensure both ex15.py and ex01.py are in the correct directory structure.

### Visualization not working
Check that matplotlib is installed: `pip install matplotlib`

### Color model not loading
Verify the checkpoint directory path in the load function matches where ex15.py saved the model.

## Example Workflow

```bash
# 1. Train color model for specific file
python arcagi/color_mapping/ex15.py --filename 28e73c20 --max_epochs 200

# 2. Run feature mapping with visualization
python arcagi/main_mapping/ex01.py --filename 28e73c20 --single_file_mode

# 3. Check generated PNG files in the current directory
ls visualization_28e73c20_*.png
```

This integration provides a powerful way to visualize how the feature mapping model's predictions translate to actual colors in the ARC domain. 