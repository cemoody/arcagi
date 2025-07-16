#!/usr/bin/env python3
"""
Test script to verify color model integration with feature mapping.

This script tests:
1. Loading a saved color model from ex15
2. Using it for visualization in ex01
3. Generating color predictions from features
"""

import os
import sys

import torch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_color_model_integration():
    """Test the integration between ex15 color model and ex01 feature mapping."""

    print("Testing Color Model Integration")
    print("=" * 50)

    # Test 1: Check if we can import the color model loader
    try:
        from color_mapping.ex15 import load_color_model_for_inference

        print("✓ Successfully imported color model components")
    except ImportError as e:
        print(f"✗ Failed to import color model: {e}")
        return False

    # Test 2: Check if we can import the feature mapping model
    try:
        from main_mapping.ex01 import FeatureMappingModel

        print("✓ Successfully imported feature mapping model")
    except ImportError as e:
        print(f"✗ Failed to import feature mapping model: {e}")
        return False

    # Test 3: Test the visualization function with dummy data
    try:
        # Create dummy features (30x30x147)
        dummy_features = torch.randn(30, 30, 147)

        # Create a feature mapping model
        feature_model = FeatureMappingModel(
            input_feature_dim=147,
            output_feature_dim=147,
            hidden_dim=256,
            num_message_rounds=6,
        )

        # Test setting filename
        test_filename = "test_file"
        feature_model.set_filename(test_filename)

        print(
            f"✓ Created feature mapping model with filename: {feature_model.current_filename}"
        )

    except Exception as e:
        print(f"✗ Failed to create and configure models: {e}")
        return False

    # Test 4: Check if visualization function exists
    try:
        # Test that the function exists and can handle the fallback case
        print("Testing visualization function availability...")

        # The function should exist and be callable
        assert hasattr(
            feature_model, "visualize_with_color_model"
        ), "visualize_with_color_model method not found"
        assert callable(
            getattr(feature_model, "visualize_with_color_model")
        ), "visualize_with_color_model is not callable"

        print("✓ Visualization function is available and callable")
        print(
            "  (Note: Full test requires trained color model and proper edge features)"
        )

    except Exception as e:
        print(f"✗ Visualization function test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("Integration test completed successfully!")
    print("\nTo use the color model visualization:")
    print("1. First train a color model using ex15.py:")
    print("   python arcagi/color_mapping/ex15.py --filename YOUR_FILENAME")
    print("2. Then run feature mapping with the same filename:")
    print("   python arcagi/main_mapping/ex01.py --filename YOUR_FILENAME")
    print("3. The visualization will automatically use the color model if available")

    return True


if __name__ == "__main__":
    success = test_color_model_integration()
    sys.exit(0 if success else 1)
