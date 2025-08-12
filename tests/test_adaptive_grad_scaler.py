#!/usr/bin/env python3
"""Tests for the AdaptiveGradScalerSkip implementation."""

import torch
import torch.nn as nn
import pytest
from arcagi.adaptive_grad_scaler import AdaptiveGradScalerSkip


def test_adaptive_grad_scaler_normal_gradients():
    """Test the adaptive grad scaler with normal gradients."""
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create the adaptive grad scaler
    scaler = AdaptiveGradScalerSkip(
        beta=0.98,
        warmup_steps=5,
        mult=4.0,
        z_thresh=2.0,
        eps=1e-12
    )

    # Test with normal gradients
    scales = []
    spikes = []
    
    for i in range(10):
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        y = model(x).sum()
        y.backward()
        
        scale, spiked = scaler.maybe_rescale_(model.parameters())
        scales.append(scale)
        spikes.append(spiked)
        
        optimizer.step()
    
    # During warmup (first 5 steps), should never spike
    assert all(not spike for spike in spikes[:5]), "Should not spike during warmup"
    assert all(scale == 1.0 for scale in scales[:5]), "Scale should be 1.0 during warmup"
    
    # After warmup, with normal gradients, should not spike
    assert sum(spikes[5:]) == 0, "Should not spike with normal gradients after warmup"


def test_adaptive_grad_scaler_with_spike():
    """Test the adaptive grad scaler with a gradient spike."""
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create the adaptive grad scaler with short warmup
    scaler = AdaptiveGradScalerSkip(
        beta=0.98,
        warmup_steps=3,
        mult=4.0,
        z_thresh=2.0,
        eps=1e-12
    )

    # Build up baseline statistics with normal gradients
    for i in range(5):
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        y = model(x).sum()
        y.backward()
        
        scale, spiked = scaler.maybe_rescale_(model.parameters())
        optimizer.step()
    
    # Now create a spike
    optimizer.zero_grad()
    x = torch.randn(32, 10) * 50  # Large input to cause large gradients
    y = model(x).sum()
    y.backward()
    
    scale, spiked = scaler.maybe_rescale_(model.parameters())
    
    # Should detect spike and scale down
    assert spiked, "Should detect gradient spike"
    assert scale < 1.0, f"Scale should be less than 1.0 when spike detected, got {scale}"
    assert scale > 0.0, f"Scale should be positive, got {scale}"


def test_adaptive_grad_scaler_ema_update():
    """Test that EMA statistics are properly updated."""
    scaler = AdaptiveGradScalerSkip(
        beta=0.9,
        warmup_steps=1,
        mult=4.0,
        z_thresh=2.0,
        eps=1e-12
    )
    
    # Create a simple model
    model = nn.Linear(5, 1)
    
    # First call should initialize mu_log
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    x = torch.randn(10, 5)
    y = model(x).sum()
    y.backward()
    
    scaler.maybe_rescale_(model.parameters())
    
    assert scaler.mu_log is not None, "mu_log should be initialized after first call"
    assert scaler.var_log == 0.0, "var_log should be 0 after first call"
    
    # Second call should update statistics
    optimizer.zero_grad()
    x = torch.randn(10, 5)
    y = model(x).sum()
    y.backward()
    
    old_mu = scaler.mu_log
    scaler.maybe_rescale_(model.parameters())
    
    # After warmup, statistics should be updated (unless spike detected)
    assert scaler.mu_log != old_mu, "mu_log should be updated after warmup"
    assert scaler.var_log > 0.0, "var_log should be positive after multiple updates"


def test_adaptive_grad_scaler_skip_on_spike():
    """Test that EMA is not updated when a spike is detected."""
    scaler = AdaptiveGradScalerSkip(
        beta=0.9,
        warmup_steps=2,
        mult=2.0,  # Lower threshold for easier spike detection
        z_thresh=1.0,  # Lower threshold for easier spike detection
        eps=1e-12
    )
    
    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Build baseline with small gradients
    for i in range(5):
        optimizer.zero_grad()
        x = torch.randn(10, 5) * 0.1  # Small inputs
        y = model(x).sum()
        y.backward()
        scaler.maybe_rescale_(model.parameters())
        optimizer.step()
    
    # Get current statistics
    old_mu = scaler.mu_log
    old_var = scaler.var_log
    
    # Create a large spike
    optimizer.zero_grad()
    x = torch.randn(10, 5) * 100  # Very large inputs
    y = model(x).sum()
    y.backward()
    
    scale, spiked = scaler.maybe_rescale_(model.parameters())
    
    # Verify spike was detected and EMA was not updated
    assert spiked, "Should detect gradient spike"
    assert scaler.mu_log == old_mu, "mu_log should not be updated on spike"
    assert scaler.var_log == old_var, "var_log should not be updated on spike"


def test_adaptive_grad_scaler_initialization():
    """Test the initialization parameters."""
    scaler = AdaptiveGradScalerSkip(
        beta=0.95,
        warmup_steps=100,
        mult=3.0,
        z_thresh=3.0,
        eps=1e-10
    )
    
    assert scaler.beta == 0.95
    assert scaler.warmup_steps == 100
    assert scaler.mult == 3.0
    assert scaler.z_thresh == 3.0
    assert scaler.eps == 1e-10
    assert scaler.mu_log is None
    assert scaler.var_log is None
    assert scaler.step == 0


if __name__ == "__main__":
    test_adaptive_grad_scaler_normal_gradients()
    print("✓ test_adaptive_grad_scaler_normal_gradients passed")
    
    test_adaptive_grad_scaler_with_spike()
    print("✓ test_adaptive_grad_scaler_with_spike passed")
    
    test_adaptive_grad_scaler_ema_update()
    print("✓ test_adaptive_grad_scaler_ema_update passed")
    
    test_adaptive_grad_scaler_skip_on_spike()
    print("✓ test_adaptive_grad_scaler_skip_on_spike passed")
    
    test_adaptive_grad_scaler_initialization()
    print("✓ test_adaptive_grad_scaler_initialization passed")
    
    print("\nAll tests passed!")