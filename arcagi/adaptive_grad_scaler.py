"""Adaptive Gradient Scaler for handling gradient spikes during training."""

import math
from typing import Iterator, Optional, Tuple

import torch
from torch import nn


class AdaptiveGradScalerSkip:
    """
    Adaptive gradient scaler that tracks gradient norm statistics and rescales spiky gradients.
    
    - Tracks EMA of log(total_grad_norm) + variance.
    - Declares a spike if z-score > z_thresh AND log_norm > mu_log + log(mult).
    - On spikes: rescales grads in-place and **does NOT** update EMAs (skip).
    - Warmup: updates EMAs but never rescales.
    """

    def __init__(
        self,
        beta: float = 0.98,
        warmup_steps: int = 500,
        mult: float = 4.0,
        z_thresh: float = 4.0,
        eps: float = 1e-12
    ) -> None:
        """
        Initialize the adaptive gradient scaler.
        
        Args:
            beta: EMA decay rate for statistics (default: 0.98)
            warmup_steps: Number of steps before allowing rescaling (default: 500)
            mult: Multiplier for spike threshold (default: 4.0)
            z_thresh: Z-score threshold for spike detection (default: 4.0)
            eps: Small value for numerical stability (default: 1e-12)
        """
        self.beta: float = beta
        self.mu_log: Optional[float] = None
        self.var_log: Optional[float] = None
        self.step: int = 0
        self.warmup_steps: int = warmup_steps
        self.mult: float = mult
        self.z_thresh: float = z_thresh
        self.eps: float = eps

    @torch.no_grad()
    def _total_grad_norm(self, params: Iterator[nn.Parameter], p: float = 2.0) -> float:
        """Calculate the total gradient norm across all parameters."""
        total: float = 0.0
        for q in params:
            if q.grad is not None:
                g = q.grad.detach()
                total += g.norm(p).item() ** p
        return total ** (1.0 / p) if total > 0 else 0.0

    def _ema_update(self, x_log: float) -> None:
        """Update the exponential moving averages of log norm and variance."""
        if self.mu_log is None:
            self.mu_log = float(x_log)
            self.var_log = 0.0
            return
        delta: float = x_log - self.mu_log
        self.mu_log += (1 - self.beta) * delta
        self.var_log = self.beta * (self.var_log or 0.0) + (1 - self.beta) * (delta * delta)

    @torch.no_grad()
    def maybe_rescale_(self, params: Iterator[nn.Parameter]) -> Tuple[float, bool]:
        """
        Check for gradient spikes and rescale if necessary.
        
        Call after loss.backward().
        
        Args:
            params: Iterator over model parameters
            
        Returns:
            (scale, spiked): scale factor applied (<=1.0) and whether a spike was detected
        """
        self.step += 1
        # Convert to list to allow multiple iterations
        params_list = list(params)
        cur: float = self._total_grad_norm(iter(params_list))
        cur_log: float = math.log(cur + self.eps)

        # Bootstrap stats
        if self.mu_log is None:
            self._ema_update(cur_log)
            return 1.0, False

        # Warmup: build baseline, never rescale
        if self.step <= self.warmup_steps:
            self._ema_update(cur_log)
            return 1.0, False

        std_log: float = math.sqrt((self.var_log or 0.0) + self.eps)
        thresh_log: float = self.mu_log + math.log(self.mult)
        z: float = (cur_log - self.mu_log) / (std_log + self.eps)

        spiked: bool = (cur_log > thresh_log) and (z > self.z_thresh)

        if spiked:
            # Only-downscale to the threshold; skip EMA update on this step
            scale: float = math.exp(thresh_log - cur_log)  # = target/cur in linear space
            if scale < 1.0:
                for p in params_list:
                    if p.grad is not None:
                        p.grad.mul_(scale)
            return scale, True
        else:
            # Normal step: update stats
            self._ema_update(cur_log)
            return 1.0, False