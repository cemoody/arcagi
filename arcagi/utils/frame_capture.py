"""Frame capture utility for visualizing neural network forward passes."""

import torch
from typing import List, Optional, Callable, Any
import numpy as np
from PIL import Image
import io
import os

class FrameCapture:
    """Minimal frame capture utility that can be injected into forward passes."""
    
    def __init__(self):
        self.frames: List[torch.Tensor] = []
        self.enabled = False
        self.transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    
    def set_transform(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        """Set a transformation function to apply to captured tensors before storing."""
        self.transform_fn = fn
        return self
    
    def capture(self, tensor: torch.Tensor, round_idx: Optional[int] = None):
        """Capture a frame if enabled. This is the single line to inject into forward pass."""
        if self.enabled:
            frame = tensor.detach().clone()
            if self.transform_fn:
                frame = self.transform_fn(frame)
            self.frames.append(frame)
    
    def clear(self):
        """Clear captured frames."""
        self.frames = []
    
    def enable(self):
        """Enable frame capture."""
        self.enabled = True
        self.clear()
        return self
    
    def disable(self):
        """Disable frame capture."""
        self.enabled = False
        return self
    
    def to_gif(self, filename: str, duration_ms: int = 100, color_map: Optional[dict] = None):
        """Convert captured frames to animated GIF.
        
        Args:
            filename: Output filename for the GIF
            duration_ms: Duration per frame in milliseconds
            color_map: Optional color mapping for ARC tasks (0-9 to RGB)
        """
        if not self.frames:
            raise ValueError("No frames captured")
        
        # Default ARC color palette
        if color_map is None:
            color_map = {
                0: (0, 0, 0),        # Black
                1: (0, 116, 217),    # Blue
                2: (255, 65, 54),    # Red
                3: (46, 204, 64),    # Green
                4: (255, 220, 0),    # Yellow
                5: (128, 128, 128),  # Gray
                6: (240, 18, 190),   # Magenta
                7: (255, 133, 27),   # Orange
                8: (0, 191, 255),    # Sky Blue
                9: (149, 0, 58),     # Maroon
                -1: (255, 255, 255), # White (for mask)
            }
        
        pil_images = []
        
        for frame in self.frames:
            # Handle different tensor shapes
            if frame.dim() == 4:  # [B, H, W, C] or [B, C, H, W]
                # Take first item in batch
                frame = frame[0]
            
            if frame.dim() == 3:
                # If it's [H, W, C] with C > 3, likely logits - take argmax
                if frame.shape[-1] > 3:
                    frame = frame.argmax(dim=-1)
                # If it's [C, H, W] format, transpose
                elif frame.shape[0] <= 3:
                    frame = frame.permute(1, 2, 0)
            
            # Convert to numpy
            if frame.dim() == 2:  # [H, W] - single channel
                frame_np = frame.cpu().numpy().astype(np.int32)
                # Map colors
                h, w = frame_np.shape
                rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
                for value, rgb in color_map.items():
                    mask = frame_np == value
                    rgb_array[mask] = rgb
                img = Image.fromarray(rgb_array)
            else:
                # Assume RGB
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                if frame_np.shape[-1] == 1:
                    frame_np = np.repeat(frame_np, 3, axis=-1)
                img = Image.fromarray(frame_np)
            
            # Optionally upscale for better visibility
            img = img.resize((img.width * 10, img.height * 10), Image.NEAREST)
            pil_images.append(img)
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(filename)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        # Save as GIF
        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration_ms,
            loop=0
        )
        
        return filename
    
    def __len__(self):
        return len(self.frames)