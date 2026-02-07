#!/usr/bin/env python3
"""
Utility to add soft shadows under ball composites for consistent contact appearance.
"""

import cv2
import numpy as np


def add_soft_shadow(image_np, center_x, center_y, radius, strength=0.35, y_offset=0.95):
    """
    Add a soft elliptical shadow under the ball for visual grounding.
    
    Args:
        image_np: BGR numpy image (H, W, 3)
        center_x, center_y: ball center pixel coordinates
        radius: ball radius in pixels
        strength: shadow darkness (0..1)
        y_offset: vertical offset multiplier (shadow slightly below center)
    
    Returns:
        image_np_with_shadow: BGR image with shadow composited
    """
    h, w = image_np.shape[:2]
    
    # Elliptical shadow dimensions proportional to radius
    shadow_w = int(round(radius * 1.6))
    shadow_h = int(round(radius * 0.5))
    cy = int(round(center_y + radius * y_offset))
    cx = int(round(center_x))
    
    # Clamp to image bounds
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    
    # Create shadow mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (shadow_w, shadow_h), 0, 0, 360, 255, -1)
    
    # Gaussian blur for soft shadow
    k = max(7, int(round(radius / 2)) | 1)  # Odd kernel size
    mask_blur = cv2.GaussianBlur(mask, (k, k), 0)
    
    # Blend shadow: darken pixels under the mask
    alpha = (mask_blur.astype(np.float32) / 255.0) * strength
    out = image_np.astype(np.float32)
    
    # Apply darkening (multiply by ~0.6 under shadow)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - alpha * 0.4)
    
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
