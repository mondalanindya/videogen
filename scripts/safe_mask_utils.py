#!/usr/bin/env python3
"""
Safe mask creation utilities for keyframe generation.
Handles coordinate conversion, bounds checking, and debug outputs.
"""
import cv2
import numpy as np
import os


def to_pixel_coords(x, y, img_w, img_h):
    """Convert possibly-normalized coords in [0,1] or absolute pixel coords to integers."""
    if 0.0 <= x <= 1.01 and 0.0 <= y <= 1.01:
        cx = int(round(float(x) * img_w))
        cy = int(round(float(y) * img_h))
    else:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
    # clip
    cx = max(0, min(img_w - 1, cx))
    cy = max(0, min(img_h - 1, cy))
    return cx, cy


def make_destination_mask(img_path, dst_x, dst_y, radius, out_mask_path=None, pad=2):
    """
    Create a uint8 mask (0/255) with a filled circle at destination.
    Returns (mask_np, (cx,cy,rad)).
    Saves mask to out_mask_path if provided.
    """
    im = cv2.imread(img_path)
    if im is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    h, w = im.shape[:2]
    cx, cy = to_pixel_coords(dst_x, dst_y, w, h)
    rad = int(round(radius))
    rad = max(1, min(min(w, h)//2, rad))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), rad + pad, 255, -1)
    if out_mask_path:
        os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
        cv2.imwrite(out_mask_path, mask)
    return mask, (cx, cy, rad + pad)


def erase_source_region(bg_path, source_mask_np, out_erased_path=None, fill_color=None):
    """
    Erase source ball region from background using source_mask_np (0/255).
    Returns erased_image_path (writes to out_erased_path).
    If fill_color is None, fill by ambient average color of the image edges.
    """
    img = cv2.imread(bg_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {bg_path}")
    h, w = img.shape[:2]
    mask = (source_mask_np > 0).astype(np.uint8)
    if fill_color is None:
        # estimate ambient color by sampling borders
        top = img[0:10, :, :].reshape(-1, 3)
        left = img[:, 0:10, :].reshape(-1, 3)
        col = np.concatenate([top, left], axis=0).mean(axis=0).astype(np.uint8).tolist()
    else:
        col = fill_color
    erased = img.copy()
    erased[mask == 1] = col
    if out_erased_path:
        os.makedirs(os.path.dirname(out_erased_path), exist_ok=True)
        cv2.imwrite(out_erased_path, erased)
    return erased
