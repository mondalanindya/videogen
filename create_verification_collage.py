#!/usr/bin/env python3
"""Create verification collage from generated keyframes"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def main(args):
    frames_dir = Path(args.frames_dir)
    sample_frames = args.sample_frames
    
    if not frames_dir.exists():
        print(f"Error: Directory {frames_dir} does not exist")
        return
    
    images = []
    for i in sample_frames:
        frame_path = frames_dir / f"keyframe_{i:04d}.png"
        if not frame_path.exists():
            print(f"Warning: {frame_path} not found, skipping")
            continue
        
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Warning: Could not read {frame_path}")
            continue
        
        # Add frame number annotation
        cv2.putText(img, f"Frame {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        images.append(img)
    
    if not images:
        print("Error: No images loaded")
        return
    
    # Create grid
    n_cols = 3
    n_rows = (len(images) + n_cols - 1) // n_cols
    
    # Pad with blank images if needed
    while len(images) < n_rows * n_cols:
        images.append(np.zeros_like(images[0]))
    
    rows = []
    for i in range(0, len(images), n_cols):
        row = np.hstack(images[i:i+n_cols])
        rows.append(row)
    
    collage = np.vstack(rows)
    
    output_path = args.output
    cv2.imwrite(output_path, collage)
    print(f"✓ Collage saved to {output_path}")
    print(f"  Contains {len(sample_frames)} frames in {n_rows}×{n_cols} grid")
    print(f"  Size: {collage.shape[1]}×{collage.shape[0]} pixels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="outputs/ip_keyframes_clean")
    parser.add_argument("--sample-frames", type=int, nargs="+",
                        default=[0, 20, 40, 60, 80, 100, 120, 140, 159])
    parser.add_argument("--output", default="outputs/collage_verification.jpg")
    args = parser.parse_args()
    main(args)
