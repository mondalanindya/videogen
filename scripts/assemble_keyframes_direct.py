#!/usr/bin/env python3
"""
Assemble keyframes directly into video without optical flow warping.
This preserves the ball patches we carefully placed at trajectory positions.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def main(keyframes_dir, trajectory_json, output_video, fps=24):
    """Assemble keyframes with linear interpolation between them."""
    
    # Load trajectory to get keyframe positions and timing
    with open(trajectory_json, 'r') as f:
        plan = json.load(f)
    
    frames = plan.get("frames", [])
    num_frames = plan.get("num_frames", 160)
    
    # Get keyframe indices from available files
    keyframe_files = sorted(Path(keyframes_dir).glob("keyframe_*.png"))
    if not keyframe_files:
        raise RuntimeError(f"No keyframes found in {keyframes_dir}")
    
    # Extract keyframe indices
    keyframe_indices = []
    keyframe_images = {}
    for kf_file in keyframe_files:
        idx = int(kf_file.stem.split('_')[1])
        keyframe_indices.append(idx)
        keyframe_images[idx] = cv2.imread(str(kf_file))
    
    keyframe_indices = sorted(keyframe_indices)
    print(f"Found {len(keyframe_indices)} keyframes at indices: {keyframe_indices}")
    
    # Get frame dimensions from first keyframe
    first_kf = cv2.imread(str(keyframe_files[0]))
    H, W = first_kf.shape[:2]
    
    # Build output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video}")
    
    # Generate frames by linear interpolation between keyframes
    out_frames = []
    for t in range(num_frames):
        # Find surrounding keyframes
        before_idx = None
        after_idx = None
        
        for idx in keyframe_indices:
            if idx <= t:
                before_idx = idx
            if idx >= t and after_idx is None:
                after_idx = idx
        
        if before_idx is None and after_idx is None:
            # No keyframes available (shouldn't happen)
            print(f"Warning: No keyframes available for frame {t}")
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        elif before_idx is None:
            # Before first keyframe, use first keyframe
            frame = keyframe_images[after_idx].copy()
        elif after_idx is None:
            # After last keyframe, use last keyframe
            frame = keyframe_images[before_idx].copy()
        elif before_idx == after_idx:
            # Exact keyframe match
            frame = keyframe_images[before_idx].copy()
        else:
            # Interpolate between two keyframes
            alpha = (t - before_idx) / (after_idx - before_idx)
            img_before = keyframe_images[before_idx].astype(np.float32)
            img_after = keyframe_images[after_idx].astype(np.float32)
            frame = (1.0 - alpha) * img_before + alpha * img_after
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        writer.write(frame)
        if (t + 1) % 20 == 0:
            print(f"  → Wrote {t+1}/{num_frames} frames")
    
    writer.release()
    print(f"✓ Video saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("keyframes_dir", help="Directory with keyframe_*.png files")
    parser.add_argument("trajectory_json", help="Path to trajectory_plan.json")
    parser.add_argument("output_video", help="Output video path")
    parser.add_argument("--fps", default=24, type=int, help="Output FPS")
    args = parser.parse_args()
    
    main(args.keyframes_dir, args.trajectory_json, args.output_video, fps=args.fps)
