#!/usr/bin/env python3
"""
Fast video assembly from existing warped frames.
"""
import os
import cv2
from pathlib import Path
import sys

def assemble_video(frame_dir, output_video, fps=24):
    """Assemble video from PNG frames."""
    frame_dir = Path(frame_dir)
    frames = sorted([f for f in frame_dir.glob('*.png')])
    
    if not frames:
        print(f"❌ No PNG frames found in {frame_dir}")
        return False
    
    print(f"📹 Assembling video from {len(frames)} frames...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️  Warning: Could not read frame {frame_path}")
            continue
        out.write(frame)
        if (i + 1) % 20 == 0:
            print(f"  → Wrote {i + 1}/{len(frames)} frames")
    
    out.release()
    print(f"✓ Video saved to {output_video}")
    return True

if __name__ == "__main__":
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/warp_frames"
    output = sys.argv[2] if len(sys.argv) > 2 else "outputs/motion_craft_output.mp4"
    
    assemble_video(frame_dir, output)
