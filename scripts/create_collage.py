#!/usr/bin/env python3
"""Create a collage of frames showing key moments in the video generation process"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_collage(frames_dir, output_path, num_cols=8, num_rows=4, frame_indices=None):
    """
    Create a collage grid of frames
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Path to save the collage
        num_cols: Number of columns in the grid
        num_rows: Number of rows in the grid
        frame_indices: Specific frame indices to include (if None, evenly spaced)
    """
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return
    
    # Determine which frames to include
    if frame_indices is None:
        # Evenly space frames across the video
        num_frames = num_cols * num_rows
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # Load first frame to get dimensions
    first_frame = Image.open(os.path.join(frames_dir, frame_files[0]))
    frame_w, frame_h = first_frame.size
    
    # Create collage canvas
    canvas_w = frame_w * num_cols
    canvas_h = frame_h * num_rows
    collage = Image.new('RGB', (canvas_w, canvas_h), color='black')
    
    # Place frames in grid
    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx >= total_frames:
            break
            
        row = idx // num_cols
        col = idx % num_cols
        
        frame_path = os.path.join(frames_dir, frame_files[frame_idx])
        frame = Image.open(frame_path)
        
        # Paste frame into collage
        x = col * frame_w
        y = row * frame_h
        collage.paste(frame, (x, y))
        
        # Add frame number overlay
        draw = ImageDraw.Draw(collage)
        text = f"Frame {frame_idx}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text with black background
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x+5, y+5, x+text_w+15, y+text_h+15], fill='black')
        draw.text((x+10, y+10), text, fill='white', font=font)
    
    # Save collage
    collage.save(output_path, quality=95)
    print(f"✓ Collage saved to {output_path}")
    print(f"  Grid: {num_cols}×{num_rows} = {len(frame_indices)} frames")
    print(f"  Size: {canvas_w}×{canvas_h} pixels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a collage of video frames")
    parser.add_argument("--frames-dir", required=True, help="Directory containing frames")
    parser.add_argument("--output", required=True, help="Output collage path")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows")
    
    args = parser.parse_args()
    create_collage(args.frames_dir, args.output, args.cols, args.rows)
