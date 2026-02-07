"""
Video Assembly Module
Compiles individual frames into a final MP4 video using OpenCV.
"""

import os
import glob
import json
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def collect_and_fill_frames(frames_dir: str, expected_num_frames: Optional[int] = None) -> List[str]:
    """
    Collect frames and fill missing ones by copying nearest previous frame.
    
    Args:
        frames_dir: Directory containing frame images
        expected_num_frames: Expected number of frames (from trajectory plan)
    
    Returns:
        List of sorted frame paths
    """
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    indices_present = {int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files}
    
    if expected_num_frames is not None:
        missing = [i for i in range(expected_num_frames) if i not in indices_present]
        if missing:
            print(f"[WARN] Missing frames: {missing[:50]} (showing up to 50). Filling with nearest previous frames.")
            for mi in missing:
                prev_candidates = [i for i in range(mi-1, -1, -1) if i in indices_present]
                if prev_candidates:
                    prev_idx = prev_candidates[0]
                else:
                    # fallback to first available frame
                    prev_idx = min(indices_present) if indices_present else None
                if prev_idx is None:
                    raise RuntimeError("No frames available to fill missing frames.")
                src = os.path.join(frames_dir, f"frame_{prev_idx:04d}.png")
                dst = os.path.join(frames_dir, f"frame_{mi:04d}.png")
                shutil.copy(src, dst)
                indices_present.add(mi)
            # refresh files list
            files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    return files


class VideoAssembler:
    """Assembles frames into video."""

    def __init__(self, fps: int = 15, codec: str = "mp4v"):
        """
        Initialize video assembler.
        
        Args:
            fps: Frames per second
            codec: Video codec (mp4v, XVID, etc.)
        """
        self.fps = fps
        self.codec = codec

    def assemble_video(
        self,
        frame_dir: str,
        output_path: str = "outputs/final_video.mp4",
        frame_pattern: str = "frame_*.png",
        width: Optional[int] = None,
        height: Optional[int] = None,
        verbose: bool = True,
        trajectory_plan_path: Optional[str] = None
    ) -> str:
        """
        Assemble video from frames.
        
        Args:
            frame_dir: Directory containing frame images
            output_path: Path to save output MP4
            frame_pattern: Glob pattern for frame files (e.g., "frame_*.png")
            width: Output video width (auto-detect if None)
            height: Output video height (auto-detect if None)
            verbose: Whether to print progress
            trajectory_plan_path: Path to trajectory plan JSON (for expected frame count)
            
        Returns:
            Path to output video
        """
        # Load expected frame count from trajectory plan if available
        expected_num_frames = None
        if trajectory_plan_path and os.path.exists(trajectory_plan_path):
            with open(trajectory_plan_path, 'r') as f:
                plan = json.load(f)
                expected_num_frames = plan.get("num_frames", len(plan.get("frames", [])))
        
        # Collect frames and fill missing ones
        frame_paths = collect_and_fill_frames(frame_dir, expected_num_frames)
        
        if not frame_paths:
            raise FileNotFoundError(f"No frames found matching {frame_pattern} in {frame_dir}")
        
        if verbose:
            print(f"Found {len(frame_paths)} frames")
            print(f"Frame pattern: {frame_pattern}")
        
        # Load first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frame_paths[0]}")
        
        h, w = first_frame.shape[:2]
        
        # Override dimensions if specified
        if width is not None:
            w = width
        if height is not None:
            h = height
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")
        
        if verbose:
            print(f"\nWriting video to {output_path}")
            print(f"  Resolution: {w}x{h}")
            print(f"  FPS: {self.fps}")
            print(f"  Duration: {len(frame_paths) / self.fps:.2f}s")
            pbar = tqdm(total=len(frame_paths), desc="Assembling", unit="frame")
        
        # Write frames
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Warning: Could not read {frame_path}")
                continue
            
            # Resize if necessary
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            writer.write(frame)
            
            if verbose:
                pbar.update(1)
        
        writer.release()
        
        if verbose:
            pbar.close()
            print(f"\n✓ Video saved to {output_path}")
        
        return output_path

    def create_preview_collage(
        self,
        frame_dir: str,
        output_path: str = "outputs/preview_collage.png",
        frame_pattern: str = "frame_*.png",
        num_frames: int = 8
    ) -> str:
        """
        Create a collage of evenly-spaced frames for quick preview.
        
        Args:
            frame_dir: Directory containing frames
            output_path: Path to save collage
            frame_pattern: Glob pattern for frames
            num_frames: Number of frames to include in collage
            
        Returns:
            Path to collage image
        """
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, frame_pattern)))
        
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {frame_dir}")
        
        # Select evenly-spaced frames
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
        selected_frames = [frame_paths[i] for i in indices]
        
        # Load images
        images = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)
            images.append(img)
        
        # Create grid collage
        cols = 4
        rows = (num_frames + cols - 1) // cols
        
        img_width, img_height = images[0].size
        collage_width = cols * img_width + (cols - 1) * 10
        collage_height = rows * img_height + (rows - 1) * 10
        
        collage = Image.new("RGB", (collage_width, collage_height), color=(255, 255, 255))
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * (img_width + 10)
            y = row * (img_height + 10)
            collage.paste(img, (x, y))
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        collage.save(output_path)
        
        print(f"✓ Preview collage saved to {output_path}")
        return output_path


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assemble video from frames")
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame images")
    parser.add_argument("--output", default="outputs/final_video.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--codec", default="mp4v", help="Video codec")
    parser.add_argument("--width", type=int, help="Output video width")
    parser.add_argument("--height", type=int, help="Output video height")
    parser.add_argument("--preview", action="store_true", help="Create preview collage")
    parser.add_argument("--trajectory-plan", type=str, help="Path to trajectory plan JSON")
    
    args = parser.parse_args()
    
    assembler = VideoAssembler(fps=args.fps, codec=args.codec)
    
    # Assemble video
    assembler.assemble_video(
        frame_dir=args.frames_dir,
        output_path=args.output,
        width=args.width,
        height=args.height,
        trajectory_plan_path=args.trajectory_plan
    )
    
    # Create preview if requested
    if args.preview:
        preview_path = os.path.join(
            os.path.dirname(args.output),
            "preview_collage.png"
        )
        assembler.create_preview_collage(
            frame_dir=args.frames_dir,
            output_path=preview_path
        )


if __name__ == "__main__":
    main()
