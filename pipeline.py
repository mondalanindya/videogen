"""
Main Pipeline Orchestrator
Coordinates the entire video generation workflow from prompt to final video.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
import sys

# Add scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "scripts"))

from plan_trajectory import TrajectoryPlanner
from generate_keyframes import KeyframeGenerator
from latent_warp_and_edit import IntermediateFrameGenerator
from composite_frames import CompositeFrameGenerator
from assemble_video import VideoAssembler


class VideoPipeline:
    """Main pipeline for end-to-end video generation."""

    def __init__(self, output_dir: str = "outputs", device: str = "cuda", verbose: bool = True):
        """
        Initialize the video generation pipeline.
        
        Args:
            output_dir: Base output directory
            device: torch device (cuda or cpu)
            verbose: Whether to print progress
        """
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose
        self.frames_dir = os.path.join(output_dir, "frames")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        if self.verbose:
            print("=" * 60)
            print("VIDEO GENERATION PIPELINE")
            print("=" * 60)
            print(f"Output directory: {self.output_dir}")
            print(f"Device: {self.device}")

    def run(
        self,
        first_frame_path: str,
        prompt: str,
        num_frames: int = 16,
        intermediate_steps: int = 2,
        fps: int = 15,
        skip_steps: Optional[list] = None,
    ) -> str:
        """
        Run the complete video generation pipeline.
        
        Args:
            first_frame_path: Path to initial frame image
            prompt: Motion description prompt
            num_frames: Total frames to generate
            intermediate_steps: Intermediate frames between keyframes
            fps: Output video frames per second
            skip_steps: List of steps to skip (for testing/debugging)
                       Options: "trajectory", "keyframes", "interpolation", "assembly"
            
        Returns:
            Path to final video file
        """
        skip_steps = skip_steps or []
        
        # Step 1: Trajectory Planning
        if "trajectory" not in skip_steps:
            print("\n" + "=" * 60)
            print("STEP 1: TRAJECTORY PLANNING")
            print("=" * 60)
            
            trajectory_path = os.path.join(self.output_dir, "trajectory_plan.json")
            
            try:
                planner = TrajectoryPlanner(device=self.device)
                trajectory_plan = planner.plan_trajectory(
                    first_frame_path=first_frame_path,
                    prompt=prompt,
                    num_frames=num_frames,
                    output_path=trajectory_path
                )
            except Exception as e:
                print(f"✗ Trajectory planning failed: {e}")
                raise
        else:
            print("\n⊘ Skipping trajectory planning")
            trajectory_path = os.path.join(self.output_dir, "trajectory_plan.json")
            if not os.path.exists(trajectory_path):
                raise FileNotFoundError(f"Trajectory plan not found at {trajectory_path}")
            with open(trajectory_path, "r") as f:
                trajectory_plan = json.load(f)
        
        # Step 2: Keyframe Generation
        if "keyframes" not in skip_steps:
            print("\n" + "=" * 60)
            print("STEP 2: KEYFRAME GENERATION")
            print("=" * 60)
            
            try:
                keyframe_generator = KeyframeGenerator(device=self.device, use_xformers=True)
                keyframe_paths = keyframe_generator.generate_keyframes(
                    first_frame_path=first_frame_path,
                    trajectory_plan=trajectory_plan,
                    output_dir=self.frames_dir,
                    prompt_template="A red ball on a wooden table"
                )
            except Exception as e:
                print(f"✗ Keyframe generation failed: {e}")
                raise
        else:
            print("\n⊘ Skipping keyframe generation")
            keyframe_paths = {}
        
        # Step 3: Frame Generation via Compositing
        if "interpolation" not in skip_steps:
            print("\n" + "=" * 60)
            print("STEP 3: FRAME GENERATION (COMPOSITING)")
            print("=" * 60)
            
            try:
                # Use compositing approach for consistent, positioned frames
                compositor = CompositeFrameGenerator()
                all_frame_paths = compositor.generate_all_frames(
                    first_frame_path=first_frame_path,
                    trajectory_plan=trajectory_plan,
                    output_dir=self.frames_dir
                )
            except Exception as e:
                print(f"✗ Frame generation failed: {e}")
                raise
        else:
            print("\n⊘ Skipping frame generation")
            all_frame_paths = []
        
        # Step 4: Video Assembly
        if "assembly" not in skip_steps:
            print("\n" + "=" * 60)
            print("STEP 4: VIDEO ASSEMBLY")
            print("=" * 60)
            
            try:
                video_path = os.path.join(self.output_dir, "final_video.mp4")
                
                assembler = VideoAssembler(fps=fps, codec="mp4v")
                output_video = assembler.assemble_video(
                    frame_dir=self.frames_dir,
                    output_path=video_path,
                    verbose=self.verbose
                )
                
                # Create preview collage
                preview_path = os.path.join(self.output_dir, "preview_collage.png")
                assembler.create_preview_collage(
                    frame_dir=self.frames_dir,
                    output_path=preview_path
                )
                
            except Exception as e:
                print(f"✗ Video assembly failed: {e}")
                raise
        else:
            print("\n⊘ Skipping video assembly")
            output_video = None
        
        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        if output_video:
            print(f"\n✓ Final video: {output_video}")
            print(f"  Trajectory: {os.path.join(self.output_dir, 'trajectory_plan.json')}")
            print(f"  Frames: {self.frames_dir}")
            print(f"  Preview: {preview_path}")
            return output_video
        else:
            print("\n⚠ Video assembly was skipped")
            return None


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="End-to-end video generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py \\
    --first-frame inputs/first_frame.png \\
    --prompt "A red ball bounces five times on a wooden table"
  
  python pipeline.py \\
    --first-frame inputs/first_frame.png \\
    --prompt "A blue cube rolls across the floor" \\
    --num-frames 32 \\
    --fps 24
  
  python pipeline.py \\
    --first-frame inputs/first_frame.png \\
    --prompt "A ball moves in a circle" \\
    --skip-steps trajectory keyframes \\
    --output-dir outputs_test
        """
    )
    
    parser.add_argument(
        "--first-frame",
        required=True,
        help="Path to first frame image"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Motion description (e.g., 'A red ball bounces 5 times')"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Total frames to generate (default: 16)"
    )
    parser.add_argument(
        "--intermediate-steps",
        type=int,
        default=2,
        help="Intermediate frames between keyframes (default: 2)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output video FPS (default: 15)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        choices=["trajectory", "keyframes", "interpolation", "assembly"],
        help="Steps to skip (for testing)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.first_frame):
        print(f"✗ Error: First frame not found: {args.first_frame}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = VideoPipeline(
        output_dir=args.output_dir,
        device=args.device,
        verbose=not args.quiet
    )
    
    try:
        output_video = pipeline.run(
            first_frame_path=args.first_frame,
            prompt=args.prompt,
            num_frames=args.num_frames,
            intermediate_steps=args.intermediate_steps,
            fps=args.fps,
            skip_steps=args.skip_steps
        )
        
        if output_video:
            print(f"\n✓ Success! Video saved to: {output_video}")
            sys.exit(0)
        else:
            print("\n⚠ Pipeline completed but video was not generated")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
