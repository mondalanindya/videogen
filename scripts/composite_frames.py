"""
Composite Frame Generator
Creates frames by compositing a ball sprite onto background at trajectory positions.
"""

import json
import os
from typing import Dict, List, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2


class CompositeFrameGenerator:
    """Generates frames by compositing a ball sprite onto background."""

    def __init__(self):
        """Initialize the compositor."""
        pass

    def extract_ball_sprite(
        self,
        first_frame_path: str,
        center_x: int,
        center_y: int,
        radius: int,
        output_path: str = "outputs/ball_sprite.png"
    ) -> str:
        """
        Extract ball sprite from first frame.
        
        Args:
            first_frame_path: Path to first frame
            center_x: Ball center x
            center_y: Ball center y
            radius: Ball radius
            output_path: Where to save sprite
            
        Returns:
            Path to saved sprite
        """
        frame = Image.open(first_frame_path).convert("RGBA")
        
        # Create a square crop around the ball with padding
        padding = 15
        size = (radius + padding) * 2
        left = center_x - radius - padding
        top = center_y - radius - padding
        right = center_x + radius + padding
        bottom = center_y + radius + padding
        
        # Ensure bounds are within image
        left = max(0, left)
        top = max(0, top)
        right = min(frame.width, right)
        bottom = min(frame.height, bottom)
        
        # Crop the ball region
        ball_crop = frame.crop((left, top, right, bottom))
        
        # Create circular alpha mask
        mask = Image.new("L", ball_crop.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Center of the cropped region
        cx = radius + padding
        cy = radius + padding
        
        # Draw circle with smooth edges
        draw.ellipse(
            [cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2],
            fill=255
        )
        
        # Apply Gaussian blur for smooth edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Apply mask to create transparent background
        ball_crop.putalpha(mask)
        
        # Save sprite
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ball_crop.save(output_path)
        print(f"✓ Ball sprite saved to {output_path}")
        
        return output_path

    def generate_frames_composite(
        self,
        background_path: str,
        ball_sprite_path: str,
        trajectory_plan: Dict[str, Any],
        output_dir: str = "outputs/frames"
    ) -> List[str]:
        """
        Generate all frames by compositing ball sprite onto background.
        
        Args:
            background_path: Path to clean background image
            ball_sprite_path: Path to ball sprite with alpha
            trajectory_plan: Trajectory data
            output_dir: Output directory
            
        Returns:
            List of generated frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        background = Image.open(background_path).convert("RGBA")
        ball_sprite_orig = Image.open(ball_sprite_path).convert("RGBA")
        
        frames_data = trajectory_plan["frames"]
        frame_paths = []
        
        print(f"Compositing {len(frames_data)} frames with squash/stretch...")
        
        for i, frame_info in enumerate(frames_data):
            x, y = frame_info["x"], frame_info["y"]
            radius = frame_info["radius"]
            velocity_y = frame_info.get("velocity_y", 0)
            
            # Create frame with ball at position
            frame = background.copy()
            
            # Apply squash/stretch based on velocity
            ball_sprite = self._apply_squash_stretch(ball_sprite_orig, velocity_y)
            
            # Add subtle shadow below ball
            shadow = self._create_shadow(background.size, x, y + 5, radius + 5)
            frame = Image.alpha_composite(frame, shadow)
            
            # Paste ball sprite centered at (x, y)
            ball_w, ball_h = ball_sprite.size
            paste_x = x - ball_w // 2
            paste_y = y - ball_h // 2
            
            frame.paste(ball_sprite, (paste_x, paste_y), ball_sprite)
            
            # Convert back to RGB and save
            frame_rgb = frame.convert("RGB")
            output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            frame_rgb.save(output_path)
            frame_paths.append(output_path)
            
            if i % 4 == 0:
                print(f"  ✓ Frame {i}: ball at ({x}, {y})")
        
        print(f"✓ Generated {len(frame_paths)} frames")
        return frame_paths

    def _apply_squash_stretch(
        self,
        sprite: Image.Image,
        velocity_y: float
    ) -> Image.Image:
        """Apply squash/stretch deformation based on velocity."""
        # Scale factors based on velocity
        stretch_factor = 1.0 + abs(velocity_y) * 0.02
        squash_factor = 1.0 / stretch_factor
        
        w, h = sprite.size
        
        if velocity_y > 2:  # Moving down fast - squash vertically
            new_w = int(w * stretch_factor)
            new_h = int(h * squash_factor)
        elif velocity_y < -2:  # Moving up fast - stretch vertically
            new_w = int(w * squash_factor)
            new_h = int(h * stretch_factor)
        else:  # Slow or at rest - normal
            return sprite
        
        # Resize with smooth interpolation
        deformed = sprite.resize((new_w, new_h), Image.LANCZOS)
        return deformed

    def _create_shadow(
        self,
        size: tuple,
        x: int,
        y: int,
        radius: int
    ) -> Image.Image:
        """Create a soft shadow for the ball."""
        shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        
        # Draw soft ellipse shadow
        shadow_color = (0, 0, 0, 60)  # Semi-transparent black
        draw.ellipse(
            [x - radius, y - radius//3, x + radius, y + radius//3],
            fill=shadow_color
        )
        
        # Blur for softness
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        
        return shadow

    def generate_all_frames(
        self,
        first_frame_path: str,
        trajectory_plan: Dict[str, Any],
        output_dir: str = "outputs/frames"
    ) -> List[str]:
        """
        Complete workflow: extract sprite, create background, composite frames.
        
        Args:
            first_frame_path: Path to initial frame
            trajectory_plan: Trajectory data
            output_dir: Output directory
            
        Returns:
            List of frame paths
        """
        base_dir = os.path.dirname(output_dir)
        
        # Get initial ball position from trajectory
        first_frame_data = trajectory_plan["frames"][0]
        ball_x = first_frame_data["x"]
        ball_y = first_frame_data["y"]
        ball_radius = first_frame_data["radius"]
        
        # Step 1: Extract ball sprite
        sprite_path = os.path.join(base_dir, "ball_sprite.png")
        self.extract_ball_sprite(
            first_frame_path,
            ball_x,
            ball_y,
            ball_radius,
            sprite_path
        )
        
        # Step 2: Create clean background by inpainting out the ball
        # For now, use a simple approach: load existing background if available
        background_path = os.path.join(base_dir, "background.png")
        
        if not os.path.exists(background_path):
            # Create background by removing ball region (simple fill)
            background = Image.open(first_frame_path).convert("RGB")
            # Use inpainting or simple blur to remove ball
            # For simplicity, we'll assume background.png is created by keyframe generator
            print(f"⚠ Background not found at {background_path}")
            print("  Using first frame as background (ball will remain in place)")
            background_path = first_frame_path
        
        # Step 3: Composite all frames
        frame_paths = self.generate_frames_composite(
            background_path,
            sprite_path,
            trajectory_plan,
            output_dir
        )
        
        return frame_paths


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate frames via compositing")
    parser.add_argument("--first-frame", required=True, help="Path to first frame")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory JSON")
    parser.add_argument("--output-dir", default="outputs/frames", help="Output directory")
    parser.add_argument("--background", help="Path to background image (optional)")
    
    args = parser.parse_args()
    
    # Load trajectory
    with open(args.trajectory, "r") as f:
        trajectory_plan = json.load(f)
    
    generator = CompositeFrameGenerator()
    
    if args.background:
        # Manual workflow with provided background
        first_data = trajectory_plan["frames"][0]
        sprite_path = "outputs/ball_sprite.png"
        generator.extract_ball_sprite(
            args.first_frame,
            first_data["x"],
            first_data["y"],
            first_data["radius"],
            sprite_path
        )
        generator.generate_frames_composite(
            args.background,
            sprite_path,
            trajectory_plan,
            args.output_dir
        )
    else:
        # Automatic workflow
        generator.generate_all_frames(
            args.first_frame,
            trajectory_plan,
            args.output_dir
        )


if __name__ == "__main__":
    main()
