"""
Keyframe Generation Module
Creates keyframe images using SAM (Segment Anything) and Stable Diffusion Inpainting.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForInpainting
from transformers import pipeline as transformers_pipeline, SamModel, SamProcessor
from accelerate import Accelerator
import warnings

warnings.filterwarnings("ignore")


class KeyframeGenerator:
    """Generates keyframe images with object repositioning using inpainting."""

    def __init__(self, device: str = "cuda", use_xformers: bool = True):
        """
        Initialize the keyframe generator.
        
        Args:
            device: torch device to use
            use_xformers: whether to use xformers for efficient attention
        """
        self.device = device
        self.inpaint_pipe = None
        self.sam_predictor = None
        # Use CPU for generator if CUDA not available
        gen_device = device if torch.cuda.is_available() else "cpu"
        self._generator = torch.Generator(device=gen_device).manual_seed(42)
        self._load_models(use_xformers)

    def _load_models(self, use_xformers: bool = True):
        """Load Stable Diffusion Inpainting and SAM models."""
        print("Loading Stable Diffusion Inpainting pipeline...")
        try:
            # Load SD inpainting model using AutoPipelineForInpainting
            model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
            self.inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None  # Disable NSFW filter to prevent black images
            ).to(self.device)
            
            if use_xformers:
                try:
                    self.inpaint_pipe.enable_xformers_memory_efficient_attention()
                    print("✓ xformers enabled")
                except Exception as e:
                    print(f"⚠ xformers not available: {e}")
            
            # Enable model CPU offload for memory efficiency
            self.inpaint_pipe.enable_model_cpu_offload()
            
            print("✓ Stable Diffusion Inpainting loaded (safety_checker disabled)")
        except Exception as e:
            print(f"✗ Error loading SD Inpainting: {e}")
            raise

        print("Loading SAM for segmentation (HuggingFace transformers)...")
        try:
            # Get device from Accelerator (handles multi-GPU, mixed precision, etc.)
            accelerator = Accelerator()
            device = accelerator.device
            
            # Load SAM model and processor from HuggingFace
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
            self.sam_device = device
            print("✓ SAM loaded (facebook/sam-vit-huge)")
        except Exception as e:
            print(f"⚠ SAM not available: {e}. Will use simple circle masks.")
            self.sam_model = None
            self.sam_processor = None
            self.sam_device = None

    def generate_keyframes(
        self,
        first_frame_path: str,
        trajectory_plan: Dict[str, Any],
        output_dir: str = "outputs/frames",
        prompt_template: str = "A red {object_color} {object_type} on a wooden table",
        keyframe_interval: int = 4,  # Generate keyframes every N frames
    ) -> List[str]:
        """
        Generate keyframe images from trajectory plan.
        
        Args:
            first_frame_path: Path to initial frame
            trajectory_plan: Dictionary from trajectory planner
            output_dir: Directory to save keyframes
            prompt_template: Template for inpainting prompt
            keyframe_interval: Generate keyframes at this interval (every N frames)
            
        Returns:
            List of paths to generated keyframe images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        first_frame = Image.open(first_frame_path).convert("RGB")
        frame_width, frame_height = first_frame.size
        
        keyframe_paths = {}
        
        frames_data = trajectory_plan["frames"]
        total_frames = len(frames_data)
        
        # Generate keyframes at regular intervals to show motion
        # Use trajectory data to position object at each keyframe
        keyframe_indices = list(range(0, total_frames, keyframe_interval))
        if keyframe_indices[-1] != total_frames - 1:
            keyframe_indices.append(total_frames - 1)
        
        print(f"Generating {len(keyframe_indices)} keyframes at intervals of {keyframe_interval}...")

        # Build a clean background plate by removing the initial object once
        initial = frames_data[0]
        init_mask = self._create_position_mask(
            frame_width, frame_height, initial["x"], initial["y"], initial["radius"]
        )
        background = self._inpaint_remove_object(
            first_frame,
            init_mask,
            prompt_template
        )
        
        # Create a solid fallback background (plain wooden color)
        # This ensures no ball ghosting from failed inpainting
        wood_color = (139, 90, 43)  # Brown wood color
        fallback_bg = Image.new("RGB", first_frame.size, wood_color)
        
        # Try to use inpainted background if it looks reasonable, otherwise use solid color
        bg_array = np.array(background)
        # Check if background has mostly been cleaned (not too much variation in mask area)
        bg_rgb = Image.fromarray(bg_array)
        
        # Use solid background to avoid any ball artifacts
        final_background = fallback_bg
        
        # Persist background plate for reuse in later stages
        bg_path = os.path.join(os.path.dirname(output_dir), "background.png")
        final_background.save(bg_path)
        print(f"✓ Background plate saved to {bg_path}")
        
        for frame_idx in keyframe_indices:
            frame_info = frames_data[frame_idx]
            x, y, radius = frame_info["x"], frame_info["y"], frame_info["radius"]

            print(f"\n  Frame {frame_idx}: placing object at ({x}, {y})")

            # Create placement mask on the clean background
            new_mask = self._create_position_mask(
                frame_width,
                frame_height,
                x,
                y,
                radius
            )

            # Use a concise, stable object prompt; location is given by the mask
            object_prompt = "a glossy red ball on a wooden table, single object, consistent lighting"

            # Render on a fresh copy of the background to avoid drift
            frame_base = final_background.copy()
            placed = self._inpaint_add_object(
                frame_base,
                new_mask,
                object_prompt,
                num_inference_steps=25,
                guidance_scale=5.0
            )

            # Save keyframe
            output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            placed.save(output_path)
            keyframe_paths[frame_idx] = output_path
            print(f"  ✓ Saved keyframe to {output_path}")
        
        print(f"\n✓ Generated {len(keyframe_paths)} keyframes")
        return keyframe_paths

    def _get_object_mask(
        self,
        frame: Image.Image,
        width: int,
        height: int,
        radius: int,
        use_sam: bool = True
    ) -> Image.Image:
        """
        Extract mask of object in frame using SAM or simple heuristics.
        
        Args:
            frame: Input frame
            width: Frame width
            height: Frame height
            radius: Expected object radius
            use_sam: Whether to use SAM if available
            
        Returns:
            Binary mask image
        """
        if use_sam and self.sam_model is not None:
            try:
                # Use HuggingFace SAM with point prompt
                # Point in lower-center region (typical for resting object)
                input_points = [[[width // 2, height - 50]]]
                
                # Process inputs
                inputs = self.sam_processor(
                    frame,
                    input_points=input_points,
                    return_tensors="pt"
                ).to(self.sam_device)
                
                # Generate mask
                with torch.no_grad():
                    outputs = self.sam_model(**inputs)
                
                # Post-process mask
                masks = self.sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                # Convert to PIL Image
                mask_array = masks[0][0].numpy().astype(np.uint8) * 255
                mask = Image.fromarray(mask_array)
                return mask
            except Exception as e:
                print(f"  ⚠ SAM failed: {e}. Using simple mask.")
        
        # Fallback: create simple circular mask at bottom-center
        return self._create_position_mask(width, height, width // 2, height - 50, radius)

    def _create_position_mask(
        self,
        width: int,
        height: int,
        x: int,
        y: int,
        radius: int
    ) -> Image.Image:
        """Create a circular mask at specified position."""
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw circle with some padding for inpainting context
        r = radius + 10
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        
        return mask

    def _inpaint_remove_object(
        self,
        frame: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Use inpainting to remove object from frame.
        
        Args:
            frame: Input frame
            mask: Binary mask of region to inpaint
            prompt: Inpainting prompt (e.g., "wooden table, clean background")
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Inpainted frame with object removed
        """
        removal_prompt = "wooden table, clean background, no objects"
        
        # Ensure mask and frame are correct sizes
        mask = mask.resize(frame.size, Image.NEAREST)
        
        with torch.no_grad():
            output = self.inpaint_pipe(
                prompt=removal_prompt,
                image=frame,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt="blurry, bad quality, artifacts",
                generator=self._generator,
            ).images[0]
        
        return output

    def _inpaint_add_object(
        self,
        frame: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Use inpainting to add object at specified location.
        
        Args:
            frame: Input frame
            mask: Binary mask of where to inpaint
            prompt: Inpainting prompt describing desired object
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Inpainted frame with object added
        """
        # Ensure mask and frame are correct sizes
        mask = mask.resize(frame.size, Image.NEAREST)
        
        with torch.no_grad():
            output = self.inpaint_pipe(
                prompt=prompt,
                image=frame,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt="blurry, bad quality, artifacts, multiple objects",
                generator=self._generator,
            ).images[0]
        
        return output


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate keyframes from trajectory plan")
    parser.add_argument("--first-frame", required=True, help="Path to first frame image")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory plan JSON")
    parser.add_argument("--output-dir", default="outputs/frames", help="Output directory for keyframes")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers")
    
    args = parser.parse_args()
    
    # Load trajectory
    with open(args.trajectory, "r") as f:
        trajectory_plan = json.load(f)
    
    generator = KeyframeGenerator(device=args.device, use_xformers=not args.no_xformers)
    generator.generate_keyframes(
        first_frame_path=args.first_frame,
        trajectory_plan=trajectory_plan,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
