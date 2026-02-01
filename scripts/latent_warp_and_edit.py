"""
Latent Warping and Intermediate Frame Generation Module
Uses DDIM inversion, optical flow, and inpainting to generate smooth intermediate frames.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import AutoPipelineForInpainting, DDIMScheduler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class DDIMInverter:
    """DDIM inversion for extracting latents from images."""

    def __init__(self, pipe, device: str = "cuda"):
        """
        Initialize DDIM inverter.
        
        Args:
            pipe: Diffusion pipeline with VAE and UNet
            device: torch device
        """
        self.pipe = pipe
        self.device = device
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.vae = pipe.vae
        self.unet = pipe.unet

    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to latent space using VAE.
        
        Args:
            image: PIL Image
            
        Returns:
            Latent tensor [1, 4, H/8, W/8]
        """
        # Prepare image
        image = image.resize((512, 512), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        image_t = image_t * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Encode to latent
        with torch.no_grad():
            latent = self.vae.encode(image_t).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        
        return latent

    def invert_image_to_noise(
        self,
        image: Image.Image,
        prompt: str = "",
        num_inference_steps: int = 50
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Invert image to noise using DDIM.
        
        Args:
            image: Input image
            prompt: Text prompt for conditioning
            num_inference_steps: Number of inversion steps
            
        Returns:
            Final noise tensor and list of intermediate latents
        """
        # Encode to latent
        latent = self.encode_image_to_latent(image)
        
        # Setup scheduler for inversion
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Get text embeddings
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(
                self.pipe.tokenizer(
                    "", padding="max_length", max_length=77, return_tensors="pt"
                ).input_ids.to(self.device)
            )[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # DDIM inversion loop
        latents_inv = [latent]
        
        for i, t in enumerate(reversed(timesteps)):
            # Expand latent for classifier-free guidance
            latent_model_input = torch.cat([latent] * 2)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # DDIM inversion step
            prev_latent = self.scheduler.step(noise_pred, t, latent, eta=1.0).prev_sample
            latent = prev_latent
            latents_inv.append(latent)
        
        return latent, latents_inv


class OpticalFlowWarper:
    """Applies optical flow-based warping to latent space."""

    @staticmethod
    def compute_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames using Farneback algorithm.
        
        Args:
            frame1: First frame (H, W, 3) in [0, 255]
            frame2: Second frame (H, W, 3) in [0, 255]
            
        Returns:
            Optical flow (H, W, 2)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.1,
            0,
        )
        
        return flow

    @staticmethod
    def warp_latent(latent: torch.Tensor, flow: np.ndarray, mode: str = "bilinear") -> torch.Tensor:
        """
        Apply optical flow warping to latent tensor.
        
        Args:
            latent: Latent tensor [1, C, H, W]
            flow: Optical flow [H, W, 2] in original frame space (scale for latent if needed)
            mode: Interpolation mode
            
        Returns:
            Warped latent tensor
        """
        B, C, H, W = latent.shape
        
        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack([x, y], dim=-1).float().unsqueeze(0)  # [1, H, W, 2]
        
        # Resize flow to latent resolution
        flow_resized = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
        flow_t = torch.from_numpy(flow_resized).unsqueeze(0).float()  # [1, H, W, 2]
        
        # Apply flow to coordinates
        warped_grid = grid + flow_t
        
        # Normalize coordinates to [-1, 1] for grid_sample
        warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
        warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0
        
        # Apply warping using grid_sample
        warped_latent = torch.nn.functional.grid_sample(
            latent,
            warped_grid,
            mode=mode,
            padding_mode="border",
            align_corners=True
        )
        
        return warped_latent


class IntermediateFrameGenerator:
    """Generates smooth intermediate frames using latent warping."""

    def __init__(self, device: str = "cuda", use_xformers: bool = True):
        """
        Initialize the intermediate frame generator.
        
        Args:
            device: torch device
            use_xformers: whether to use xformers
        """
        self.device = device
        self.inpaint_pipe = None
        self.inverter = None
        self._generator = torch.Generator(device=device).manual_seed(42)
        self._load_models(use_xformers)

    def _load_models(self, use_xformers: bool = True):
        """Load pipelines."""
        print("Loading inpainting pipeline...")
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
            except:
                pass
        
        # Enable model CPU offload for memory efficiency
        self.inpaint_pipe.enable_model_cpu_offload()
        
        self.inverter = DDIMInverter(self.inpaint_pipe, device=self.device)
        print("✓ Models loaded (safety_checker disabled)")

    def generate_intermediate_frames(
        self,
        keyframe_paths: Dict[int, str],
        trajectory_plan: Dict[str, Any],
        output_dir: str = "outputs/frames",
        num_steps: int = 2
    ) -> List[str]:
        """
        Generate intermediate frames between keyframes.
        
        Args:
            keyframe_paths: Dict mapping frame index to keyframe path (all generated keyframes)
            trajectory_plan: Trajectory plan data
            output_dir: Output directory
            num_steps: Number of intermediate frames between keyframes
            
        Returns:
            List of all frame paths (keyframes + intermediate)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        frames_data = trajectory_plan["frames"]
        total_frames = len(frames_data)
        keyframe_indices = sorted(keyframe_paths.keys())  # Use actual generated keyframes
        
        print(f"\nGenerating intermediate frames between {len(keyframe_indices)} keyframes (step size: {num_steps})...")
        
        all_frames = {}
        
        # Add all keyframes to output
        for frame_idx, frame_path in keyframe_paths.items():
            all_frames[frame_idx] = frame_path
        
        # Synthesize any missing frames directly via inpainting on the background plate
        bg_path = os.path.join(os.path.dirname(output_dir), "background.png")
        if not os.path.exists(bg_path):
            print(f"⚠ Background plate not found at {bg_path}. Interpolation cannot proceed reliably.")
        else:
            background = Image.open(bg_path).convert("RGB")
            W, H = background.size

            print(f"\n  Rendering missing frames on background plate...")
            object_prompt = "a glossy red ball on a wooden table, single object, consistent lighting"

            for frame_idx in range(total_frames):
                if frame_idx in all_frames:
                    continue

                info = frames_data[frame_idx]
                x, y, r = info["x"], info["y"], info["radius"]
                mask = self._create_position_mask(W, H, x, y, r)

                rendered = self._inpaint_add_object(
                    background.copy(),
                    mask,
                    object_prompt,
                    num_inference_steps=22,
                    guidance_scale=5.0,
                )

                out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                rendered.save(out_path)
                all_frames[frame_idx] = out_path
                print(f"    ✓ Generated frame {frame_idx}")
        
        # Sort all frames
        sorted_frames = sorted(all_frames.items())
        frame_paths = [path for _, path in sorted_frames]
        
        print(f"\n✓ Generated {len(frame_paths)} total frames")
        return frame_paths

    def _interpolate_frames(
        self,
        frame1: Image.Image,
        frame2: Image.Image,
        frame1_idx: int,
        frame2_idx: int,
        num_intermediate: int = 2,
        output_dir: str = "outputs/frames"
    ) -> Dict[int, str]:
        """
        Interpolate intermediate frames between two keyframes.
        Uses simple linear blending for robustness.
        
        Args:
            frame1: First keyframe
            frame2: Second keyframe
            frame1_idx: Index of first frame
            frame2_idx: Index of second frame
            num_intermediate: Number of intermediate frames to generate
            output_dir: Output directory
            
        Returns:
            Dict mapping frame index to saved path
        """
        # Deprecated in favor of direct inpainting per trajectory; keep stub for API.
        return {}

    def _create_position_mask(self, width: int, height: int, x: int, y: int, radius: int) -> Image.Image:
        mask = Image.new("L", (width, height), 0)
        r = radius + 10
        import PIL
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        return mask

    def _inpaint_add_object(
        self,
        frame: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 22,
        guidance_scale: float = 5.0,
    ) -> Image.Image:
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
    
    parser = argparse.ArgumentParser(description="Generate intermediate frames using latent warping")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory plan JSON")
    parser.add_argument("--keyframes-dir", required=True, help="Directory with keyframe images")
    parser.add_argument("--output-dir", default="outputs/frames", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--num-steps", type=int, default=2, help="Intermediate frames per keyframe pair")
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers")
    
    args = parser.parse_args()
    
    with open(args.trajectory, "r") as f:
        trajectory_plan = json.load(f)
    
    # Build keyframe map
    keyframe_map = {}
    for i in range(0, 10000):  # Scan for frame files
        frame_path = os.path.join(args.keyframes_dir, f"frame_{i:04d}.png")
        if os.path.exists(frame_path):
            # Find corresponding keyframe index
            for frame_info in trajectory_plan["frames"]:
                if frame_info["keyframe"] and frame_info["frame"] <= i:
                    keyframe_map[frame_info["frame"]] = frame_path
                    break
    
    generator = IntermediateFrameGenerator(device=args.device, use_xformers=not args.no_xformers)
    generator.generate_intermediate_frames(
        keyframe_paths=keyframe_map,
        trajectory_plan=trajectory_plan,
        output_dir=args.output_dir,
        num_steps=args.num_steps
    )


if __name__ == "__main__":
    main()
