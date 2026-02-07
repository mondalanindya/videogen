#!/usr/bin/env python3
"""
Solution A: Clean Slate Generation
Generate background without ball, then paste synthetic red balls at trajectory positions.
This bypasses all detection issues and guarantees correct positioning.
"""

import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

def create_red_ball_patch(radius=28, with_shading=True):
    """
    Create a synthetic red ball patch with optional 3D shading.
    Returns: (patch_bgr, mask_alpha) where mask is 0-255 grayscale
    """
    size = int(radius * 2.2)  # Extra space for anti-aliasing
    center = size // 2
    
    # Create base images
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    if with_shading:
        # Create 3D shaded ball
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= radius:
                    # Circular mask
                    alpha = 255
                    if dist > radius - 2:  # Anti-aliasing edge
                        alpha = int(255 * (radius - dist) / 2)
                    mask[y, x] = alpha
                    
                    # 3D shading: light from top-left
                    light_x = center - radius * 0.4
                    light_y = center - radius * 0.4
                    light_dx = x - light_x
                    light_dy = y - light_y
                    light_dist = np.sqrt(light_dx*light_dx + light_dy*light_dy)
                    
                    # Calculate surface normal (sphere)
                    if dist < radius:
                        z = np.sqrt(radius*radius - dist*dist)
                        nx = dx / radius
                        ny = dy / radius
                        nz = z / radius
                        
                        # Light direction (normalized)
                        light_vec = np.array([light_dx, light_dy, radius * 0.5])
                        light_vec = light_vec / np.linalg.norm(light_vec)
                        
                        # Lambertian shading
                        diffuse = max(0, -nx * light_vec[0] - ny * light_vec[1] + nz * light_vec[2])
                        diffuse = diffuse * 0.7 + 0.3  # Ambient + diffuse
                        
                        # Apply to red channel (BGR format)
                        patch[y, x] = [0, 0, int(255 * diffuse)]
                    else:
                        patch[y, x] = [0, 0, 255]
    else:
        # Flat red circle
        cv2.circle(patch, (center, center), radius, (0, 0, 255), -1)
        cv2.circle(mask, (center, center), radius, 255, -1)
        # Anti-aliasing edge
        cv2.circle(patch, (center, center), radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(mask, (center, center), radius, 255, 2, cv2.LINE_AA)
    
    return patch, mask


def generate_clean_background(prompt, negative_prompt, seed=42, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Generate empty wooden table background without any ball.
    """
    print(f"Loading Stable Diffusion model: {model_id}")
    
    # Use float32 for CPU, float16 for CUDA
    dtype = torch.float32 if device == "cpu" else torch.float16
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Generating background with prompt: '{prompt}'")
    print(f"Negative prompt: '{negative_prompt}'")
    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=8.0,
        generator=generator,
        height=512,
        width=512
    )
    
    return output.images[0]


def paste_ball_on_frame(background_bgr, ball_patch, ball_mask, center_x, center_y):
    """
    Paste ball patch onto background at specified center position.
    Uses alpha blending for smooth edges.
    """
    frame = background_bgr.copy()
    ph, pw = ball_patch.shape[:2]
    
    # Calculate top-left position
    tlx = int(center_x - pw // 2)
    tly = int(center_y - ph // 2)
    
    # Clip to image boundaries
    H, W = frame.shape[:2]
    
    # Source region (from patch)
    src_x1 = max(0, -tlx)
    src_y1 = max(0, -tly)
    src_x2 = min(pw, W - tlx)
    src_y2 = min(ph, H - tly)
    
    # Destination region (on frame)
    dst_x1 = max(0, tlx)
    dst_y1 = max(0, tly)
    dst_x2 = min(W, tlx + pw)
    dst_y2 = min(H, tly + ph)
    
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        print(f"Warning: Ball completely outside frame at ({center_x}, {center_y})")
        return frame
    
    # Extract regions
    ball_region = ball_patch[src_y1:src_y2, src_x1:src_x2]
    mask_region = ball_mask[src_y1:src_y2, src_x1:src_x2]
    bg_region = frame[dst_y1:dst_y2, dst_x1:dst_x2]
    
    # Alpha blend
    alpha = mask_region.astype(float) / 255.0
    alpha_3ch = np.stack([alpha] * 3, axis=2)
    
    blended = (ball_region.astype(float) * alpha_3ch + 
               bg_region.astype(float) * (1 - alpha_3ch)).astype(np.uint8)
    
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended
    
    return frame


def validate_positions(frames_dir, trajectory_data, num_samples=10):
    """
    Validate that balls are at correct positions in generated frames.
    """
    print("\n" + "="*70)
    print("POSITION VALIDATION")
    print("="*70)
    
    frames = trajectory_data['frames']
    sample_indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
    
    errors = []
    
    print(f"\n{'Frame':<8} | {'Expected Y':<12} | {'Actual Y':<12} | {'Error (px)':<12}")
    print("-" * 60)
    
    for idx in sample_indices:
        frame_data = frames[idx]
        frame_num = frame_data['frame']
        expected_y = frame_data['y']
        
        frame_path = os.path.join(frames_dir, f"keyframe_{frame_num:04d}.png")
        if not os.path.exists(frame_path):
            print(f"{frame_num:<8} | MISSING")
            continue
        
        # Detect red pixels
        img = cv2.imread(frame_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        ys, xs = np.where(red_mask > 0)
        
        if len(ys) > 0:
            actual_y = ys.mean()
            error = abs(actual_y - expected_y)
            errors.append(error)
            
            status = "✓" if error < 5 else "✗"
            print(f"{frame_num:<8} | {expected_y:<12.1f} | {actual_y:<12.1f} | {error:<12.1f} {status}")
        else:
            print(f"{frame_num:<8} | {expected_y:<12.1f} | NO RED DETECTED")
    
    if errors:
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        print("\n" + "-" * 60)
        print(f"Average position error: {avg_error:.2f} pixels")
        print(f"Maximum position error: {max_error:.2f} pixels")
        
        if avg_error < 5:
            print("✓ VALIDATION PASSED: Ball positions accurate")
        else:
            print("✗ VALIDATION FAILED: Position errors too large")
    
    print("="*70 + "\n")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trajectory
    print(f"Loading trajectory from: {args.trajectory}")
    with open(args.trajectory, 'r') as f:
        trajectory_data = json.load(f)
    
    frames = trajectory_data['frames']
    print(f"Trajectory contains {len(frames)} frames")
    
    # Step 1: Generate clean background
    background_path = os.path.join(args.output_dir, "background_clean.png")
    
    if os.path.exists(background_path) and not args.regenerate_bg:
        print(f"Loading existing background from {background_path}")
        background_pil = Image.open(background_path)
    else:
        background_pil = generate_clean_background(
            prompt=args.bg_prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            model_id=args.model_id,
            device=args.device
        )
        background_pil.save(background_path)
        print(f"Background saved to {background_path}")
    
    background_bgr = cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2BGR)
    
    # Verify background has no red pixels
    hsv_bg = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_bg, np.array([0, 120, 120]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_bg, np.array([170, 120, 120]), np.array([180, 255, 255]))
    red_bg = cv2.bitwise_or(mask1, mask2)
    red_count = np.count_nonzero(red_bg)
    
    if red_count > 1000:
        print(f"⚠ WARNING: Background contains {red_count} red pixels!")
        print("  Consider regenerating with stronger negative prompt")
    else:
        print(f"✓ Background clean: {red_count} red pixels (acceptable)")
    
    # Step 2: Create ball patch
    ball_patch_path = os.path.join(args.output_dir, "ball_patch.png")
    ball_mask_path = os.path.join(args.output_dir, "ball_mask.png")
    
    if os.path.exists(ball_patch_path) and not args.regenerate_ball:
        print(f"Loading existing ball patch from {ball_patch_path}")
        ball_patch = cv2.imread(ball_patch_path)
        ball_mask = cv2.imread(ball_mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Use radius from first frame
        radius = frames[0].get('radius', 28)
        print(f"Creating ball patch with radius={radius}, shading={not args.no_shading}")
        
        ball_patch, ball_mask = create_red_ball_patch(
            radius=radius,
            with_shading=not args.no_shading
        )
        
        cv2.imwrite(ball_patch_path, ball_patch)
        cv2.imwrite(ball_mask_path, ball_mask)
        print(f"Ball patch saved: {ball_patch_path}")
        print(f"Ball mask saved: {ball_mask_path}")
    
    print(f"Ball patch size: {ball_patch.shape[1]}×{ball_patch.shape[0]}")
    
    # Step 3: Generate all frames
    print(f"\nGenerating {len(frames)} keyframes...")
    
    for i, frame_data in enumerate(frames):
        frame_num = frame_data['frame']
        cx = int(frame_data['x'])
        cy = int(frame_data['y'])
        
        output_path = os.path.join(args.output_dir, f"keyframe_{frame_num:04d}.png")
        
        if os.path.exists(output_path) and not args.force:
            if i % 20 == 0:
                print(f"  Frame {frame_num:04d} exists, skipping... ({i+1}/{len(frames)})")
            continue
        
        # Paste ball on background
        frame = paste_ball_on_frame(background_bgr, ball_patch, ball_mask, cx, cy)
        
        cv2.imwrite(output_path, frame)
        
        if i % 20 == 0 or i == len(frames) - 1:
            print(f"  Generated keyframe {frame_num:04d} at position ({cx}, {cy}) - {i+1}/{len(frames)}")
    
    print(f"\n✓ All {len(frames)} keyframes generated in: {args.output_dir}")
    
    # Step 4: Validate positions
    if args.validate:
        validate_positions(args.output_dir, trajectory_data, num_samples=args.validation_samples)
    
    print("\nNext steps:")
    print(f"  1. View collage: python3 -c 'exec(open(\"create_verification_collage.py\").read())'")
    print(f"  2. Assemble video:")
    print(f"     python3 scripts/assemble_keyframes_direct.py \\")
    print(f"       --frames-dir {args.output_dir} \\")
    print(f"       --output outputs/ball_bounce_clean.mp4 \\")
    print(f"       --fps 24")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solution A: Clean Slate Generation")
    
    # Input/Output
    parser.add_argument("--trajectory", default="outputs/trajectory_plan.json",
                        help="Path to trajectory JSON file")
    parser.add_argument("--output-dir", default="outputs/ip_keyframes_clean",
                        help="Output directory for keyframes")
    
    # Background generation
    parser.add_argument("--bg-prompt", default="Empty wooden table, realistic lighting, photorealistic, high quality",
                        help="Prompt for background generation")
    parser.add_argument("--negative-prompt", 
                        default="ball, sphere, red object, any objects, toys, cluttered, blurry, low quality",
                        help="Negative prompt to avoid objects in background")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda",
                        help="Device to run on (cuda/cpu)")
    
    # Ball appearance
    parser.add_argument("--no-shading", action="store_true",
                        help="Use flat red circle instead of 3D-shaded ball")
    
    # Regeneration flags
    parser.add_argument("--regenerate-bg", action="store_true",
                        help="Force regenerate background even if exists")
    parser.add_argument("--regenerate-ball", action="store_true",
                        help="Force regenerate ball patch even if exists")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing keyframes")
    
    # Validation
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate ball positions after generation")
    parser.add_argument("--validation-samples", type=int, default=10,
                        help="Number of frames to sample for validation")
    
    args = parser.parse_args()
    main(args)
