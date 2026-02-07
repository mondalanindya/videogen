# Clean Slate Video Generation - Ball Bounce Solution

## Overview
This is a working solution for generating synthetic video of a bouncing ball using the **Clean Slate approach**. All non-working code and experimental pipelines have been removed.

## What This Does
Generates a 160-frame video (6.67 seconds @ 24 FPS) of a red ball bouncing vertically on a wooden table:
- **Ball motion**: y-position 94→386 pixels (smooth vertical bounce)
- **Background**: Generated wooden table (clean, no artifacts)
- **Method**: Stable Diffusion v1.5 (CPU) + synthetic ball compositing

## Quick Start

### Generate Video
```bash
sbatch run_clean_slate.sub
```

Or run directly:
```bash
python3 generate_keyframes_clean_slate.py \
  --trajectory outputs/trajectory_plan.json \
  --output-dir outputs/ip_keyframes_clean \
  --validate
```

### Outputs
- `outputs/ball_bounce_clean.mp4` - Final video (770 KB)
- `outputs/collage_verification.jpg` - Frame grid (926 KB)
- `outputs/ip_keyframes_clean/` - All 160 frames as PNG files

## Core Scripts

### `generate_keyframes_clean_slate.py` (366 lines)
Main implementation:
- `generate_clean_background()` - Creates empty wooden table (lines 82-103)
- `create_red_ball_patch()` - Synthesizes red ball with 3D shading (lines 15-79)
- `paste_ball_on_frame()` - Alpha-blends ball onto background (lines 137-187)
- `validate_positions()` - Verifies ball placement accuracy (lines 191-228)

**Key fix**: Uses `torch.float32` for CPU compatibility (no CUDA)

### `create_verification_collage.py`
Creates 3×3 grid visualization of sample frames for quick inspection.

### `run_clean_slate.sub`
SLURM batch script:
- Partition: `workq` (CPU-only, 32GB RAM)
- Time: 1 hour
- Output: `logs/clean_slate_<jobid>.out`

## Results
✅ **Success Metrics**:
- Ball moves correctly (y: 94→386)
- All 160 frames generated
- Clean background (0 red artifacts)
- Video assembled (770KB, 6.67 sec)
- Position accuracy: 98% (5.3px error vs 293px range)

## Technical Details

### Ball Creation
- Size: 61×61 pixels (radius 28)
- Color: Pure red (BGR: 0, 0, 255)
- Shading: 3D Lambertian with top-left light source
- Alpha blending: Smooth edges on wooden background

### Background Generation
- Model: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- Device: CPU (torch.float32 dtype)
- Prompt: "Empty wooden table, realistic lighting, photorealistic"
- Negative: "ball, sphere, red object, any objects"
- Result: Clean table with 0 red pixels

### Trajectory
- Source: `outputs/trajectory_plan.json` (160 keyframes)
- Motion: Parabolic bounce (y: 99→392 planned, 94→386 actual)
- X-position: Centered at x=251

## Helper Scripts (in `scripts/`)
- `plan_trajectory.py` - Generate trajectory plans
- `assemble_video.py` - Assemble frames into video
- `add_shadow.py` - Add drop shadow for depth perception
- `create_collage.py` - Create visualization grids

## Dependencies
```
opencv-python==4.8.1.78
diffusers==0.24.0
torch==2.1.2
transformers==4.35.2
numpy>=1.24.0
Pillow>=10.0.0
```

## Notes
- Clean background is generated once, then ball is pasted 160 times
- This avoids the spatial mismatch issues of previous pipelines
- Position error (5.3px) is due to ball patch centering; impact is cosmetic
- No expensive diffusion steps for each frame (generation is ~11 min for full run)

## Future Enhancements
1. **Realistic texture** - Extract ball texture from SD generation
2. **Complex motion** - Circular, diagonal, or parabolic trajectories
3. **Physics validation** - Verify energy conservation, contact timing
4. **Multi-object** - Ball-ball collision, ball-paddle interaction

---
**Last Updated**: Feb 7, 2026  
**Status**: ✅ Working
