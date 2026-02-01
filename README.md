# Video Generation Pipeline

An end-to-end framework for generating videos from text prompts by planning object trajectories and synthesizing motion through iterative image editing.

## Overview

This pipeline transforms a simple prompt like **"A red ball bounces five times on a wooden table"** into a coherent video by:

1. **Planning** object trajectories using a Vision-Language Model (Qwen-VL)
2. **Generating** keyframes using SAM segmentation + Stable Diffusion inpainting
3. **Interpolating** smooth intermediate frames via latent warping
4. **Assembling** frames into a final MP4 video

## Features

✨ **Vision-Language Planning**: Uses Qwen-VL to understand complex motion descriptions
🎨 **SAM-Based Segmentation**: Intelligent object extraction without manual masks
🖼️ **Diffusion-Based Synthesis**: Stable Diffusion inpainting for realistic frame generation
⚡ **Latent Warping**: Optical flow + latent space manipulation for smooth transitions
📹 **Multi-Format Output**: MP4 video + frame collages for quick preview

## Quick Start

### Installation
```bash
# 1. Install PyTorch (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Basic Usage
```bash
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A red ball bounces five times on a wooden table"
```

Output files:
- `outputs/final_video.mp4` - Generated video
- `outputs/trajectory_plan.json` - Planned positions
- `outputs/frames/` - Individual video frames
- `outputs/preview_collage.png` - Visual preview

## Pipeline Components

### 1. Trajectory Planner (`plan_trajectory.py`)
Generates frame-by-frame object positions using Qwen-VL

**Output Format:**
```json
{
  "frames": [
    {
      "frame": 0,
      "x": 100,
      "y": 300,
      "radius": 20,
      "angle": 0,
      "keyframe": true,
      "confidence": 1.0
    },
    ...
  ]
}
```

**Key Features:**
- Understands motion descriptions (bouncing, rolling, rotating)
- Identifies keyframe events (bounces, direction changes)
- Fallback synthetic generation for offline testing

### 2. Keyframe Generator (`generate_keyframes.py`)
Creates realistic keyframe images using inpainting

**Process:**
1. Use SAM to segment object in source frame
2. Inpaint background (remove object)
3. Inpaint new position with refined prompt
4. Save keyframe

**Key Features:**
- Adaptive object detection via SAM
- Background consistency via inpainting
- Configurable prompt templates

### 3. Latent Warping (`latent_warp_and_edit.py`)
Generates smooth intermediate frames

**Process:**
1. DDIM inversion to extract image latents
2. Optical flow computation between keyframes
3. Latent-space warping using flow
4. Masked inpainting for final refinement

**Key Features:**
- DDIM inversion for precise image reconstruction
- Farneback optical flow for motion estimation
- Multi-step blending for smooth transitions

### 4. Video Assembler (`assemble_video.py`)
Compiles frames into final video

**Features:**
- OpenCV VideoWriter for robust encoding
- Resolution auto-detection
- Preview collage generation
- Configurable FPS

## Advanced Usage

### Full Control
```bash
python pipeline.py \
  --first-frame inputs/ball.png \
  --prompt "A blue sphere bounces in a circle" \
  --num-frames 32 \           # Total frames
  --intermediate-steps 3 \     # Frames between keyframes
  --fps 24 \                  # Output video fps
  --output-dir outputs_v2 \
  --device cuda
```

### Testing Individual Components
```bash
# Test trajectory planning
python scripts/plan_trajectory.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball bounces" \
  --num-frames 16

# Test keyframe generation
python scripts/generate_keyframes.py \
  --first-frame inputs/first_frame.png \
  --trajectory trajectory_plan.json

# Test video assembly
python scripts/assemble_video.py \
  --frames-dir outputs/frames \
  --fps 15 \
  --preview
```

### Skipping Steps (for Iteration)
```bash
# Reuse previous trajectory and keyframes
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "..." \
  --skip-steps trajectory keyframes
```

## Project Structure

```
videogen/
├── pipeline.py                 # Main orchestrator
├── requirements.txt            # Dependencies
├── INSTALL.md                  # Detailed setup guide
├── README.md                   # This file
│
├── scripts/
│   ├── plan_trajectory.py      # VLM trajectory planning
│   ├── generate_keyframes.py   # SAM + inpainting
│   ├── latent_warp_and_edit.py # Latent warping & interpolation
│   └── assemble_video.py       # Video assembly
│
├── inputs/                     # Input images and prompts
├── outputs/                    # Generated videos
│   ├── frames/                # Individual frames
│   ├── trajectory_plan.json   # Planned trajectory
│   ├── final_video.mp4        # Output video
│   └── preview_collage.png    # Frame preview
└── sam_vit_b.pth              # SAM checkpoint (auto-downloaded)
```

## Models Used

| Component | Model | Source |
|-----------|-------|--------|
| **Vision-Language** | Qwen/Qwen2.5-VL-7B-Instruct | HuggingFace Hub |
| **Image Inpainting** | stable-diffusion-v1-5/stable-diffusion-inpainting | HuggingFace Hub |
| **Segmentation** | SAM ViT-B | Facebook Research |
| **Optical Flow** | Farneback (OpenCV) | OpenCV |

## Performance Benchmarks

On V100 (32GB VRAM):
- **Trajectory Planning**: 1-2 min
- **Per Keyframe**: 2-3 min
- **Per Interpolated Frame**: 30-45 sec
- **Video Assembly**: 10-20 sec
- **Total** (16 frames, 2 intermediate): ~20-30 min

## Sanity Checks

The pipeline includes built-in validation:
- ✓ Frame count verification
- ✓ Position consistency checking
- ✓ Mask coverage validation
- ✓ Optical flow quality assessment
- ✓ Video framerate verification

Run tests:
```bash
python -m pytest tests/  # (if tests are added)
```

## Troubleshooting

### Out of Memory
```python
# Reduce frames or use CPU
python pipeline.py ... --device cpu  # Much slower
```

### SAM Download Issues
```bash
# Pre-download checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth
```

### Slow Performance
- Check GPU is being used: `nvidia-smi`
- Verify xformers installation: `python -c "import xformers"`
- Reduce `num-frames` or `intermediate-steps`

## Configuration

Key hyperparameters in code:

**Trajectory Planning:**
- `num_frames`: 16 (adjustable)
- `inference_steps`: 30 for inpainting

**Keyframe Generation:**
- `num_inference_steps`: 30
- `guidance_scale`: 7.5

**Video Assembly:**
- `fps`: 15 (adjustable)
- `codec`: "mp4v" (H.264)

## Limitations & Future Work

### Current Limitations
- Single object focus (multi-object support TODO)
- Static background (dynamic backgrounds TODO)
- 512x512 resolution (higher res support planned)
- Simple linear interpolation (advanced motion synthesis planned)

### Future Enhancements
- [ ] Multi-object trajectory planning
- [ ] Dynamic background synthesis
- [ ] Higher resolution support (1024x1024+)
- [ ] Cross-frame attention for better coherence
- [ ] 3D-aware motion synthesis
- [ ] Text2Video-Zero integration
- [ ] Real-time preview generation

## Citation

If you use this pipeline, please cite:

```bibtex
@software{videogen2024,
  title={Video Generation Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/...}
}
```

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request

## Support

For issues and questions:
- Check INSTALL.md for setup help
- Review example usages above
- Open an issue on GitHub

## References

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Diffusers](https://github.com/huggingface/diffusers)
- [DDIM](https://arxiv.org/abs/2010.02502)

---

**Made with ❤️ for video synthesis**
