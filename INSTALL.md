# Installation and Setup Guide

## Prerequisites
- Python 3.10+
- CUDA 11.8 or 12.1 (for GPU acceleration)
- 24+ GB VRAM recommended (for large diffusion models)

## Installation Steps

### 1. Clone or Setup Repository
```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/videogen
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch
Choose the version appropriate for your system:

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio
```

### 4. Install Other Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Qwen VL Utilities (Optional but Recommended)
For faster video/image loading with Qwen models:
```bash
pip install qwen-vl-utils[decord]==0.0.8
```

### 6. Install Segment Anything (SAM)
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint (optional - will auto-download on first use)
# Place in project root as 'sam_vit_b.pth'
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth
```

### 7. Optional: Install Optical Flow Dependencies
For advanced optical flow capabilities:
```bash
pip install flowiz
```

## Quick Start

### Basic Usage
```bash
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A red ball bounces five times on a wooden table"
```

### Advanced Usage with Options
```bash
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A blue cube rolls across the floor" \
  --num-frames 32 \
  --intermediate-steps 3 \
  --fps 24 \
  --output-dir outputs_custom \
  --device cuda
```

### Test Individual Components
```bash
# Generate trajectory only
python scripts/plan_trajectory.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball bounces" \
  --output trajectory_plan.json

# Generate keyframes
python scripts/generate_keyframes.py \
  --first-frame inputs/first_frame.png \
  --trajectory trajectory_plan.json \
  --output-dir outputs/frames

# Assemble video
python scripts/assemble_video.py \
  --frames-dir outputs/frames \
  --output outputs/final_video.mp4 \
  --fps 15 \
  --preview
```

## Project Structure
```
videogen/
├── pipeline.py                    # Main orchestrator
├── requirements.txt               # Dependencies
├── INSTALL.md                     # This file
├── inputs/                        # Input images and prompts
├── outputs/                       # Generated videos and frames
│   ├── frames/                   # Individual video frames
│   ├── trajectory_plan.json      # Planned object trajectory
│   ├── final_video.mp4           # Final output video
│   └── preview_collage.png       # Preview of all frames
└── scripts/
    ├── plan_trajectory.py        # VLM-based trajectory planning
    ├── generate_keyframes.py     # SAM + inpainting for keyframes
    ├── latent_warp_and_edit.py   # Intermediate frame generation
    └── assemble_video.py         # Video assembly
```

## Configuration

### Trajectory Planning
- **Model**: Qwen/Qwen2.5-VL-72B-Instruct (configurable)
- **Frames**: Default 16 frames
- **Output**: JSON file with per-frame positions and keyframe markers

### Keyframe Generation
- **Inpainting Model**: stabilityai/stable-diffusion-2-inpaint
- **Segmentation**: SAM (Segment Anything) for object masks
- **Optimization**: xformers for memory efficiency

### Video Assembly
- **Format**: MP4 with H.264 codec (mp4v)
- **FPS**: Default 15 fps (adjustable)
- **Resolution**: Auto-detected from frames

## Tips & Best Practices

1. **GPU Memory**: If OOM errors occur:
   - Reduce `num-frames` or `intermediate-steps`
   - Use `--device cpu` for CPU-only (much slower)
   - Reduce batch sizes in model configs

2. **Quality**: For better results:
   - Use high-resolution first frames (512x512+)
   - Write detailed, specific prompts
   - Increase `intermediate-steps` (3-4) for smoother motion

3. **Speed**: To iterate quickly:
   - Use `--skip-steps trajectory keyframes` to reuse previous plans
   - Reduce `num-frames` for testing
   - Use `--intermediate-steps 1` for quick tests

4. **Debugging**:
   - Check `trajectory_plan.json` to verify planned positions
   - Review individual frames in `outputs/frames/`
   - Use `--preview` flag to create a quick collage

## Common Issues

**Issue: CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
Solution:
- Reduce batch size or frame count
- Enable CPU offloading in diffusers
- Use gradient checkpointing

**Issue: Model Download Fails**
```
FileNotFoundError: Pretrained model not found
```
Solution:
- Check internet connection
- Set HF_HOME environment variable to custom cache
- Pre-download models manually from HuggingFace

**Issue: SAM Not Found**
```
ImportError: No module named 'segment_anything'
```
Solution:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Performance Notes

- **Trajectory Planning**: ~1-2 min (first VLM inference)
- **Keyframe Generation**: ~2-5 min per keyframe (depends on model size)
- **Interpolation**: ~30 sec per intermediate frame
- **Assembly**: ~10 sec for typical 16-frame video

Total for 16-frame video with 2 intermediate steps: ~10-20 minutes on V100

## GPU Requirements

| Component | Min VRAM | Recommended |
|-----------|----------|-------------|
| Qwen-VL   | 16 GB    | 24+ GB      |
| SD Inpaint| 8 GB     | 12+ GB      |
| SAM       | 4 GB     | 8 GB        |
| **Total** | **24 GB**| **40+ GB**  |

## License & Attribution

This pipeline uses:
- Qwen-VL: https://github.com/QwenLM/Qwen-VL
- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- Segment Anything: https://github.com/facebookresearch/segment-anything
- Diffusers: https://github.com/huggingface/diffusers

See respective repositories for licenses.
