"""
Example Scripts and Usage Patterns
Quick reference for common tasks
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

"""
Simplest way to run the pipeline:

python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A red ball bounces five times on a wooden table"
"""

# ============================================================================
# EXAMPLE 1: Generate Video from Prompt
# ============================================================================

"""
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A red ball bounces five times on a wooden table" \
  --num-frames 16 \
  --fps 15
"""

# ============================================================================
# EXAMPLE 2: High-Quality Output
# ============================================================================

"""
For smoother, higher-quality video with more interpolated frames:

python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A blue cube rolls across a wooden floor" \
  --num-frames 32 \
  --intermediate-steps 4 \
  --fps 24 \
  --output-dir outputs_hq
"""

# ============================================================================
# EXAMPLE 3: Quick Testing
# ============================================================================

"""
For fast iteration (fewer frames, less interpolation):

python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball bounces" \
  --num-frames 8 \
  --intermediate-steps 1 \
  --fps 10
"""

# ============================================================================
# EXAMPLE 4: CPU-Only Mode (slow but no GPU needed)
# ============================================================================

"""
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball moves slowly" \
  --device cpu \
  --num-frames 8
"""

# ============================================================================
# EXAMPLE 5: Iterate Without Regenerating Keyframes
# ============================================================================

"""
If you want to test different interpolation settings without 
regenerating keyframes, skip earlier steps:

# First run: generate trajectory and keyframes
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball bounces" \
  --output-dir outputs_v1

# Second run: reuse trajectory/keyframes, just change interpolation
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball bounces" \
  --output-dir outputs_v1 \
  --skip-steps trajectory keyframes \
  --intermediate-steps 4
"""

# ============================================================================
# EXAMPLE 6: Individual Component Testing
# ============================================================================

"""
Test trajectory planning only:
python scripts/plan_trajectory.py \
  --first-frame inputs/first_frame.png \
  --prompt "A ball bounces five times" \
  --num-frames 16 \
  --output trajectory_plan.json

Test keyframe generation:
python scripts/generate_keyframes.py \
  --first-frame inputs/first_frame.png \
  --trajectory trajectory_plan.json \
  --output-dir outputs/frames

Generate intermediate frames:
python scripts/latent_warp_and_edit.py \
  --trajectory trajectory_plan.json \
  --keyframes-dir outputs/frames \
  --output-dir outputs/frames \
  --num-steps 2

Assemble final video:
python scripts/assemble_video.py \
  --frames-dir outputs/frames \
  --output outputs/final_video.mp4 \
  --fps 15 \
  --preview
"""

# ============================================================================
# EXAMPLE 7: Batch Processing Multiple Prompts
# ============================================================================

"""
Create a batch processing script:

#!/bin/bash

FIRST_FRAME="inputs/first_frame.png"
PROMPTS=(
  "A red ball bounces five times"
  "A blue ball rolls to the right"
  "A yellow ball spins in place"
)

for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  OUTPUT_DIR="outputs_batch_$i"
  
  echo "Processing: $PROMPT"
  python pipeline.py \
    --first-frame "$FIRST_FRAME" \
    --prompt "$PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames 16 \
    --intermediate-steps 2
  
  echo "✓ Completed: $OUTPUT_DIR"
done
"""

# ============================================================================
# EXAMPLE 8: Using Different Models
# ============================================================================

"""
For trajectory planning, you can modify plan_trajectory.py to use
different VLM models. By default it uses Qwen2.5-VL-72B-Instruct.

To use GPT-4V (requires OpenAI API):

# Modify scripts/plan_trajectory.py to use OpenAI API:
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": base64_image}},
            {"type": "text", "text": vlm_prompt}
        ]
    }],
    max_tokens=2000
)

For inpainting, you can try other SD models:
- stabilityai/stable-diffusion-3-medium (newer, better quality)
- stabilityai/stable-diffusion-xl-base-1.0 (larger, high quality)

Just update model_id in generate_keyframes.py
"""

# ============================================================================
# EXAMPLE 9: Advanced: Custom Prompts for Different Objects
# ============================================================================

"""
Ball examples:
  "A red ball bounces five times on a wooden table"
  "A blue ball rolls slowly across the floor"
  "A metallic sphere spins and bounces"

Cube examples:
  "A wooden cube slides across the table"
  "A blue cube falls and rolls"
  "A gold cube tumbles down stairs"

Other objects:
  "An apple rolls from left to right"
  "A pendulum swings back and forth"
  "A spinning top gradually slows down"

Complex motions:
  "A ball bounces in a circular pattern"
  "A cube rotates on its corner"
  "Multiple objects (ball and cube) collide"
"""

# ============================================================================
# EXAMPLE 10: Custom Configuration File
# ============================================================================

"""
Create a config.json for reproducible runs:

{
  "trajectory": {
    "num_frames": 16,
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "inference_steps": 30
  },
  "keyframes": {
    "inpaint_model": "stabilityai/stable-diffusion-2-inpaint",
    "guidance_scale": 7.5,
    "num_inference_steps": 30
  },
  "interpolation": {
    "intermediate_steps": 2,
    "flow_method": "farneback",
    "warp_mode": "bilinear"
  },
  "video": {
    "fps": 15,
    "codec": "mp4v",
    "width": 512,
    "height": 512
  }
}

Then load in pipeline.py:
import json
with open('config.json') as f:
    config = json.load(f)
"""

# ============================================================================
# DEBUGGING AND TROUBLESHOOTING
# ============================================================================

"""
Check generated trajectory:
  1. Look at outputs/trajectory_plan.json
  2. Verify frame positions make sense
  3. Check keyframe markers align with motion events

Inspect individual frames:
  1. Look at outputs/frames/frame_0000.png etc.
  2. Verify object is in correct position
  3. Check for inpainting artifacts

Test optical flow:
  import cv2
  frame1 = cv2.imread("outputs/frames/frame_0000.png")
  frame2 = cv2.imread("outputs/frames/frame_0001.png")
  flow = cv2.calcOpticalFlowFarneback(...)
  
  # Visualize flow
  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  
Verify video:
  ffprobe outputs/final_video.mp4
  
Preview frames:
  Display outputs/preview_collage.png
"""

# ============================================================================
# MEMORY AND PERFORMANCE OPTIMIZATION
# ============================================================================

"""
If running out of memory:

1. Reduce frame count:
   --num-frames 8

2. Reduce interpolation:
   --intermediate-steps 1

3. Use CPU offloading (slow but saves VRAM):
   # In the code:
   pipe.enable_sequential_cpu_offload()

4. Use smaller models:
   # In generate_keyframes.py:
   model_id = "stabilityai/stable-diffusion-1-5-inpaint"

5. Reduce inference steps:
   # In code, reduce num_inference_steps

For faster execution:
1. Enable xformers (already done by default)
2. Use smaller num_frames
3. Reduce intermediate_steps
4. Use fewer inference_steps (default 30, try 20)
"""

# ============================================================================
# OUTPUT FORMATS
# ============================================================================

"""
Trajectory JSON format:
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

Frame files:
  outputs/frames/frame_0000.png
  outputs/frames/frame_0001.png
  ...

Video output:
  outputs/final_video.mp4 (H.264 MP4)
  
Preview:
  outputs/preview_collage.png (grid of selected frames)
"""

# ============================================================================
# EVALUATION AND METRICS
# ============================================================================

"""
Basic quality checks:

1. Frame count:
   len(glob.glob("outputs/frames/frame_*.png"))

2. Position consistency:
   # Check positions follow trajectory plan

3. Visual continuity:
   # Use optical flow consistency metric

4. Inpainting quality:
   # Check for artifacts, blur, color discontinuities

5. Video playback:
   ffplay outputs/final_video.mp4 -loop 0
"""

print(__doc__)
