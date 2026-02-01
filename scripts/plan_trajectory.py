"""
Trajectory Planning Module
Generates a frame-by-frame trajectory plan using Qwen-VL to understand motion dynamics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class TrajectoryPlanner:
    """Plans object trajectories using vision-language models."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct", device: str = "cuda"):
        """
        Initialize the trajectory planner.
        
        Args:
            model_name: HuggingFace model ID for the VLM
            device: torch device to use
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen2.5-VL model and processor."""
        print(f"Loading {self.model_name}...")
        try:
            # Load model using correct class for Qwen2.5-VL
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def plan_trajectory(
        self,
        first_frame_path: str,
        prompt: str,
        num_frames: int = 16,
        output_path: str = "trajectory_plan.json"
    ) -> Dict[str, Any]:
        """
        Generate a trajectory plan for the given prompt and first frame.
        
        Args:
            first_frame_path: Path to the initial frame image
            prompt: Text description of desired motion (e.g., "A red ball bounces 5 times")
            num_frames: Total number of frames to plan
            output_path: Where to save the trajectory JSON
            
        Returns:
            Dictionary containing the trajectory plan
        """
        if not os.path.exists(first_frame_path):
            raise FileNotFoundError(f"First frame not found: {first_frame_path}")

        # Load first frame
        frame = Image.open(first_frame_path).convert("RGB")
        frame_h, frame_w = frame.size

        # Construct VLM prompt
        vlm_prompt = f"""Analyze this image and the following motion prompt.
Generate a JSON plan with exactly {num_frames} frames describing the object motion.

Motion Prompt: "{prompt}"

For each frame, provide:
- frame: frame index (0 to {num_frames - 1})
- x: horizontal position (0 to {frame_w - 1})
- y: vertical position (0 to {frame_h - 1})
- radius: object radius (pixels, assume consistent)
- angle: rotation angle in degrees (0-360)
- keyframe: boolean, true for key motion events (bounces, direction changes, etc.)
- confidence: float 0-1 indicating confidence in this position

Return ONLY a valid JSON object with a "frames" array.
Example format:
{{
  "frames": [
    {{"frame": 0, "x": 100, "y": 300, "radius": 20, "angle": 0, "keyframe": true, "confidence": 1.0}},
    {{"frame": 1, "x": 110, "y": 290, "radius": 20, "angle": 15, "keyframe": false, "confidence": 0.95}}
  ]
}}"""

        print(f"Planning trajectory for: {prompt}")
        print(f"Generating {num_frames} frames...")

        # Call VLM
        try:
            # For Qwen2.5-VL, use the correct inference pattern
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frame,
                        },
                        {
                            "type": "text",
                            "text": vlm_prompt
                        }
                    ]
                }
            ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2000)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Extract JSON from response
            trajectory_plan = self._extract_json(response_text)
            
        except Exception as e:
            print(f"✗ VLM inference error: {e}")
            print("Falling back to synthetic trajectory generation...")
            trajectory_plan = self._generate_synthetic_trajectory(
                num_frames, frame_w, frame_h, prompt
            )

        # Save trajectory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(trajectory_plan, f, indent=2)
        
        print(f"✓ Trajectory saved to {output_path}")
        print(f"  - Total frames: {len(trajectory_plan['frames'])}")
        print(f"  - Keyframes: {sum(1 for f in trajectory_plan['frames'] if f['keyframe'])}")
        
        return trajectory_plan

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response."""
        # Find JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                raise ValueError("Failed to parse JSON from model response")
        
        raise ValueError("No JSON found in model response")

    def _generate_synthetic_trajectory(
        self,
        num_frames: int,
        width: int,
        height: int,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Generate a synthetic trajectory when VLM is unavailable.
        Serves as a fallback for testing.
        """
        frames = []
        
        # Physics-based bouncing ball trajectory
        is_bouncing = "bounce" in prompt.lower()
        
        # Extract number of bounces from prompt
        import re
        bounce_match = re.search(r'(\d+)\s+time', prompt.lower())
        num_bounces = int(bounce_match.group(1)) if bounce_match else 5
        
        # Physics parameters
        x_start = width // 2
        y_ground = height - 80
        gravity = 0.8  # Stronger gravity for realistic falls
        initial_vy = -15.0  # Stronger initial velocity
        damping = 0.7  # Energy loss per bounce
        
        # State variables
        x = x_start
        y = float(y_ground)
        vy = initial_vy
        bounce_count = 0
        
        for i in range(num_frames):
            if is_bouncing:
                # Apply gravity
                vy += gravity
                y += vy
                
                # Horizontal drift
                x = x_start + 40 * (i / num_frames)
                
                # Bounce detection
                if y >= y_ground:
                    y = y_ground
                    vy = -abs(vy) * damping
                    bounce_count += 1
                    keyframe = True
                else:
                    keyframe = False
                
                if bounce_count > num_bounces:
                    y = y_ground
                    vy = 0
            else:
                # Linear motion
                x = x_start + (width // 4) * (i / num_frames)
                y = y_ground - (height // 3) * (i / num_frames)
                keyframe = (i % max(1, num_frames // 4) == 0)
            
            # Calculate velocity for squash/stretch
            if i > 0:
                velocity_y = y - frames[-1]["y"]
            else:
                velocity_y = 0
            
            frames.append({
                "frame": i,
                "x": int(x),
                "y": int(y),
                "radius": 20,
                "angle": (i * 30) % 360,
                "velocity_y": velocity_y,
                "keyframe": keyframe,
                "confidence": 0.85
            })
        
        return {"frames": frames}


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trajectory plan from video prompt")
    parser.add_argument("--first-frame", required=True, help="Path to first frame image")
    parser.add_argument("--prompt", required=True, help="Motion prompt (e.g., 'A red ball bounces 5 times')")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--output", default="trajectory_plan.json", help="Output trajectory JSON")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    planner = TrajectoryPlanner(device=args.device)
    planner.plan_trajectory(
        first_frame_path=args.first_frame,
        prompt=args.prompt,
        num_frames=args.num_frames,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
