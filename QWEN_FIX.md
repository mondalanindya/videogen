# Qwen2.5-VL Fix Summary

## Issues Found and Fixed

### 1. ❌ Incorrect Model Loading
**Problem:** Using `AutoModelForCausalLM` which doesn't support Qwen2.5-VL configuration
```python
# WRONG
self.model = AutoModelForCausalLM.from_pretrained(...)
```

**Solution:** Use the correct model class `Qwen2_5_VLForConditionalGeneration`
```python
# CORRECT
from transformers import Qwen2_5_VLForConditionalGeneration
self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
```

### 2. ❌ Deprecated torch_dtype Parameter
**Problem:** Using `torch_dtype=torch.float16` which is deprecated
```python
# WRONG
torch_dtype=torch.float16
```

**Solution:** Use `torch_dtype="auto"` for automatic dtype selection
```python
# CORRECT
torch_dtype="auto"
```

### 3. ❌ Wrong Processor Usage
**Problem:** Missing `AutoProcessor` and not using `process_vision_info()`
```python
# WRONG
text = self.tokenizer.apply_chat_template(...)
inputs = self.tokenizer(text=text, images=image_inputs, ...)
```

**Solution:** Use `AutoProcessor` with proper vision info processing
```python
# CORRECT
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained(model_name)
text = processor.apply_chat_template(messages, ...)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, ...)
```

### 4. ❌ Incorrect Batch Decoding
**Problem:** Using tokenizer.decode() on single output
```python
# WRONG
response_text = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], ...)
```

**Solution:** Use processor.batch_decode() properly
```python
# CORRECT
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response_text = self.processor.batch_decode(generated_ids_trimmed, ...)[0]
```

### 5. ❌ Model Size Too Large
**Problem:** Using 72B model requires 80+ GB VRAM
```python
# WRONG
model_name = "Qwen/Qwen2.5-VL-72B-Instruct"  # Too large
```

**Solution:** Use 7B model which requires ~16-24 GB VRAM
```python
# CORRECT
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # Fits in V100
```

## Changes Made

### Modified Files:
1. **scripts/plan_trajectory.py**
   - ✓ Updated imports to use correct model class
   - ✓ Fixed model loading with `torch_dtype="auto"`
   - ✓ Updated to use `AutoProcessor` instead of tokenizer
   - ✓ Implemented correct inference pattern with `process_vision_info()`
   - ✓ Fixed batch decoding logic
   - ✓ Changed default model to 7B version

2. **requirements.txt**
   - ✓ Added `qwen-vl-utils>=0.0.8` dependency
   - ✓ Updated transformers minimum version to 4.40.0

3. **INSTALL.md**
   - ✓ Added step for installing qwen-vl-utils with decord support
   - ✓ Reordered installation steps

4. **README.md & PROJECT_SUMMARY.md**
   - ✓ Updated model references from 72B to 7B

## Memory Requirements (Updated)

| Component | Model | VRAM |
|-----------|-------|------|
| Qwen2.5-VL-7B | 7B parameters | 14-16 GB |
| Stable Diffusion | 2B parameters | 8-10 GB |
| SAM | 86M parameters | 4 GB |
| **Total** | | **26-30 GB** |

✅ Now fits on V100 (32GB VRAM)

## Installation Instructions

```bash
# 1. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install qwen-vl-utils with decord support
pip install qwen-vl-utils[decord]==0.0.8

# 4. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Testing the Fix

```bash
python pipeline.py \
  --first-frame inputs/first_frame.png \
  --prompt "A red ball bounces five times on a wooden table"
```

The pipeline should now:
1. ✓ Load the Qwen2.5-VL-7B model successfully
2. ✓ Process the image with correct vision info
3. ✓ Generate trajectory plan without errors
4. ✓ Continue to keyframe generation and video assembly

## Reference Implementation

The fix follows the official Qwen2.5-VL inference guide:
https://github.com/QwenLM/Qwen2.5-VL

Key patterns used:
- `Qwen2_5_VLForConditionalGeneration` for model loading
- `AutoProcessor` for preprocessing
- `process_vision_info()` for image/video handling
- Batch decoding with trimmed token IDs
