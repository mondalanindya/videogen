#!/bin/bash
# Fix PyTorch version incompatibility
# Your system has PyTorch 2.0.1 (CPU-only from conda)
# but transformers 4.57.6 requires PyTorch >= 2.1

# Option 1: Upgrade PyTorch (CPU-only, no CUDA support on ARM64)
echo "=== Upgrading PyTorch to 2.1+ (CPU-only) ==="
/projects/u6bl/myprojects/vscode_pseudo_home/miniconda3/envs/videogen/bin/pip install --upgrade --force-reinstall torch torchvision

# Option 2 (if Option 1 fails): Downgrade transformers to work with PyTorch 2.0.1
# Uncomment these lines and comment out Option 1 above
# echo "=== Downgrading transformers to work with PyTorch 2.0.1 ==="
# /projects/u6bl/myprojects/vscode_pseudo_home/miniconda3/envs/videogen/bin/pip install \
#   "transformers==4.30.0" \
#   "diffusers==0.21.4" \
#   "accelerate==0.21.0"

echo "=== Verifying installation ==="
/projects/u6bl/myprojects/vscode_pseudo_home/miniconda3/envs/videogen/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}')"
