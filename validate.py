#!/usr/bin/env python3
"""
Validation script to verify the video generation pipeline setup
"""

import os
import sys
from pathlib import Path


def check_project_structure():
    """Verify all necessary files exist"""
    print("=" * 60)
    print("VIDEO GENERATION PIPELINE - VALIDATION REPORT")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent
    
    required_files = {
        "Python Scripts": [
            "pipeline.py",
            "scripts/plan_trajectory.py",
            "scripts/generate_keyframes.py",
            "scripts/latent_warp_and_edit.py",
            "scripts/assemble_video.py",
        ],
        "Documentation": [
            "README.md",
            "INSTALL.md",
            "EXAMPLES.md",
            "requirements.txt",
        ],
        "Input Files": [
            "inputs/first_frame.png",
            "inputs/prompt.txt",
        ],
        "Directories": [
            "outputs",
            "outputs/frames",
            "scripts",
            "inputs",
        ]
    }
    
    all_checks_passed = True
    
    for category, files in required_files.items():
        print(f"📋 {category}:")
        for file_path in files:
            full_path = project_root / file_path
            if full_path.exists():
                if full_path.is_file():
                    size = full_path.stat().st_size
                    size_str = format_bytes(size)
                    print(f"   ✓ {file_path:40} ({size_str})")
                else:
                    print(f"   ✓ {file_path:40} (directory)")
            else:
                print(f"   ✗ {file_path:40} (MISSING)")
                all_checks_passed = False
        print()
    
    return all_checks_passed


def format_bytes(bytes_val):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def check_python_imports():
    """Verify key dependencies can be imported"""
    print("📦 Python Dependencies:")
    
    dependencies = [
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Hugging Face Diffusers"),
        ("transformers", "Hugging Face Transformers"),
    ]
    
    all_available = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✓ {name:35} (installed)")
        except ImportError:
            print(f"   ✗ {name:35} (NOT installed)")
            all_available = False
    
    print()
    return all_available


def check_code_quality():
    """Basic syntax check"""
    print("🔍 Code Quality Check:")
    
    project_root = Path(__file__).parent
    python_files = list(project_root.glob("**/*.py"))
    
    all_valid = True
    
    for py_file in python_files:
        try:
            with open(py_file) as f:
                compile(f.read(), str(py_file), 'exec')
            print(f"   ✓ {py_file.relative_to(project_root)} (valid syntax)")
        except SyntaxError as e:
            print(f"   ✗ {py_file.relative_to(project_root)} (syntax error: {e})")
            all_valid = False
    
    print()
    return all_valid


def show_next_steps():
    """Display next steps for user"""
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    
    print("2. Install SAM:")
    print("   pip install git+https://github.com/facebookresearch/segment-anything.git")
    print()
    
    print("3. Run the pipeline:")
    print("   python pipeline.py \\")
    print("     --first-frame inputs/first_frame.png \\")
    print("     --prompt 'A red ball bounces five times on a wooden table'")
    print()
    
    print("4. View results:")
    print("   - Video: outputs/final_video.mp4")
    print("   - Trajectory: outputs/trajectory_plan.json")
    print("   - Preview: outputs/preview_collage.png")
    print()


def main():
    """Run all validations"""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    checks_passed = True
    
    # Check structure
    checks_passed &= check_project_structure()
    
    # Check imports
    imports_ok = check_python_imports()
    if not imports_ok:
        print("⚠️  Warning: Some dependencies are not installed yet.")
        print("    Run: pip install -r requirements.txt")
        print()
    
    # Check code quality
    checks_passed &= check_code_quality()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print()
    
    if checks_passed:
        print("✓ Project structure is valid!")
        print("✓ All required files are present!")
        if not imports_ok:
            print("⚠ Dependencies need to be installed")
        else:
            print("✓ All Python dependencies are available!")
    else:
        print("✗ Some files are missing!")
        return 1
    
    print()
    show_next_steps()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
