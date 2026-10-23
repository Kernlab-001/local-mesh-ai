"""
Verification script for Local Mesh AI environment installation.
Validates Python executable, PyTorch, TorchVision, CUDA capability,
and all required machine learning and 3D geometry dependencies.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

REQUIRED_MODULES = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("diffusers", "Hugging Face Diffusers"),
    ("transformers", "Transformers"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("gradio", "Gradio"),
    ("trimesh", "Trimesh"),
    ("rembg", "Rembg"),
    ("xatlas", "Xatlas"),
    ("moderngl", "ModernGL"),
    ("PIL", "Pillow"),
    ("numpy", "NumPy"),
    ("pytest", "Pytest")
]


def verify_python_version() -> Tuple[bool, str]:
    """Ensures Python version is within the supported 3.10 - 3.12 range."""
    major, minor = sys.version_info.major, sys.version_info.minor
    ver_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major != 3 or minor < 10 or minor > 12:
        return False, f"Python {ver_str} is unsupported. Python 3.10 - 3.12 is required."
    return True, f"Python {ver_str} (Executable: {sys.executable})"


def verify_pytorch_and_cuda() -> Tuple[bool, Dict[str, Any]]:
    """Inspects PyTorch and TorchVision compatibility and accurately distinguishes GPU vs CPU."""
    info = {
        "torch_version": None,
        "torchvision_version": None,
        "cuda_available": False,
        "device_name": "CPU Only",
        "vram_gb": 0.0,
        "cuda_version": None,
        "error": None
    }
    try:
        import torch
        import torchvision
        info["torch_version"] = torch.__version__
        info["torchvision_version"] = torchvision.__version__

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
        else:
            info["cuda_available"] = False
            info["device_name"] = "CPU Only (No CUDA GPU detected)"
            info["vram_gb"] = 0.0

        # Check torch/torchvision compatibility
        # TorchVision 0.21.x requires PyTorch 2.6.x
        if "0.21." in info["torchvision_version"] and not info["torch_version"].startswith("2.6."):
            info["error"] = (
                f"Version mismatch: TorchVision {info['torchvision_version']} requires PyTorch 2.6.x, "
                f"but found PyTorch {info['torch_version']}."
            )
            return False, info

        return True, info
    except Exception as exc:
        info["error"] = str(exc)
        return False, info


def verify_required_imports() -> Tuple[bool, List[str], List[str]]:
    """Verifies that all core dependencies can be imported successfully."""
    succeeded = []
    failed = []

    for mod_name, label in REQUIRED_MODULES:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            succeeded.append(f"{label} ({mod_name} {ver})")
        except Exception as exc:
            failed.append(f"{label} ({mod_name}): {exc}")

    return len(failed) == 0, succeeded, failed


def run_full_verification() -> bool:
    """Runs all checks, prints detailed diagnostics, and returns True on success."""
    print("=" * 66)
    print("  LOCAL MESH AI - ENVIRONMENT VERIFICATION")
    print("=" * 66)

    # 1. Python version
    py_ok, py_msg = verify_python_version()
    print(f"  [1/3] Python Runtime:  {py_msg}")
    if not py_ok:
        print(f"  [ERROR] {py_msg}")
        return False

    # 2. PyTorch & CUDA
    torch_ok, torch_info = verify_pytorch_and_cuda()
    if not torch_ok:
        print(f"  [ERROR] PyTorch / TorchVision check failed: {torch_info.get('error')}")
        return False

    print(f"  [2/3] PyTorch Core:    torch {torch_info['torch_version']} | torchvision {torch_info['torchvision_version']}")
    if torch_info["cuda_available"]:
        print(f"        GPU Acceleration: [ENABLED] {torch_info['device_name']} ({torch_info['vram_gb']} GB VRAM, CUDA {torch_info['cuda_version']})")
    else:
        print(f"        GPU Acceleration: [CPU ONLY] CUDA is not available. System will run in CPU fallback mode.")

    # 3. Imports
    imports_ok, succeeded, failed = verify_required_imports()
    print(f"  [3/3] Neural & 3D Libs: {len(succeeded)}/{len(REQUIRED_MODULES)} modules verified.")
    if not imports_ok:
        print("\n  [ERROR] The following required modules failed to import:")
        for item in failed:
            print(f"    - {item}")
        return False

    print("=" * 66)
    print("  [SUCCESS] All dependencies verified and ready for execution!")
    print("=" * 66)
    return True


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)
