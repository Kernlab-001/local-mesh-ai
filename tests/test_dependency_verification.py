"""
Unit tests for Dependency & Environment Verification (Repair 01).
Ensures dependency resolution, PyTorch CUDA compatibility, and installer verification checks.
"""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from core.verify_installation import (
    verify_python_version,
    verify_pytorch_and_cuda,
    verify_required_imports,
    run_full_verification,
    REQUIRED_MODULES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_python_version_check():
    """Verifies Python version validator accurately reflects runtime."""
    ok, msg = verify_python_version()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and 10 <= minor <= 12:
        assert ok is True
        assert f"Python {major}.{minor}" in msg
    else:
        assert ok is False


def test_pytorch_and_cuda_verification():
    """Verifies PyTorch and TorchVision versions and CUDA detection."""
    ok, info = verify_pytorch_and_cuda()
    assert ok is True
    assert info["torch_version"] is not None
    assert info["torchvision_version"] is not None
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["vram_gb"], float)
    if info["cuda_available"]:
        assert info["vram_gb"] > 0
        assert info["device_name"] != "CPU Only"
    else:
        assert "CPU" in info["device_name"]


def test_pytorch_torchvision_mismatch_detection():
    """Verifies that an incompatible TorchVision/PyTorch pair is flagged as a failure."""
    fake_cuda = type("CUDA", (), {"is_available": staticmethod(lambda: False)})
    fake_torch = type("Torch", (), {"__version__": "2.14.1", "cuda": fake_cuda})
    fake_vision = type("Vision", (), {"__version__": "0.21.0+cu124"})

    with patch.dict("sys.modules", {"torch": fake_torch, "torchvision": fake_vision}):
        ok, info = verify_pytorch_and_cuda()
        assert ok is False
        assert "Version mismatch" in info["error"]


def test_all_required_imports():
    """Verifies that all 14 required modules import without error."""
    ok, succeeded, failed = verify_required_imports()
    assert ok is True
    assert len(failed) == 0
    assert len(succeeded) == len(REQUIRED_MODULES)


def test_requirements_file_consistency():
    """Verifies requirements.txt does not contain known incompatible Dependabot hallucinations."""
    req_path = PROJECT_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt must exist at project root"

    content = req_path.read_text(encoding="utf-8")

    # Disallow known conflicting hallucinated versions
    assert "torch==2.13.0" not in content, "torch 2.13.0 does not exist"
    assert "torch==2.14.1" not in content, "torch 2.14.1 does not exist"
    assert "torchvision==0.29.1" not in content, "torchvision 0.29.1 is invalid"
    assert "transformers==5.10.1" not in content, "transformers 5.x is invalid"
    assert "numpy>=2.5.4" not in content, "numpy >= 2.5.4 is invalid"
    assert "gradio>=6.30.0" not in content, "gradio >= 6.30.0 is invalid"

    # Confirm correct cu124 index and pins
    assert "--extra-index-url https://download.pytorch.org/whl/cu124" in content
    assert "torch==2.6.0+cu124" in content
    assert "torchvision==0.21.0+cu124" in content


def test_full_verification_runner():
    """Verifies run_full_verification() returns True in the valid environment."""
    assert run_full_verification() is True
