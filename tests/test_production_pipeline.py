"""
Unit & Integration Tests for Production-Grade Upgrades:
1. Sequential Model Manager (Zero-VRAM Overlap).
2. Multi-View Consistent Conditioning Engine.
3. Automated Retopology & High-to-Low Texture Baking Bridge.
"""

import os
import pytest
from PIL import Image
from core.model_lifecycle import model_lifecycle, ModelSlot, ModelLocation
from core.multiview_synthesizer import multiview_synthesizer

def test_model_lifecycle_vram_and_purge():
    # Verify VRAM usage report schema
    usage = model_lifecycle.get_vram_usage()
    assert "allocated_gb" in usage
    assert "reserved_gb" in usage
    assert "total_gb" in usage
    assert "free_gb" in usage
    assert isinstance(usage["allocated_gb"], float)

    # Test purging does not raise
    model_lifecycle.purge_cuda_memory()
    assert model_lifecycle.active_slot is None

def test_model_lifecycle_slot_transition():
    dummy_model = {"state": "active"}

    # Simulate slot activation
    model_lifecycle.active_slot = ModelSlot.CONCEPT_DIFFUSION
    model_lifecycle._gpu_resident[ModelSlot.CONCEPT_DIFFUSION] = dummy_model

    # Offload
    model_lifecycle.offload_active_model(keep_in_cpu_ram=False)
    assert model_lifecycle.active_slot is None
    assert ModelSlot.CONCEPT_DIFFUSION not in model_lifecycle._gpu_resident

def test_multiview_synthesis(tmp_path):
    # Create test input image
    test_img = Image.new("RGBA", (256, 256), color=(0, 200, 100, 255))
    img_path = str(tmp_path / "test_concept.png")
    test_img.save(img_path)

    out_dir = str(tmp_path / "mv_output")
    result = multiview_synthesizer.generate_canonical_views(img_path, out_dir, use_zero123=False)

    assert result["status"] == "success"
    assert os.path.exists(result["front"])
    assert os.path.exists(result["right"])
    assert os.path.exists(result["back"])
    assert os.path.exists(result["left"])
    assert os.path.exists(result["grid"])
    assert result["num_views"] == 4
