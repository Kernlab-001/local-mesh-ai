"""
Unit Tests for Phase 3 Model Management & Hardware Orchestration.
Tests:
- AI Model Catalogue metadata, tasks, licensing, and installation state
- Hardware Profile Selector (4GB, 6GB, 8GB, 16GB tiers)
- RTX 3050 6GB primary testbed profile detection
- User manual overrides and validation
- Sequential Model Lifecycle state machine and VRAM release verification
"""

import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_model_catalog import ai_model_catalog, ModelTaskCategory, VerificationStatus
from core.model_profile_selector import model_profile_selector, VRAMTier, RAMTier
from core.model_lifecycle import SequentialModelManager, ModelSlot, ModelLocation


def test_ai_model_catalog_entries():
    """Verifies all required model tasks are represented with valid metadata."""
    entries = ai_model_catalog.get_all_entries()
    assert len(entries) >= 8

    # Check key models exist
    model_ids = [e.model_id for e in entries]
    assert "triposr" in model_ids
    assert "qwen2.5_coder_7b" in model_ids
    assert "qwen3_vl_4b_instruct" in model_ids
    assert "partfield" in model_ids
    assert "geometric_feature_field" in model_ids
    assert "cpu_geometry_engine" in model_ids
    assert "blender_deterministic_tools" in model_ids

    # Verify task categories
    recon_entries = ai_model_catalog.get_entries_by_task(ModelTaskCategory.IMAGE_TO_3D_RECONSTRUCTION)
    assert any(e.model_id == "triposr" for e in recon_entries)

    vlm_entries = ai_model_catalog.get_entries_by_task(ModelTaskCategory.SEMANTIC_VISION_RECOGNITION)
    assert any("qwen" in e.model_id for e in vlm_entries)


def test_hardware_profile_evaluation():
    """Verifies automatic detection and recommendation profile generation."""
    rep = model_profile_selector.evaluate_hardware_profile()

    assert rep.detected_gpu_name != ""
    assert rep.usable_vram_budget_gb <= rep.detected_vram_gb
    assert rep.vram_tier in list(VRAMTier)
    assert rep.ram_tier in list(RAMTier)

    # Check that recommendation stack covers all core tasks
    stack = rep.recommended_stack
    assert "image_to_3d_reconstruction" in stack
    assert "technical_artist_planning" in stack
    assert "semantic_vision_recognition" in stack
    assert "segmentation_3d" in stack
    assert "geometry_construction_editing" in stack

    # Primary hardware check (RTX 3050 Laptop)
    if rep.is_primary_target_hardware:
        assert stack["image_to_3d_reconstruction"].selected_model_id == "triposr"
        assert stack["technical_artist_planning"].selected_model_id == "qwen2.5_coder_7b"
        assert stack["technical_artist_planning"].execution_device == "cuda:0"


def test_user_override_mechanism():
    """Verifies user can override recommendations and clear overrides."""
    # Set override for planning model
    model_profile_selector.set_user_override(
        ModelTaskCategory.TECHNICAL_ARTIST_PLANNING,
        "qwen2.5_coder_1.5b"
    )
    rep = model_profile_selector.evaluate_hardware_profile()
    selected = rep.recommended_stack["technical_artist_planning"]
    assert selected.selected_model_id == "qwen2.5_coder_1.5b"
    assert selected.is_user_override

    # Clear override
    model_profile_selector.clear_user_override(ModelTaskCategory.TECHNICAL_ARTIST_PLANNING)
    rep_restored = model_profile_selector.evaluate_hardware_profile()
    assert not rep_restored.recommended_stack["technical_artist_planning"].is_user_override


def test_invalid_override_rejection():
    """Verifies invalid model ID or task mismatch is rejected."""
    with pytest.raises(ValueError):
        model_profile_selector.set_user_override(
            ModelTaskCategory.TECHNICAL_ARTIST_PLANNING,
            "nonexistent_model_id"
        )


def test_sequential_model_lifecycle_zero_vram():
    """Verifies single-slot residency and zero-VRAM overlap enforcement."""
    mgr = SequentialModelManager(vram_budget_gb=6.0)

    # Prepare slot 1
    dummy_model_1 = {"name": "model_1"}
    mgr.prepare_slot(ModelSlot.NEURAL_RECONSTRUCTION, lambda: dummy_model_1)
    assert mgr.active_slot == ModelSlot.NEURAL_RECONSTRUCTION

    # Prepare slot 2: slot 1 must be evicted!
    dummy_model_2 = {"name": "model_2"}
    mgr.prepare_slot(ModelSlot.LLM_PLANNER, lambda: dummy_model_2)
    assert mgr.active_slot == ModelSlot.LLM_PLANNER
    assert ModelSlot.NEURAL_RECONSTRUCTION not in mgr._gpu_resident

    # Offload and purge
    mgr.offload_active_model()
    assert mgr.active_slot is None
    vram = mgr.get_vram_usage()
    assert "allocated_gb" in vram
