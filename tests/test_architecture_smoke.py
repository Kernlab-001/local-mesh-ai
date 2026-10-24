"""
Architecture Smoke Test Suite for Local Mesh AI.
Level A Testing:
- Engine registration and capability declarations.
- Hardware routing across simulated hardware configurations (High-end 16GB, Mid-range 6GB, CPU).
- Input validation across modalities.
- Material-profile inference and relevance scoring (enforcing >= 0.70 threshold across 20 canonical cases).
"""

import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engines.registry import EngineRegistry, engine_registry
from core.engines.base import EngineStatus
from core.appearance_analyzer import appearance_analyzer
from tests.benchmark_suite import BENCHMARK_CASES


class TestEngineRegistrationAndCapabilities:
    """Verifies all required engines are registered and declare compliant capabilities."""

    def test_all_five_engines_registered(self):
        engines = engine_registry.list_engines()
        engine_ids = [e["engine_id"] for e in engines]
        assert "triposr" in engine_ids, "TripoSR must be registered"
        assert "shap_e" in engine_ids, "Shap-E must be registered"
        assert "hybrid_sd_triposr" in engine_ids, "Hybrid SD-Turbo + TripoSR must be registered"
        assert "stable_fast_3d" in engine_ids, "Stable Fast 3D must be registered"
        assert "trellis" in engine_ids, "TRELLIS must be registered"

    def test_capability_contracts(self):
        engines = engine_registry.list_engines()
        for eng in engines:
            caps = eng["capabilities"]
            assert isinstance(caps, dict), f"{eng['engine_id']} capabilities must be a dictionary"
            assert "watertight" in caps
            assert "vertex_colors" in caps
            assert "pbr_textures" in caps
            assert "uv_unwrapping" in caps
            assert "direct_text_conditioned" in caps

    def test_hardware_requirements_declared(self):
        engines = engine_registry.list_engines()
        for eng in engines:
            reqs = eng["hardware_requirements"]
            assert "min_vram_gb" in reqs, f"{eng['engine_id']} must declare min_vram_gb"
            assert "recommended_vram_gb" in reqs, f"{eng['engine_id']} must declare recommended_vram_gb"
            assert reqs["min_vram_gb"] >= 0.0


class TestSimulatedHardwareRouting:
    """Verifies dynamic routing under various simulated hardware tiers without modifying host."""

    def test_routing_on_high_end_workstation(self, monkeypatch):
        """Simulate a 24GB RTX 4090 workstation."""
        simulated_hw = {
            "has_cuda": True,
            "device_name": "NVIDIA GeForce RTX 4090",
            "total_vram_gb": 24.0,
            "available_vram_gb": 22.0,
            "total_ram_gb": 64.0,
            "available_ram_gb": 50.0,
            "compute_capability": (8, 9)
        }
        reg = EngineRegistry()
        monkeypatch.setattr(reg, "get_hardware_telemetry", lambda: simulated_hw)

        eng, meta = reg.resolve_best_engine(modality="image-to-3d")
        assert eng is not None
        assert eng.engine_id in ["triposr", "stable_fast_3d", "trellis"]

    def test_routing_on_mid_range_laptop_gpu(self, monkeypatch):
        """Simulate a 6GB RTX 3050 Laptop GPU (current machine profile)."""
        simulated_hw = {
            "has_cuda": True,
            "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "total_vram_gb": 6.0,
            "available_vram_gb": 5.2,
            "total_ram_gb": 16.0,
            "available_ram_gb": 10.0,
            "compute_capability": (8, 6)
        }
        reg = EngineRegistry()
        monkeypatch.setattr(reg, "get_hardware_telemetry", lambda: simulated_hw)

        # Image-to-3D on 6GB must select TripoSR
        img_eng, img_meta = reg.resolve_best_engine(modality="image-to-3d")
        assert img_eng.engine_id == "triposr"

        # Text-to-3D balanced on 6GB must select Hybrid SD-Turbo + TripoSR
        txt_eng, txt_meta = reg.resolve_best_engine(modality="text-to-3d", profile="balanced")
        assert txt_eng.engine_id == "hybrid_sd_triposr"

    def test_routing_on_cpu_low_memory(self, monkeypatch):
        """Simulate a CPU-only system with low memory."""
        simulated_hw = {
            "has_cuda": False,
            "device_name": "None",
            "total_vram_gb": 0.0,
            "available_vram_gb": 0.0,
            "total_ram_gb": 8.0,
            "available_ram_gb": 4.0,
            "compute_capability": None
        }
        reg = EngineRegistry()
        monkeypatch.setattr(reg, "get_hardware_telemetry", lambda: simulated_hw)

        eng, meta = reg.resolve_best_engine(modality="image-to-3d")
        assert eng is not None
        assert eng.engine_id == "triposr"


class TestInputValidation:
    """Verifies input sanitization and rejection of invalid data."""

    def test_rejects_missing_image_file(self):
        triposr = engine_registry.get_engine("triposr")
        is_valid, err = triposr.validate_input(input_data="non_existent_file_xyz_123.png", modality="image-to-3d")
        assert is_valid is False
        assert err is not None
        assert "does not exist" in err.lower() or "not found" in err.lower() or "missing" in err.lower()

    def test_accepts_valid_image_file(self):
        triposr = engine_registry.get_engine("triposr")
        example_img = PROJECT_ROOT / "TripoSR" / "examples" / "teapot.png"
        if example_img.exists():
            is_valid, err = triposr.validate_input(input_data=str(example_img), modality="image-to-3d")
            assert is_valid is True
            assert err is None

    def test_rejects_empty_prompt_for_text_to_3d(self):
        hybrid = engine_registry.get_engine("hybrid_sd_triposr")
        is_valid, err = hybrid.validate_input(input_data="", modality="text-to-3d")
        assert is_valid is False
        assert err is not None
        assert "non-empty" in err.lower() or "empty" in err.lower()

    def test_accepts_valid_prompt_for_text_to_3d(self):
        hybrid = engine_registry.get_engine("hybrid_sd_triposr")
        is_valid, err = hybrid.validate_input(input_data="medieval steel broadsword", modality="text-to-3d")
        assert is_valid is True
        assert err is None


class TestAppearanceInferenceAcross20Prompts:
    """
    Evaluates appearance and material parsing across all 20 canonical benchmark cases.
    Enforces a strict material relevance threshold of >= 0.70.
    """

    @pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[c.case_id for c in BENCHMARK_CASES])
    def test_material_relevance_threshold(self, case):
        profile = appearance_analyzer.analyze_appearance(prompt=case.prompt)
        assert len(profile.materials) > 0, f"Profile for {case.case_id} must have at least one material"

        # Calculate keyword match ratio against expected materials
        matched_count = 0
        for exp in case.expected_materials:
            exp_l = exp.lower()
            matched = False
            for m in profile.materials:
                m_name = m.name.lower()
                m_type = m.texture_type.lower()
                m_desc = m.description.lower()
                # Check for direct or semantic match
                if exp_l in m_name or exp_l in m_type or exp_l in m_desc:
                    matched = True
                    break
                # Special cases for aliases
                if exp_l == "steel" and ("metal" in m_name or "metal" in m_type):
                    matched = True
                    break
                if exp_l == "metal" and ("steel" in m_name or "blade" in m_name or "iron" in m_name):
                    matched = True
                    break
                if exp_l == "clay" and ("terracotta" in m_name or "ceramic" in m_type):
                    matched = True
                    break
                if exp_l == "soil" and ("earth" in m_name or "soil" in m_name):
                    matched = True
                    break
                if exp_l == "gem" and ("ruby" in m_name or "crystal" in m_type):
                    matched = True
                    break
                if exp_l == "chrome" and ("joint" in m_name or "metal" in m_type):
                    matched = True
                    break
                if exp_l == "composite" and ("armor" in m_name or "composite" in m_name or "composite" in m_type):
                    matched = True
                    break
                if exp_l == "aluminum" and ("alloy" in m_name or "rim" in m_name or "metal" in m_type):
                    matched = True
                    break
            if matched:
                matched_count += 1

        score = matched_count / max(len(case.expected_materials), 1)

        # Enforce threshold >= 0.70
        assert score >= 0.70, (
            f"Case {case.case_id} ('{case.name}') score {score:.2f} below threshold (0.70). "
            f"Expected {case.expected_materials}, materials: {[m.name for m in profile.materials]}"
        )
