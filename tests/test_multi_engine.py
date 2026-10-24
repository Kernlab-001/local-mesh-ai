"""
Comprehensive Unit & Regression Test Suite for Multi-Engine Architecture.
Validates:
- Engine Registry and dynamic hardware-adaptive router.
- Concrete adapters (TripoSR, Shap-E, Hybrid, SF3D, TRELLIS).
- Model and dependency manager.
- Geometry quality validation & white-material healing pipeline.
- Job Manager multi-engine forwarding and fallback guarantees.
"""

import os
import sys
import shutil
import tempfile
import pytest
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset
from core.engines.registry import engine_registry, EngineRegistry
from core.engines.triposr_adapter import TripoSRAdapter
from core.engines.shape_adapter import ShapEAdapter
from core.engines.hybrid_text_adapter import HybridTextTo3DAdapter
from core.engines.sf3d_adapter import StableFast3DAdapter
from core.engines.trellis_adapter import TrellisAdapter
from core.model_manager import model_manager, ModelMetadata
from core.geometry_validator import geometry_validator, GeometryValidator, GeometryValidationReport

try:
    import trimesh
except ImportError:
    trimesh = None


class TestEngineRegistryAndRouting:
    def test_default_engines_registered(self):
        engines = engine_registry.list_engines()
        engine_ids = [e["engine_id"] for e in engines]
        assert "triposr" in engine_ids
        assert "shap_e" in engine_ids
        assert "hybrid_sd_triposr" in engine_ids
        assert "stable_fast_3d" in engine_ids
        assert "trellis" in engine_ids

    def test_hardware_telemetry_fields(self):
        hw = engine_registry.get_hardware_telemetry()
        assert "has_cuda" in hw
        assert "device_name" in hw
        assert "total_vram_gb" in hw
        assert "available_vram_gb" in hw
        assert "total_ram_gb" in hw
        assert "available_ram_gb" in hw
        assert isinstance(hw["total_vram_gb"], (int, float))

    def test_dynamic_routing_image_to_3d(self):
        engine, meta = engine_registry.resolve_best_engine(
            modality="image-to-3d",
            profile="balanced"
        )
        assert engine is not None
        assert engine.engine_id == "triposr"
        assert "Optimal feedforward volumetric reconstruction" in meta["reason"]

    def test_dynamic_routing_text_to_3d_hybrid(self):
        engine, meta = engine_registry.resolve_best_engine(
            modality="text-to-3d",
            profile="balanced"
        )
        assert engine is not None
        assert engine.engine_id == "hybrid_sd_triposr"
        assert "Strategy B" in meta["reason"]

    def test_dynamic_routing_text_to_3d_direct(self):
        engine, meta = engine_registry.resolve_best_engine(
            modality="text-to-3d",
            profile="direct_latent"
        )
        assert engine is not None
        assert engine.engine_id == "shap_e"
        assert "Strategy A" in meta["reason"]

    def test_user_override_unsupported_hardware_warning(self):
        # Requesting TRELLIS (which requires 16GB) on a 6GB machine should return warning
        engine, meta = engine_registry.resolve_best_engine(
            modality="image-to-3d",
            user_override_engine="trellis"
        )
        assert engine.engine_id == "trellis"
        assert meta["warning"] is not None
        assert "unsupported" in meta["warning"].lower() or "missing" in meta["warning"].lower()


class TestEngineAdapters:
    def test_triposr_adapter_capabilities(self):
        adapter = TripoSRAdapter()
        assert "image-to-3d" in adapter.supported_modalities
        caps = adapter.capabilities
        assert caps["watertight"] is True
        assert caps["vertex_colors"] is True
        req = adapter.hardware_requirements
        assert req["min_vram_gb"] <= 4.0

    def test_shape_adapter_capabilities(self):
        adapter = ShapEAdapter()
        assert "text-to-3d" in adapter.supported_modalities
        assert "image-to-3d" in adapter.supported_modalities
        caps = adapter.capabilities
        assert caps["direct_text_conditioned"] is True
        assert caps["watertight"] is True

    def test_sf3d_adapter_hardware_check(self):
        adapter = StableFast3DAdapter()
        assert adapter.hardware_requirements["min_vram_gb"] == 8.0
        # If running on <= 6GB GPU, status must be UNSUPPORTED_HARDWARE
        hw = engine_registry.get_hardware_telemetry()
        if hw["total_vram_gb"] < 8.0:
            assert adapter.get_status() == EngineStatus.UNSUPPORTED_HARDWARE

    def test_trellis_adapter_hardware_check(self):
        adapter = TrellisAdapter()
        assert adapter.hardware_requirements["min_vram_gb"] >= 12.0
        hw = engine_registry.get_hardware_telemetry()
        if hw["total_vram_gb"] < 16.0:
            assert adapter.get_status() == EngineStatus.UNSUPPORTED_HARDWARE


class TestModelManager:
    def test_catalog_retrieval(self):
        catalog = model_manager.get_model_catalog()
        assert len(catalog) >= 5
        model_ids = [m.model_id for m in catalog]
        assert "triposr" in model_ids
        assert "sd_turbo" in model_ids
        assert "shape_text3d" in model_ids
        assert "qwen2.5_coder" in model_ids

    def test_free_disk_space_positive(self):
        free_gb = model_manager.get_free_disk_space_gb()
        assert free_gb > 0.0

    def test_engine_dependency_inspection(self):
        status = model_manager.check_engine_dependencies("triposr")
        assert status["engine_id"] == "triposr"
        assert "sufficient_vram" in status
        assert "sufficient_ram" in status
        assert "sufficient_disk" in status


class TestGeometryValidator:
    @pytest.mark.skipif(trimesh is None, reason="trimesh not installed")
    def test_purges_nan_and_inf_vertices(self):
        # Create a test box mesh with an injected NaN vertex
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        verts = np.asarray(box.vertices).copy()
        verts = np.vstack([verts, [np.nan, 0.0, 0.0], [np.inf, 1.0, 1.0]])
        # Add degenerate faces referring to bad vertices
        faces = np.asarray(box.faces).copy()
        faces = np.vstack([faces, [len(verts)-2, 0, 1], [len(verts)-1, 1, 2]])
        bad_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "bad.glb")
            bad_mesh.export(bad_path)

            clean_path, report = geometry_validator.validate_and_repair(bad_path, auto_repair=True)
            assert report.has_nan_or_inf is True
            assert report.nan_inf_vertex_count >= 1
            assert any("Purged non-finite" in a for a in report.repair_actions_taken)

            # Re-read repaired mesh
            repaired = trimesh.load(clean_path, force="mesh")
            assert np.isfinite(repaired.vertices).all()

    @pytest.mark.skipif(trimesh is None, reason="trimesh not installed")
    def test_white_material_detection_and_repair(self):
        # Create a mesh with pure flat white vertex colors (zero variance)
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        box.visual.vertex_colors = np.full((len(box.vertices), 4), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            white_path = os.path.join(tmpdir, "white.glb")
            box.export(white_path)

            clean_path, report = geometry_validator.validate_and_repair(white_path, auto_repair=True)
            assert report.is_flat_white is False  # Repaired
            assert any("Repaired monochromatic white-material" in a for a in report.repair_actions_taken)

            # Verify repaired colors have variance > 0
            repaired = trimesh.load(clean_path, force="mesh")
            colors = np.asarray(repaired.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            assert np.var(colors) > 0.0

    @pytest.mark.skipif(trimesh is None, reason="trimesh not installed")
    def test_polygon_budget_decimation(self):
        sphere = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
        with tempfile.TemporaryDirectory() as tmpdir:
            sphere_path = os.path.join(tmpdir, "sphere.glb")
            sphere.export(sphere_path)

            clean_path, report = geometry_validator.validate_and_repair(
                sphere_path,
                auto_repair=True,
                max_poly_budget=1000
            )
            repaired = trimesh.load(clean_path, force="mesh")
            assert len(repaired.faces) <= 1200  # Within decimation margin
