"""
Regression Test Suite for Material Diagnostic, White-Material Repair, and Scale Calibration.
Specifically validates the 4 distinct cases:
- Case 1: Valid multicolour textures / vertex colors (preserved untouched).
- Case 2: Missing colors and textures (labeled procedural PBR fallback applied).
- Case 3: Intentional monochromatic white assets (e.g. skull, porcelain, marble) preserved without destructive tinting.
- Case 4: Neural extraction flat-white syndrome (monochromatic white detected and repaired with procedural studio shading).
Also validates physical scale calibration (bounding box height scaling) and GLB round-trip integrity.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_validator import geometry_validator, GeometryValidator, GeometryValidationReport

try:
    import trimesh
except ImportError:
    trimesh = None


@pytest.mark.skipif(trimesh is None, reason="trimesh not installed")
class TestMaterialDiagnosticAndRepairCases:

    def test_case_1_valid_multicolour_preserved_untouched(self):
        """Case 1: Multicolour mesh must be detected as HAS_VALID_PALETTE and left untouched."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        n_verts = len(box.vertices)
        # Create distinctly non-white, diverse vertex colors (e.g. blue to red gradient)
        colors = np.zeros((n_verts, 4), dtype=np.uint8)
        colors[:, 0] = np.linspace(50, 220, n_verts, dtype=np.uint8)
        colors[:, 1] = 40
        colors[:, 2] = np.linspace(200, 50, n_verts, dtype=np.uint8)
        colors[:, 3] = 255
        box.visual.vertex_colors = colors

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_path = os.path.join(tmpdir, "multicolour.glb")
            box.export(orig_path)

            clean_path, report = geometry_validator.validate_and_repair(orig_path, auto_repair=True)

            assert report.appearance_status == "HAS_VALID_PALETTE"
            assert report.is_procedural_fallback is False
            assert report.is_flat_white is False
            # Ensure repair actions did NOT touch colors
            assert not any("monochromatic white-material" in a.lower() for a in report.repair_actions_taken)

            # Re-read and verify colors are identical
            reloaded = trimesh.load(clean_path, force="mesh")
            reloaded_colors = np.asarray(reloaded.visual.vertex_colors)
            np.testing.assert_allclose(reloaded_colors[:, :3], colors[:, :3], atol=2)

    def test_case_2_missing_color_data_applies_labeled_fallback(self):
        """Case 2: Geometry without vertex colors or textures gets labeled procedural fallback."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        # Remove any visual attributes
        box.visual = trimesh.visual.ColorVisuals(mesh=box)
        box.visual.vertex_colors = None

        with tempfile.TemporaryDirectory() as tmpdir:
            no_color_path = os.path.join(tmpdir, "nocolor.glb")
            box.export(no_color_path)

            clean_path, report = geometry_validator.validate_and_repair(no_color_path, auto_repair=True)

            assert report.is_procedural_fallback is True
            assert report.appearance_status == "PROCEDURAL_PBR_FALLBACK_APPLIED"
            assert report.fallback_details is not None
            assert any("procedural PBR fallback" in a for a in report.repair_actions_taken)

            # Re-read: mesh now has valid synthesized vertex colors
            reloaded = trimesh.load(clean_path, force="mesh")
            assert reloaded.visual.vertex_colors is not None
            assert len(reloaded.visual.vertex_colors) == len(reloaded.vertices)

    def test_case_3_intentional_monochromatic_white_preserved(self):
        """Case 3: An intentional white asset (e.g. porcelain skull/marble statue) must NOT be tinted."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        # Pure white vertex colors
        box.visual.vertex_colors = np.full((len(box.vertices), 4), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            white_path = os.path.join(tmpdir, "skull.glb")
            box.export(white_path)

            clean_path, report = geometry_validator.validate_and_repair(
                white_path,
                auto_repair=True,
                prompt="ancient weathered white porcelain skull ornament"
            )

            assert report.appearance_status == "INTENTIONAL_MONOCHROMATIC_WHITE"
            assert report.is_procedural_fallback is False
            assert any("Preserved intentional monochromatic white" in w for w in report.warnings)

            # Re-read: vertex colors should still be clean white (mean > 0.95)
            reloaded = trimesh.load(clean_path, force="mesh")
            rgb = np.asarray(reloaded.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            assert np.mean(rgb) > 0.95

    def test_case_4_neural_flat_white_syndrome_repaired_with_labeled_fallback(self):
        """Case 4: An unintended flat-white mesh (e.g. car or weapon) must be repaired and labeled."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        # Pure flat white vertex colors
        box.visual.vertex_colors = np.full((len(box.vertices), 4), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            flat_path = os.path.join(tmpdir, "battleaxe.glb")
            box.export(flat_path)

            clean_path, report = geometry_validator.validate_and_repair(
                flat_path,
                auto_repair=True,
                prompt="medieval glowing battleaxe with carved runes and iron handle"
            )

            assert report.is_procedural_fallback is True
            assert report.appearance_status == "PROCEDURAL_PBR_FALLBACK_APPLIED"
            assert any("Repaired monochromatic white-material" in a for a in report.repair_actions_taken)

            # Re-read: colors now have variation across Z-depth
            reloaded = trimesh.load(clean_path, force="mesh")
            rgb = np.asarray(reloaded.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            assert np.var(rgb) > 0.0001


@pytest.mark.skipif(trimesh is None, reason="trimesh not installed")
class TestPhysicalScaleCalibration:

    def test_scale_calibrated_to_target_height(self):
        """Verifies mesh Z-height is scaled precisely to target canonical height."""
        # Create a unit box (1.0m height)
        box = trimesh.creation.box(extents=(0.5, 0.5, 1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            box_path = os.path.join(tmpdir, "box.glb")
            box.export(box_path)

            # Calibrate to a 0.22m chalice height
            clean_path, report = geometry_validator.validate_and_repair(
                box_path,
                auto_repair=True,
                target_height_m=0.22
            )

            reloaded = trimesh.load(clean_path, force="mesh")
            actual_height = float(reloaded.extents[2])
            assert pytest.approx(actual_height, rel=1e-2) == 0.22
            assert any("Calibrated physical scale" in a for a in report.repair_actions_taken)


@pytest.mark.skipif(trimesh is None, reason="trimesh not installed")
class TestGLBRoundTripIntegrity:

    def test_glb_round_trip_preserves_topology(self):
        """Verifies GLB export and re-import preserves vertex count, face count, and bounds."""
        sphere = trimesh.creation.icosphere(subdivisions=2)
        v_count = len(sphere.vertices)
        f_count = len(sphere.faces)

        with tempfile.TemporaryDirectory() as tmpdir:
            sphere_path = os.path.join(tmpdir, "sphere.glb")
            sphere.export(sphere_path)

            reloaded = trimesh.load(sphere_path, force="mesh")
            assert len(reloaded.vertices) == v_count
            assert len(reloaded.faces) == f_count
            assert np.isfinite(reloaded.vertices).all()
            assert reloaded.is_watertight is True
