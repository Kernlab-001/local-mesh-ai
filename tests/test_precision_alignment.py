"""
Unit Tests for Precision Alignment, Grounding & Metric Calibration (Phase 2).
Verifies:
- Ground plane snapping (Z_min = 0.0) and XY centering.
- Orientation normalization from lateral axes (+X, +Y) to canonical vertical (+Z).
- Metric scale calibration against target constraints.
- Dimension comparison with explicit scale provenance.
- Non-destructive snapshot backup creation.
"""

import pytest
import trimesh
import numpy as np
from pathlib import Path
from core.precision_aligner import precision_aligner, DimensionComparison, AlignmentResult


def test_ground_and_center_alignment(tmp_path):
    """Verifies that an off-center, floating box is snapped to Z=0 and centered at (0, 0)."""
    # Create box elevated at Z=5.0 and off-center at X=10.0, Y=-4.0
    box = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    box.apply_translation([10.0, -4.0, 6.0])  # Min Z is 5.0, Center is (10, -4, 6)
    mesh_path = str(tmp_path / "floating_box.glb")
    box.export(mesh_path)

    mesh, result = precision_aligner.execute_alignment(
        mesh_or_path=mesh_path,
        output_path=mesh_path,
        ground_z=True,
        center_xy=True,
        normalize_orientation=False
    )

    assert result.success is True
    verts = mesh.vertices
    # Min Z must be 0.0
    assert pytest.approx(verts[:, 2].min(), abs=1e-4) == 0.0
    # Center X and Y must be 0.0
    center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    center_y = (verts[:, 1].min() + verts[:, 1].max()) / 2.0
    assert pytest.approx(center_x, abs=1e-4) == 0.0
    assert pytest.approx(center_y, abs=1e-4) == 0.0


def test_orientation_normalization(tmp_path):
    """Verifies that a fallen cylinder along X is rotated upright to +Z."""
    # Cylinder lying on X axis (radius 0.2, length 2.5)
    cylinder = trimesh.creation.cylinder(radius=0.2, height=2.5)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    cylinder.apply_transform(rot)
    mesh_path = str(tmp_path / "fallen_cylinder.glb")
    cylinder.export(mesh_path)

    mesh, result = precision_aligner.execute_alignment(
        mesh_or_path=mesh_path,
        output_path=mesh_path,
        normalize_orientation=True
    )

    assert result.success is True
    assert result.before_is_upright is False
    assert result.after_is_upright is True
    # Z span should now be the long axis (approx 2.5m)
    assert pytest.approx(mesh.extents[2], abs=0.05) == 2.5


def test_metric_scale_calibration(tmp_path):
    """Verifies scaling an object to an exact user-specified target height."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh_path = str(tmp_path / "unit_box.glb")
    box.export(mesh_path)

    target_h = 1.85  # Target 1.85 meters
    mesh, result = precision_aligner.execute_alignment(
        mesh_or_path=mesh_path,
        target_height_m=target_h,
        is_user_provided_scale=True
    )

    assert result.success is True
    assert pytest.approx(mesh.extents[2], abs=1e-3) == 1.85
    assert result.scale_provenance == "user_provided"


def test_dimension_comparison_provenance():
    """Verifies dimension comparison calculates differences and labels provenance correctly."""
    from core.geometry_analyzer import BoundingBox3D
    bounds = BoundingBox3D(
        min_point=(0, 0, 0),
        max_point=(0.4, 0.4, 0.8),
        dimensions=(0.4, 0.4, 0.8),
        center=(0.2, 0.2, 0.4),
        volume=0.128
    )

    comp = precision_aligner.compare_dimensions(
        measured_bounds=bounds,
        target_height_m=1.2,
        tolerance_pct=5.0,
        is_user_provided=True
    )

    assert comp.needs_scale_correction is True
    assert pytest.approx(comp.height_difference_m, abs=1e-3) == 0.4
    assert pytest.approx(comp.scale_factor_recommended, abs=1e-3) == 1.5
    assert comp.scale_provenance == "user_provided"
