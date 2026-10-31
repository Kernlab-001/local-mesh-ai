"""
Unit Tests for Phase 3 Geometry Editing Engine.
Tests parameterized mesh modifications:
- Translation, rotation, uniform scale, single-axis scale with anchor
- Mirroring, array generation, subdivision, solidify, cleanup
- Component-level targeted editing with invariant preservation
"""

import os
import sys
import pytest
import numpy as np
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_construction import GeometryConstructionEngine
from core.geometry_editing import GeometryEditingEngine, EditComponentArgs


def test_transformations():
    """Verifies translation, rotation, and uniform scale."""
    cube = GeometryConstructionEngine.create_cube((1.0, 1.0, 1.0))

    # Translation
    translated = GeometryEditingEngine.translate_mesh(cube, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(translated.bounds[0], [1.5, -0.5, 0.5], atol=1e-4)

    # Scaling
    scaled = GeometryEditingEngine.scale_mesh(cube, 2.0)
    np.testing.assert_allclose(scaled.extents, [2.0, 2.0, 2.0], atol=1e-4)

    # Rotation (90 deg around Z)
    rotated = GeometryEditingEngine.rotate_mesh(cube, [0, 0, 90])
    np.testing.assert_allclose(rotated.extents, [1.0, 1.0, 1.0], atol=1e-4)


def test_single_axis_scaling_with_anchor():
    """Verifies scaling along Z with min anchor (e.g. making backrest 20% taller from base)."""
    cube = GeometryConstructionEngine.create_cube((1.0, 1.0, 1.0), center=(0, 0, 0.5))  # Z: [0, 1]
    scaled = GeometryEditingEngine.scale_component_axis(cube, axis="z", factor=1.5, anchor="min")

    # Z min remains at 0.0, Z max reaches 1.5
    assert abs(scaled.bounds[0][2] - 0.0) < 1e-4
    assert abs(scaled.bounds[1][2] - 1.5) < 1e-4
    # X and Y remain unchanged
    np.testing.assert_allclose(scaled.extents[:2], [1.0, 1.0], atol=1e-4)


def test_mirror_and_array_generation():
    """Verifies bilateral mirroring and linear array generation."""
    box = GeometryConstructionEngine.create_cube((0.5, 0.5, 0.5), center=(1.0, 0.0, 0.25))

    # Mirror across X plane (x=0)
    mirrored = GeometryEditingEngine.mirror_mesh(box, plane="x", plane_offset=0.0, duplicate=True)
    assert len(mirrored.faces) == len(box.faces) * 2
    np.testing.assert_allclose(mirrored.bounds[0][0], -1.25, atol=1e-4)
    np.testing.assert_allclose(mirrored.bounds[1][0], 1.25, atol=1e-4)

    # Array along Y (3 copies spaced by 1.0m)
    array_mesh = GeometryEditingEngine.array_mesh(box, count=3, offset_vector=[0.0, 1.0, 0.0])
    assert len(array_mesh.faces) == len(box.faces) * 3
    assert abs(array_mesh.extents[1] - 2.5) < 1e-4


def test_subdivision_and_solidify():
    """Verifies surface subdivision and solidify thickness."""
    plane = GeometryConstructionEngine.create_plane(1.0, 1.0)
    assert not plane.is_watertight

    # Solidify open plane into watertight slab
    solid = GeometryEditingEngine.solidify_mesh(plane, thickness=0.05)
    assert solid.is_watertight
    assert abs(solid.extents[2] - 0.05) < 1e-4

    # Subdivide cube
    cube = GeometryConstructionEngine.create_cube((1.0, 1.0, 1.0))
    sub = GeometryEditingEngine.subdivide_mesh(cube, count=1)
    assert len(sub.faces) == len(cube.faces) * 4


def test_component_level_targeted_editing(tmp_path):
    """
    Verifies targeted component edit: modifies seat component while
    guaranteeing that non-target components remain strictly unchanged.
    """
    c1 = GeometryConstructionEngine.create_cube((0.8, 0.8, 0.1), center=(0, 0, 0.4))
    c2 = GeometryConstructionEngine.create_cube((0.8, 0.1, 0.6), center=(0, 0.35, 0.75))
    chair = trimesh.util.concatenate([c1, c2])

    test_file = str(tmp_path / "test_chair.glb")
    out_file = str(tmp_path / "edited_chair.glb")
    chair.export(test_file)

    args = EditComponentArgs(
        mesh_path=test_file,
        operation="scale_axis",
        parameters={"axis": "z", "factor": 1.3, "anchor": "min"},
        output_path=out_file
    )

    res = GeometryEditingEngine.execute_component_edit(args)
    assert res.status == "success"
    assert res.non_target_components_preserved
    assert os.path.exists(out_file)

    edited_mesh = trimesh.load(out_file, force="mesh")
    # Verify overall height increased by ~30%
    assert edited_mesh.extents[2] > chair.extents[2]
