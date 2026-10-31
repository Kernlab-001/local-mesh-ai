"""
Unit Tests for Phase 3 Geometry Construction Engine.
Tests parametric primitive generation:
- Cube, UV Sphere, Icosphere, Cylinder, Cone, Torus, Plane, Circle
- Custom extruded 2D profile
Verifies watertightness, metric extents, and deterministic reproducibility.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_construction import GeometryConstructionEngine, ConstructPrimitiveArgs


def test_cube_construction():
    """Verifies cube creation with exact metric extents."""
    cube = GeometryConstructionEngine.create_cube(extents=(1.5, 2.0, 0.8), center=(0, 0, 0.4))
    assert cube.is_watertight
    assert len(cube.vertices) == 8
    assert len(cube.faces) == 12
    np.testing.assert_allclose(cube.extents, [1.5, 2.0, 0.8], atol=1e-4)
    np.testing.assert_allclose(cube.bounds[0], [-0.75, -1.0, 0.0], atol=1e-4)


def test_spheres_construction():
    """Verifies UV sphere and icosphere creation."""
    uv = GeometryConstructionEngine.create_uv_sphere(radius=0.75, segments=16, rings=8)
    assert uv.is_watertight
    np.testing.assert_allclose(uv.extents, [1.5, 1.5, 1.5], atol=0.05)

    ico = GeometryConstructionEngine.create_icosphere(radius=0.5, subdivisions=2)
    assert ico.is_watertight
    assert len(ico.faces) == 320


def test_cylinder_and_cone_construction():
    """Verifies cylinder and cone creation."""
    cyl = GeometryConstructionEngine.create_cylinder(radius=0.3, height=1.2, sections=24)
    assert cyl.is_watertight
    np.testing.assert_allclose(cyl.extents[2], 1.2, atol=1e-3)

    cone = GeometryConstructionEngine.create_cone(radius=0.4, height=1.0, sections=24)
    assert cone.is_watertight
    np.testing.assert_allclose(cone.extents[2], 1.0, atol=1e-3)


def test_torus_and_plane_construction():
    """Verifies torus and plane construction."""
    torus = GeometryConstructionEngine.create_torus(major_radius=0.8, minor_radius=0.2)
    assert torus.is_watertight
    np.testing.assert_allclose(torus.extents[0], 2.0, atol=0.05)

    plane = GeometryConstructionEngine.create_plane(size_x=2.0, size_y=1.5)
    assert not plane.is_watertight  # 2D open surface
    np.testing.assert_allclose(plane.extents[:2], [2.0, 1.5], atol=1e-3)


def test_custom_profile_extrusion():
    """Verifies 2D polygon profile extrusion along Z."""
    pts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [0.0, 0.8], [-0.5, 0.5]]
    poly_mesh = GeometryConstructionEngine.create_extruded_profile(points_2d=pts, height=0.6)
    assert poly_mesh.is_watertight
    np.testing.assert_allclose(poly_mesh.extents[2], 0.6, atol=1e-3)


def test_construct_from_args_factory(tmp_path):
    """Verifies construction through typed Pydantic ConstructPrimitiveArgs."""
    out_file = str(tmp_path / "factory_cylinder.glb")
    args = ConstructPrimitiveArgs(
        primitive_type="cylinder",
        radius=0.25,
        height=1.5,
        output_path=out_file
    )
    mesh = GeometryConstructionEngine.construct_from_args(args)
    assert mesh.is_watertight
    assert os.path.exists(out_file)
