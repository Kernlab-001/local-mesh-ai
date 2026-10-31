"""
Unit Tests for Geometry & Topology Analysis Engine (Phase 2).
Verifies:
- Metric bounding boxes, dimensions, surface area, and volume.
- Connected component counting.
- Watertightness, boundary edges, and non-manifold edges.
- NaN and Infinity coordinate detection.
- Bilateral symmetry evaluation.
- Orientation PCA diagnostics.
"""

import pytest
import numpy as np
import trimesh
from pathlib import Path
from core.geometry_analyzer import (
    geometry_analyzer,
    CompleteGeometryReport,
    BoundingBox3D,
    TopologyMeasurements
)


def test_cube_geometry_analysis(tmp_path):
    """Verifies exact dimensions, area, and volume for a standard 1m x 2m x 3m box."""
    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    box_path = str(tmp_path / "box_1x2x3.glb")
    box.export(box_path)

    report = geometry_analyzer.analyze_mesh(box_path)

    assert isinstance(report, CompleteGeometryReport)
    assert report.is_valid_mesh is True
    # Check dimensions
    dims = report.bounds.dimensions
    assert pytest.approx(dims[0], abs=1e-3) == 1.0
    assert pytest.approx(dims[1], abs=1e-3) == 2.0
    assert pytest.approx(dims[2], abs=1e-3) == 3.0

    # Volume: 1 * 2 * 3 = 6.0
    assert report.mesh_volume is not None
    assert pytest.approx(report.mesh_volume, abs=1e-3) == 6.0

    # Surface Area: 2*(1*2 + 1*3 + 2*3) = 22.0
    assert pytest.approx(report.surface_area, abs=1e-3) == 22.0

    # Watertightness
    assert report.topology.is_watertight is True
    assert report.topology.boundary_edge_count == 0
    assert report.topology.non_manifold_edge_count == 0
    assert report.topology.degenerate_face_count == 0
    assert report.quality.has_nan_or_inf is False
    assert report.quality.quality_score >= 0.95


def test_nan_and_inf_coordinate_detection(tmp_path):
    """Verifies that non-finite coordinates are flagged and penalized in quality score."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    # Inject NaN and Inf vertices
    verts = mesh.vertices.copy()
    verts[0, 0] = np.nan
    verts[5, 1] = np.inf
    mesh.vertices = verts

    # Save to file
    bad_path = str(tmp_path / "corrupt_nan_inf.obj")
    mesh.export(bad_path)

    report = geometry_analyzer.analyze_mesh(bad_path)
    assert report.quality.has_nan_or_inf is True
    assert report.quality.nan_vertex_count >= 2
    assert report.quality.quality_score < 0.8
    assert any("non-finite" in w.lower() or "nan" in w.lower() for w in report.quality.warnings)


def test_open_mesh_boundary_edge_detection(tmp_path):
    """Verifies open boundaries are detected for non-watertight geometry."""
    # A flat plane (grid) has open boundary edges and is not watertight
    plane = trimesh.creation.box(extents=(2.0, 2.0, 0.1))
    # Remove top face to create an open cavity
    plane.faces = plane.faces[:-2]

    report = geometry_analyzer.analyze_mesh(plane)
    assert report.topology.is_watertight is False
    assert report.topology.boundary_edge_count > 0


def test_bilateral_symmetry_detection(tmp_path):
    """Verifies that symmetric objects have low deviation scores while asymmetric have high."""
    # Symmetric sphere
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere_rep = geometry_analyzer.analyze_mesh(sphere)
    assert sphere_rep.symmetry.is_approximately_symmetric is True
    assert sphere_rep.symmetry.deviation_score <= 0.05

    # Asymmetric sheared object: add unilateral deformations on both +X and +Y
    asym_verts = sphere.vertices.copy()
    asym_verts[asym_verts[:, 0] > 0.2, 0] *= 3.0
    asym_verts[asym_verts[:, 1] > 0.2, 1] *= 3.0
    asym_mesh = trimesh.Trimesh(vertices=asym_verts, faces=sphere.faces)

    asym_rep = geometry_analyzer.analyze_mesh(asym_mesh)
    assert asym_rep.symmetry.deviation_score > 0.08


def test_orientation_pca_elongation():
    """Verifies PCA correctly detects elongation axis for standing vs fallen cylinder."""
    # Tall upright cylinder (Z elongation)
    tall = trimesh.creation.cylinder(radius=0.2, height=3.0)
    tall_rep = geometry_analyzer.analyze_mesh(tall)
    assert tall_rep.orientation.is_upright is True

    # Fallen cylinder along X axis
    fallen = trimesh.creation.cylinder(radius=0.2, height=3.0)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    fallen.apply_transform(rot)
    fallen_rep = geometry_analyzer.analyze_mesh(fallen)
    assert fallen_rep.orientation.estimated_up_axis in ("+X", "-X")
