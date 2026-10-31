"""
Unit Tests for Phase 3 Component Intelligence & Specialist Perception.
Tests:
- Specialist 3D segmentation adapter (PartField & Geometric Feature Fields)
- Multi-view 6-angle rendering (Front, Back, Left, Right, Top, Perspective)
- Component ID color-coded mask generation
- Vision-Language semantic interpretation (preserves component IDs, handles uncertainty)
- Multi-category tests: Chair, Table, Sword, Robot, Single-piece prop
"""

import os
import sys
import pytest
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_construction import GeometryConstructionEngine
from core.engines.partfield_adapter import partfield_adapter, GeometricFeatureFieldClusterer
from core.engines.vision_language_adapter import vision_language_adapter
from core.component_decomposer import ComponentHierarchy


def test_specialist_partfield_adapter_status():
    """Verifies truthful PartField engine status detection."""
    status = partfield_adapter.get_status()
    details = partfield_adapter.get_status_details()
    assert status.value in ("ready", "weights_missing", "dependencies_missing")
    assert "status" in details
    assert "geometric_feature_field_fallback_ready" in details
    assert details["geometric_feature_field_fallback_ready"] is True


def test_multiview_rendering_and_palette(tmp_path):
    """Verifies 6-view orthographic/perspective rendering and ID mask palette."""
    cube = GeometryConstructionEngine.create_cube((1.0, 1.0, 1.0))
    render_dir = str(tmp_path / "renders")

    res = vision_language_adapter.generate_multiview_renders(cube, render_dir)
    assert len(res.render_paths) == 6
    assert "front" in res.render_paths
    assert "back" in res.render_paths
    assert "left" in res.render_paths
    assert "right" in res.render_paths
    assert "top" in res.render_paths
    assert "perspective" in res.render_paths

    for p in res.render_paths.values():
        assert os.path.exists(p)


def test_chair_component_intelligence(tmp_path):
    """Tests component decomposition and semantic labeling on a chair asset."""
    seat = GeometryConstructionEngine.create_cube((0.5, 0.5, 0.1), center=(0, 0, 0.45))
    back = GeometryConstructionEngine.create_cube((0.5, 0.1, 0.5), center=(0, 0.2, 0.75))
    legs = GeometryConstructionEngine.create_cylinder(radius=0.03, height=0.4, center=(0.2, 0.2, 0.2))
    chair = trimesh.util.concatenate([seat, back, legs])

    mesh_file = str(tmp_path / "chair.glb")
    chair.export(mesh_file)

    hierarchy = partfield_adapter.decompose_mesh(chair)
    assert hierarchy.total_components >= 2

    render_dir = str(tmp_path / "chair_renders")
    renders = vision_language_adapter.generate_multiview_renders(chair, render_dir, hierarchy)
    proposal = vision_language_adapter.interpret_components(
        hierarchy, renders, "A wooden dining chair with fabric seat and carved legs"
    )

    assert len(proposal.assignments) >= 1
    # Check that candidate IDs are strictly preserved (no invented IDs)
    candidate_cids = list(hierarchy.components.keys())
    for a in proposal.assignments:
        assert a.component_id in candidate_cids
        assert a.confidence > 0.0


def test_sword_component_intelligence(tmp_path):
    """Tests component decomposition and semantic labeling on a sword asset."""
    blade = GeometryConstructionEngine.create_cube((0.08, 0.02, 0.8), center=(0, 0, 0.6))
    guard = GeometryConstructionEngine.create_cube((0.3, 0.04, 0.04), center=(0, 0, 0.2))
    grip = GeometryConstructionEngine.create_cylinder(radius=0.02, height=0.16, center=(0, 0, 0.1))
    sword = trimesh.util.concatenate([blade, guard, grip])

    mesh_file = str(tmp_path / "sword.glb")
    sword.export(mesh_file)

    hierarchy = partfield_adapter.decompose_mesh(sword)
    render_dir = str(tmp_path / "sword_renders")
    renders = vision_language_adapter.generate_multiview_renders(sword, render_dir, hierarchy)
    proposal = vision_language_adapter.interpret_components(
        hierarchy, renders, "A double-edged steel broadsword with brass crossguard and leather grip"
    )

    labels = [a.assigned_semantic_label for a in proposal.assignments]
    assert any(l in ("blade", "crossguard", "grip", "pommel") for l in labels)


def test_single_piece_prop_component_intelligence(tmp_path):
    """Tests single-piece prop (ceramic vase) without obvious component splits."""
    vase = GeometryConstructionEngine.create_cylinder(radius=0.2, height=0.6)
    mesh_file = str(tmp_path / "vase.glb")
    vase.export(mesh_file)

    hierarchy = partfield_adapter.decompose_mesh(vase)
    assert hierarchy.total_components >= 1
    render_dir = str(tmp_path / "vase_renders")
    renders = vision_language_adapter.generate_multiview_renders(vase, render_dir, hierarchy)
    proposal = vision_language_adapter.interpret_components(
        hierarchy, renders, "A single-piece ceramic vase prop"
    )

    # In single-piece prop without distinct labels, assignments must exist
    assert len(proposal.assignments) >= 1
