"""
Unit Tests for Component Decomposition & Hierarchy (Phase 2).
Verifies:
- Connected component decomposition on multi-part assets.
- Spatial profile / cross-sectional slice decomposition.
- Component hierarchy tree representation and parent-child navigation.
- Provenance tracking (original mesh path preserved).
"""

import pytest
import trimesh
from pathlib import Path
from core.component_decomposer import component_decomposer, ComponentHierarchy, ComponentNode


def test_connected_component_decomposition(tmp_path):
    """Verifies that an asset with discrete sub-parts is cleanly separated."""
    # Create an assembly of 3 disconnected boxes: base, pillar, top
    base = trimesh.creation.box(extents=(2.0, 2.0, 0.2))
    base.apply_translation([0, 0, 0.1])

    pillar = trimesh.creation.cylinder(radius=0.2, height=1.5)
    pillar.apply_translation([0, 0, 1.0])

    top = trimesh.creation.icosphere(radius=0.5)
    top.apply_translation([0, 0, 2.2])

    assembly = base + pillar + top
    mesh_path = str(tmp_path / "three_part_statue.glb")
    assembly.export(mesh_path)

    hierarchy = component_decomposer.decompose(mesh_path, asset_id="statue_01", method="connected")

    assert isinstance(hierarchy, ComponentHierarchy)
    assert hierarchy.is_decomposed is True
    assert hierarchy.total_components >= 3  # Root assembly + 3 parts
    assert hierarchy.hierarchy_depth == 2

    # Verify root assembly links to children
    root_id = hierarchy.root_component_ids[0]
    root_node = hierarchy.get_component(root_id)
    assert root_node is not None
    assert len(root_node.children_ids) == 3

    # Check each child component has proper provenance and bounds
    for cid in root_node.children_ids:
        child = hierarchy.get_component(cid)
        assert child is not None
        assert child.parent_id == root_id
        assert child.face_count > 0
        assert child.bounds.dimensions[0] > 0


def test_spatial_slice_decomposition(tmp_path):
    """Verifies spatial vertical profile slicing on a single continuous mesh."""
    # Create single continuous cylinder
    single_mesh = trimesh.creation.cylinder(radius=0.3, height=3.0)
    mesh_path = str(tmp_path / "continuous_column.glb")
    single_mesh.export(mesh_path)

    hierarchy = component_decomposer.decompose(mesh_path, asset_id="column_01", method="spatial")

    assert isinstance(hierarchy, ComponentHierarchy)
    assert hierarchy.is_decomposed is True
    # Should partition into lower, mid, and upper sections
    assert hierarchy.total_components >= 3

    # Verify relative Z elevation ordering
    child_nodes = hierarchy.get_children(hierarchy.root_component_ids[0])
    z_positions = [c.relative_z_position for c in child_nodes]
    # Check that sections are ordered by height
    assert min(z_positions) < max(z_positions)


def test_provenance_preservation(tmp_path):
    """Verifies that decomposing an asset preserves the immutable source mesh path."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh_path = str(tmp_path / "source_box.glb")
    box.export(mesh_path)

    hierarchy = component_decomposer.decompose(mesh_path, asset_id="box_prov")
    assert hierarchy.provenance_mesh_path == str(Path(mesh_path).resolve())
    assert Path(hierarchy.provenance_mesh_path).exists()
