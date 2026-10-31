"""
Integration Tests for Phase 2: Component Decomposition & Precision Alignment.
Tests the real geometry pipeline across multiple asset categories:
1. Chair (Genuine TripoSR reconstruction output from Phase 1)
2. Table (Multi-part asset: tabletop + 4 legs)
3. Sword (Multi-part assembly: blade + crossguard + grip + pommel)
4. Single-piece prop (Atomic asset)
5. Headless Blender 5.2 Integration (Verifies real Collections, object metadata, transforms)
"""

import os
import sys
import json
import pytest
import trimesh
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_analyzer import geometry_analyzer
from core.component_decomposer import component_decomposer
from core.precision_aligner import precision_aligner
from core.tool_registry import tool_registry
from core.bounded_correction_loop import bounded_correction_loop
from core.hardware_detector import hardware_detector


# ==============================================================================
# Synthetic Multi-Part Asset Builders for Deterministic Category Testing
# ==============================================================================

def build_table_asset(out_path: str) -> str:
    """Builds a multi-part table with a tabletop and 4 legs."""
    top = trimesh.creation.box(extents=(1.2, 0.8, 0.05))
    top.apply_translation([0, 0, 0.725])

    leg_h = 0.70
    leg_r = 0.03
    legs = []
    offsets = [(-0.5, -0.3), (0.5, -0.3), (-0.5, 0.3), (0.5, 0.3)]
    for ox, oy in offsets:
        leg = trimesh.creation.cylinder(radius=leg_r, height=leg_h)
        leg.apply_translation([ox, oy, leg_h / 2.0])
        legs.append(leg)

    table = top
    for leg in legs:
        table = table + leg

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table.export(out_path)
    return out_path


def build_sword_asset(out_path: str) -> str:
    """Builds a multi-part double-edged sword: blade, crossguard, grip, pommel."""
    # Blade (top)
    blade = trimesh.creation.box(extents=(0.06, 0.015, 0.75))
    blade.apply_translation([0, 0, 0.58])

    # Crossguard (mid)
    guard = trimesh.creation.box(extents=(0.24, 0.035, 0.03))
    guard.apply_translation([0, 0, 0.20])

    # Grip (lower)
    grip = trimesh.creation.cylinder(radius=0.018, height=0.15)
    grip.apply_translation([0, 0, 0.10])

    # Pommel (base)
    pommel = trimesh.creation.icosphere(radius=0.025)
    pommel.apply_translation([0, 0, 0.015])

    sword = blade + guard + grip + pommel
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sword.export(out_path)
    return out_path


def build_single_piece_chalice(out_path: str) -> str:
    """Builds a single-piece continuous chalice/cup."""
    cup = trimesh.creation.cylinder(radius=0.2, height=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cup.export(out_path)
    return out_path


# ==============================================================================
# Multi-Asset Integration Tests
# ==============================================================================

def test_real_triposr_chair_pipeline(tmp_path):
    """
    Executes Phase 2 pipeline on the genuine TripoSR chair mesh generated in Phase 1:
    Analyze -> Decompose -> Precision Align & Ground -> Measure.
    """
    chair_glb = Path("E:/local-mesh-ai-v6.0/output/triposr_1793468726_39502dbb/0/mesh.glb")
    if not chair_glb.exists():
        pytest.skip(f"Phase 1 chair mesh not found at {chair_glb}")

    test_copy = str(tmp_path / "triposr_chair_phase2.glb")
    import shutil
    shutil.copy2(str(chair_glb), test_copy)

    # 1. Geometry Analysis
    initial_rep = geometry_analyzer.analyze_mesh(test_copy)
    assert initial_rep.counts.vertex_count > 10000
    assert initial_rep.counts.face_count > 20000
    assert initial_rep.topology.is_watertight is True

    # 2. Component Decomposition
    hierarchy = component_decomposer.decompose(test_copy, asset_id="triposr_chair")
    assert hierarchy.total_components >= 1
    assert hierarchy.is_decomposed is True

    # 3. Precision Alignment & Grounding to target chair height 0.90m
    aligned_mesh, result = precision_aligner.execute_alignment(
        mesh_or_path=test_copy,
        output_path=test_copy,
        ground_z=True,
        center_xy=True,
        normalize_orientation=True,
        target_height_m=0.90
    )

    assert result.success is True
    assert result.after_is_upright is True
    # Verify exact height
    assert pytest.approx(aligned_mesh.extents[2], abs=0.02) == 0.90
    # Verify ground plane snapping
    assert pytest.approx(aligned_mesh.vertices[:, 2].min(), abs=1e-3) == 0.0

    # 4. Final Verification
    final_rep = geometry_analyzer.analyze_mesh(test_copy)
    assert final_rep.quality.quality_score >= 0.90
    assert final_rep.orientation.is_upright is True


def test_table_multi_part_decomposition_and_alignment(tmp_path):
    """
    Tests decomposition and alignment on a multi-part table.
    """
    table_path = str(tmp_path / "table.glb")
    build_table_asset(table_path)

    # 1. Analyze
    rep = geometry_analyzer.analyze_mesh(table_path)
    assert rep.bounds.dimensions[0] > 1.0

    # 2. Decompose
    hierarchy = component_decomposer.decompose(table_path, asset_id="table_hero", method="connected")
    assert hierarchy.is_decomposed is True
    assert hierarchy.total_components >= 5  # Top + 4 legs

    # 3. Precision Alignment to 0.75m standard table height
    mesh, res = precision_aligner.execute_alignment(
        mesh_or_path=table_path,
        target_height_m=0.75,
        ground_z=True
    )
    assert res.success is True
    assert pytest.approx(mesh.extents[2], abs=0.01) == 0.75


def test_sword_vertical_hierarchy_pipeline(tmp_path):
    """
    Tests decomposition on an elongated weapon assembly (blade, crossguard, grip, pommel).
    """
    sword_path = str(tmp_path / "broadsword.glb")
    build_sword_asset(sword_path)

    # 1. Analyze
    rep = geometry_analyzer.analyze_mesh(sword_path)
    assert rep.bounds.dimensions[2] > 0.8

    # 2. Decompose
    hierarchy = component_decomposer.decompose(sword_path, asset_id="broadsword_01", method="connected")
    assert hierarchy.is_decomposed is True
    assert hierarchy.total_components >= 4

    # 3. Ground and scale to 1.15m length
    mesh, res = precision_aligner.execute_alignment(
        mesh_or_path=sword_path,
        target_height_m=1.15,
        ground_z=True,
        center_xy=True
    )
    assert res.success is True
    assert pytest.approx(mesh.extents[2], abs=0.02) == 1.15
    assert pytest.approx(mesh.vertices[:, 2].min(), abs=1e-3) == 0.0


def test_single_piece_prop_stays_atomic(tmp_path):
    """
    Verifies that a simple continuous single-piece prop is handled cleanly.
    """
    cup_path = str(tmp_path / "chalice.glb")
    build_single_piece_chalice(cup_path)

    hierarchy = component_decomposer.decompose(cup_path, asset_id="chalice_01", method="connected")
    # Single component asset
    assert hierarchy.total_components == 1
    assert hierarchy.is_decomposed is False


def test_real_blender_headless_integration(tmp_path):
    """
    Executes actual headless Blender 5.2 on bridge/blender_hierarchy_bridge.py
    and verifies .blend and .glb outputs exist with non-zero size.
    """
    blender_exe = hardware_detector.find_blender() or r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
    if not os.path.exists(blender_exe):
        pytest.skip(f"Blender executable not found at {blender_exe}")

    test_mesh = str(tmp_path / "blender_input.glb")
    build_sword_asset(test_mesh)

    out_blend = str(tmp_path / "blender_output.blend")
    out_glb = str(tmp_path / "blender_output.glb")
    report_json = str(tmp_path / "blender_report.json")

    bridge_script = str(PROJECT_ROOT / "bridge" / "blender_hierarchy_bridge.py")

    import subprocess
    cmd = [
        blender_exe,
        "--background",
        "--python", bridge_script,
        "--",
        "--mesh", test_mesh,
        "--out-blend", out_blend,
        "--out-glb", out_glb,
        "--report-json", report_json,
        "--height", "1.10"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    assert res.returncode == 0, f"Blender bridge failed: {res.stderr}"

    # Verify outputs exist and are non-empty
    assert os.path.exists(out_blend) and os.path.getsize(out_blend) > 10000
    assert os.path.exists(out_glb) and os.path.getsize(out_glb) > 1000
    assert os.path.exists(report_json)

    with open(report_json, "r", encoding="utf-8") as f:
        rep_data = json.load(f)
    assert rep_data["status"] == "success"
    assert rep_data["master_collection"] == "AI_Asset_Hierarchy"
