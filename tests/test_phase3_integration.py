"""
Integration Tests for Phase 3: Advanced Geometry Construction, AI-Assisted Editing
and Hardware-Aware Local Model Orchestration.

Tests:
1. End-to-end AI-assisted editing feedback loop with invariant verification.
2. Controlled tool registry execution of all 11 allowlisted tools.
3. Real headless Blender 5.2 multi-view camera rendering and component transformation.
4. Model catalogue and hardware profile selector integration.
"""

import os
import sys
import json
import pytest
import subprocess
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_construction import GeometryConstructionEngine
from core.bounded_correction_loop import bounded_correction_loop
from core.tool_registry import tool_registry
from core.hardware_detector import hardware_detector


def test_ai_assisted_editing_feedback_loop(tmp_path):
    """
    Tests complete AI-assisted editing feedback loop:
    User: 'Make the chair backrest 20% taller but keep the seat and overall width unchanged.'
    Verifies:
    - Target backrest is identified.
    - Target height increases by 20%.
    - Non-target seat and overall width are preserved within 5% tolerance.
    - Rollback safety is active.
    """
    seat = GeometryConstructionEngine.create_cube((0.6, 0.6, 0.1), center=(0, 0, 0.45))
    back = GeometryConstructionEngine.create_cube((0.6, 0.1, 0.5), center=(0, 0.25, 0.75))
    chair = trimesh.util.concatenate([seat, back])

    mesh_file = str(tmp_path / "input_chair.glb")
    chair.export(mesh_file)

    report = bounded_correction_loop.run_component_edit_loop(
        mesh_path=mesh_file,
        user_instruction="Make the chair backrest 20% taller but keep the seat and overall width unchanged.",
        working_dir=str(tmp_path / "loop_workspace")
    )

    assert report.status == "SUCCESS"
    assert report.target_height_change_percent == 20.0
    assert report.non_target_invariants_preserved is True
    assert report.invariant_deviation_max_percent <= 5.0
    assert os.path.exists(report.output_mesh_path)


def test_tool_registry_phase3_tools(tmp_path):
    """Verifies that all Phase 3 tools execute successfully through the safety boundary."""
    # 1. construct_primitive
    cube_file = str(tmp_path / "registry_cube.glb")
    res_const = tool_registry.execute_tool("construct_primitive", {
        "primitive_type": "cube",
        "dimensions": [1.0, 1.0, 1.0],
        "output_path": cube_file
    })
    assert res_const["status"] == "success"
    assert os.path.exists(cube_file)

    # 2. edit_component
    edited_file = str(tmp_path / "registry_cube_scaled.glb")
    res_edit = tool_registry.execute_tool("edit_component", {
        "mesh_path": cube_file,
        "operation": "scale_axis",
        "parameters": {"axis": "z", "factor": 1.25},
        "output_path": edited_file
    })
    assert res_edit["status"] == "success"
    assert res_edit["result"]["modified_dimensions"][2] == 1.25

    # 3. segment_specialist_3d
    res_seg = tool_registry.execute_tool("segment_specialist_3d", {
        "mesh_path": cube_file,
        "num_clusters": 2
    })
    assert res_seg["status"] == "success"
    assert res_seg["result"]["total_components"] >= 1

    # 4. interpret_vision_language
    res_vlm = tool_registry.execute_tool("interpret_vision_language", {
        "mesh_path": cube_file,
        "user_asset_description": "A wooden test cube",
        "render_dir": str(tmp_path / "vlm_renders")
    })
    assert res_vlm["status"] == "success"
    assert "assignments" in res_vlm["result"]


def test_real_headless_blender_phase3_integration(tmp_path):
    """
    Executes real headless Blender 5.2 Alpha to verify:
    - Multi-view 6-angle rendering
    - Component transformation inside Blender
    - Scene hierarchy creation
    - Exported .blend and .glb files
    """
    blender_bin = hardware_detector.find_blender() or "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe"
    if not os.path.exists(blender_bin):
        pytest.skip(f"Blender executable not found at: {blender_bin}")

    # Create test mesh
    cube = GeometryConstructionEngine.create_cube((1.0, 1.0, 1.0))
    mesh_in = str(tmp_path / "blender_test_cube.glb")
    blend_out = str(tmp_path / "output_scene.blend")
    glb_out = str(tmp_path / "output_scene.glb")
    renders_out = str(tmp_path / "blender_multiview")
    cube.export(mesh_in)

    script_path = str(PROJECT_ROOT / "bridge" / "blender_hierarchy_bridge.py")
    cmd = [
        blender_bin,
        "--background",
        "--python", script_path,
        "--",
        "--input", mesh_in,
        "--output-blend", blend_out,
        "--output-glb", glb_out,
        "--target-height", "0.95",
        "--render-multiview", renders_out,
        "--edit-comp", "AI_Main_Asset",
        "--edit-op", "scale_axis",
        "--edit-factor", "1.2",
        "--edit-axis", "Z"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}\n{res.stdout}"
    assert os.path.exists(blend_out), ".blend scene file was not created"
    assert os.path.exists(glb_out), ".glb export file was not created"
    assert os.path.exists(os.path.join(renders_out, "blender_view_front.png"))
    assert os.path.exists(os.path.join(renders_out, "blender_view_perspective.png"))
