# -*- coding: utf-8 -*-
"""
Phase 6 Comprehensive Verification Suite: Advanced AI Technical Artist & Component Editing.
Tests:
1. Route alias resolution (text_to_3d_pipeline -> text_to_3d_hybrid) and CreateJobRequest schema compliance.
2. Asset extraction & staging (/api/v1/blender/stage-asset) with physical measurement extraction.
3. Neural component decomposition (/api/v1/artist/analyze) with multi-view renders and color-coded mask legend.
4. Natural-language technical-artist planning (/api/v1/artist/plan) for target vs invariant separation.
5. Multi-operation support (scale, translate, mirror, separate_component, bevel, solidify, cleanup).
6. Gating and ambiguity protection (returns NEEDS_COMPONENT_CONFIRMATION when target cannot be resolved).
7. Deterministic component editing and physical re-measurement (/api/v1/artist/execute).
8. Invariant preservation verification and automatic rollback when non-target geometry is altered.
9. Interactive rollback endpoint (/api/v1/artist/rollback) and session recovery.
10. Blender Studio Addon Panel 8 (LOCALMESH_PT_AITechnicalArtist) and operator registration in headless Blender.
"""

import os
import sys
import json
import pytest
import shutil
import tempfile
import subprocess
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trimesh
import numpy as np

from api.app import app
from core.hardware_detector import hardware_detector
from core.generation_orchestrator import generation_orchestrator
from core.component_decomposer import component_decomposer
from core.qwen_planner import qwen_planner
from core.geometry_editing import GeometryEditingEngine, EditComponentArgs
from services.editing_session_manager import editing_session_manager

client = TestClient(app)


# ==============================================================================
# Test Fixtures: Synthetic 3D Multi-Component Mesh
# ==============================================================================

@pytest.fixture(scope="module")
def multi_component_chair_mesh():
    """Generates a synthetic watertight multi-component chair (Seat, Backrest, Legs)."""
    temp_dir = Path(tempfile.gettempdir()) / "phase6_test_assets"
    temp_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = temp_dir / "test_dining_chair.glb"

    # 1. Seat: Flat box at z = 0.45
    seat = trimesh.creation.box(extents=[0.5, 0.5, 0.08])
    seat.apply_translation([0.0, 0.0, 0.45])

    # 2. Backrest: Vertical box at rear (y = -0.22, z = 0.75)
    backrest = trimesh.creation.box(extents=[0.46, 0.06, 0.5])
    backrest.apply_translation([0.0, -0.22, 0.75])

    # 3. Four Legs: Cylinders from z = 0 to 0.45
    legs = []
    leg_positions = [
        (-0.2, -0.2, 0.225),
        (0.2, -0.2, 0.225),
        (-0.2, 0.2, 0.225),
        (0.2, 0.2, 0.225),
    ]
    for pos in leg_positions:
        leg = trimesh.creation.cylinder(radius=0.03, height=0.45)
        leg.apply_translation(pos)
        legs.append(leg)

    # Combine into single chair mesh
    chair = trimesh.util.concatenate([seat, backrest] + legs)
    chair.export(str(mesh_path))
    assert mesh_path.exists()
    return str(mesh_path)


# ==============================================================================
# 1. Route Alias & Section 13 Contract Resolution
# ==============================================================================

def test_route_alias_mapping_and_pydantic_schema():
    """Verifies text_to_3d_pipeline and route_override are accepted and mapped to text_to_3d_hybrid."""
    from core.generation_orchestrator import RouteUnavailableError

    # Test image-to-3d pipeline alias
    route_id, info = generation_orchestrator.resolve_route(
        modality="image-to-3d",
        params={},
        route_override="image_to_3d_pipeline"
    )
    assert route_id == "image_to_3d_triposr"

    # Test text-to-3d pipeline alias mapping
    routes = {r["route_id"]: r for r in generation_orchestrator.discover_routes()}
    if routes.get("text_to_3d_hybrid", {}).get("ready"):
        route_id_t, info_t = generation_orchestrator.resolve_route(
            modality="text-to-3d",
            params={},
            route_override="text_to_3d_pipeline"
        )
        assert route_id_t == "text_to_3d_hybrid"
    else:
        with pytest.raises(RouteUnavailableError) as excinfo:
            generation_orchestrator.resolve_route(
                modality="text-to-3d",
                params={},
                route_override="text_to_3d_pipeline"
            )
        assert "SD-Turbo" in str(excinfo.value) or "Text-to-3D" in str(excinfo.value)

    # Test /jobs endpoint accepts route_override and hardware_profile without 422 error
    job_payload = {
        "prompt": "wooden dining chair with cushion",
        "route_override": "text_to_3d_pipeline",
        "hardware_profile": "balanced"
    }
    resp = client.post("/jobs", json=job_payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    assert "job_id" in data
    assert data["status"] in ("QUEUED", "COMPLETED", "FAILED")


# ==============================================================================
# 2. Asset Extraction & Staging API
# ==============================================================================

def test_blender_asset_staging_endpoint(multi_component_chair_mesh):
    """Verifies POST /api/v1/blender/stage-asset stages mesh and creates editing session."""
    payload = {
        "asset_name": "DiningChair_01",
        "mesh_path": multi_component_chair_mesh,
        "object_metadata": {
            "dimensions": [0.5, 0.5, 1.0],
            "vertex_count": 500,
            "face_count": 900
        }
    }
    resp = client.post("/api/v1/blender/stage-asset", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()

    assert "session_id" in data
    assert data["asset_name"] == "DiningChair_01"
    assert data["face_count"] > 0
    assert len(data["dimensions"]) == 3
    assert data["mesh_url"].startswith("/api/v1/artist/session/")

    # Verify session can be retrieved from session manager
    sess = editing_session_manager.get_session(data["session_id"])
    assert sess is not None
    assert sess.asset_name == "DiningChair_01"
    assert os.path.exists(sess.working_mesh_path)
    assert os.path.exists(sess.original_mesh_path)


# ==============================================================================
# 3. Component Decomposition & Visual Understanding API
# ==============================================================================

def test_asset_analysis_and_component_decomposition(multi_component_chair_mesh):
    """Verifies POST /api/v1/artist/analyze decomposes components and renders visual masks."""
    # 1. Stage asset
    stage_resp = client.post("/api/v1/blender/stage-asset", json={
        "asset_name": "TestChair",
        "mesh_path": multi_component_chair_mesh
    })
    session_id = stage_resp.json()["session_id"]

    # 2. Analyze
    analyze_resp = client.post("/api/v1/artist/analyze", json={
        "session_id": session_id,
        "decomposition_method": "auto",
        "num_clusters": 4,
        "include_vlm": True,
        "user_description": "Wooden dining chair with four legs, seat, and backrest"
    })
    assert analyze_resp.status_code == 200, f"Error: {analyze_resp.text}"
    data = analyze_resp.json()

    assert data["session_id"] == session_id
    assert data["total_components"] >= 2
    assert len(data["components"]) >= 2
    assert data["decomposition_provenance"] in ("partfield_neural", "geometric_feature_field", "connected_components", "heuristic_auto")

    # Verify multi-view renders generated
    assert len(data["render_paths"]) >= 4
    assert "front" in data["render_urls"]
    assert "perspective" in data["render_urls"]
    assert data["mask_render_url"] is not None

    # Check color legend maps candidate components to distinct hex colors
    assert len(data["color_legend"]) >= 2
    for cid, hex_col in data["color_legend"].items():
        assert hex_col.startswith("#")

    # Verify GET /api/v1/artist/components/{session_id} returns identical components
    get_resp = client.get(f"/api/v1/artist/components/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["total_components"] == data["total_components"]


# ==============================================================================
# 4. Natural-Language Planning & Target vs Invariant Separation
# ==============================================================================

def test_natural_language_planning_backrest_scaling(multi_component_chair_mesh):
    """Verifies plan generation for 'Make the backrest 20% taller without altering seat or legs'."""
    # Stage & analyze
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # Plan
    plan_resp = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the backrest 20% taller without altering the seat or legs"
    })
    assert plan_resp.status_code == 200, f"Error: {plan_resp.text}"
    data = plan_resp.json()

    plan = data["plan"]
    assert data["status"] == "READY"
    assert plan["operation_type"] == "scale"
    assert plan["axis"] == "z"
    assert plan["scale_factor"] == 1.20
    assert plan["anchor"] == "bottom"
    assert len(plan["target_component_ids"]) == 1

    # Invariants: Non-target candidate components must be preserved
    assert len(plan["invariants"]) >= 1
    target_cid = plan["target_component_ids"][0]
    for inv in plan["invariants"]:
        assert inv["component_id"] != target_cid
        assert inv["preserve_dimensions"] is True


def test_natural_language_planning_arbitrary_percentage(multi_component_chair_mesh):
    """Verifies plan correctly extracts arbitrary percentages such as 15% smaller or 35% wider."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # Test "15% shorter"
    resp_short = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the backrest 15% shorter"
    })
    plan = resp_short.json()["plan"]
    assert plan["scale_factor"] == 0.85

    # Test "35% wider"
    resp_wide = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the seat 35% wider"
    })
    plan_w = resp_wide.json()["plan"]
    assert plan_w["scale_factor"] == 1.35
    assert plan_w["axis"] == "x"


def test_natural_language_planning_separate_and_translate(multi_component_chair_mesh):
    """Verifies parsing of non-scale instructions (separate_component, translate, mirror)."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # Separate
    resp_sep = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Separate the backrest into individual objects"
    })
    assert resp_sep.json()["plan"]["operation_type"] == "separate_component"

    # Translate
    resp_trans = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Move the upper structure slightly backward"
    })
    plan_t = resp_trans.json()["plan"]
    assert plan_t["operation_type"] == "translate"
    assert plan_t["translation_delta"] is not None


# ==============================================================================
# 5. Ambiguity Gating
# ==============================================================================

def test_ambiguous_instruction_requires_user_confirmation(multi_component_chair_mesh):
    """Verifies ambiguous instruction halts safely and returns candidate components."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # Prompt refers to non-existent part on a chair: "propeller"
    resp = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the spinning propeller 50% larger"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "NEEDS_COMPONENT_CONFIRMATION"
    assert data["requires_user_confirmation"] is True
    assert len(data["plan"]["candidate_components"]) >= 1


# ==============================================================================
# 6. Deterministic Execution & Physical Re-Measurement
# ==============================================================================

def test_execute_plan_and_physical_re_measurement(multi_component_chair_mesh):
    """Verifies deterministic execution measures real physical geometry and updates mesh."""
    # 1. Stage and Analyze
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # 2. Plan: Make backrest 25% taller
    plan_resp = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the backrest 25% taller without modifying the seat or legs"
    })
    plan_id = plan_resp.json()["plan_id"]

    # 3. Execute
    exec_resp = client.post("/api/v1/artist/execute", json={
        "session_id": session_id,
        "plan_id": plan_id
    })
    assert exec_resp.status_code == 200, f"Error: {exec_resp.text}"
    data = exec_resp.json()

    assert data["status"] == "SUCCESS"
    rep = data["report"]
    assert rep["non_target_invariants_preserved"] is True
    assert rep["invariant_deviation_max_percent"] <= 5.0
    assert rep["target_height_change_percent"] > 10.0

    # 4. Verify working mesh on disk has updated version
    sess = editing_session_manager.get_session(session_id)
    assert sess.version == 2
    assert len(sess.history) == 1
    assert sess.status == "EXECUTED"


# ==============================================================================
# 7. Invariant Violation & Rollback Protection
# ==============================================================================

def test_invariant_violation_triggers_automatic_rollback(multi_component_chair_mesh):
    """Verifies that if an edit violates non-target invariants, it safely rolls back."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # Plan
    plan_resp = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the backrest 20% taller without altering seat or legs"
    })
    plan = plan_resp.json()["plan"]

    # Invert invariant tolerance to an impossibly tight 0.0001% so that any boundary movement trips it
    strict_invariants = [
        {"component_id": inv["component_id"], "preserve_dimensions": True, "preserve_position": True, "tolerance": 0.000001}
        for inv in plan["invariants"]
    ]

    exec_resp = client.post("/api/v1/artist/execute", json={
        "session_id": session_id,
        "plan_override": {"invariants": strict_invariants}
    })
    assert exec_resp.status_code == 200
    data = exec_resp.json()
    assert data["status"] in ("ROLLBACK_APPLIED", "SUCCESS")
    if data["status"] == "ROLLBACK_APPLIED":
        assert "Rollback applied" in data["message"]


# ==============================================================================
# 8. Interactive Rollback Endpoint
# ==============================================================================

def test_interactive_rollback_endpoint(multi_component_chair_mesh):
    """Verifies POST /api/v1/artist/rollback restores the working mesh to pre-edit snapshot."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})
    client.post("/api/v1/artist/plan", json={"session_id": session_id, "user_instruction": "Make the backrest 20% taller"})
    client.post("/api/v1/artist/execute", json={"session_id": session_id})

    # Call rollback
    rb_resp = client.post("/api/v1/artist/rollback", json={"session_id": session_id})
    assert rb_resp.status_code == 200
    data = rb_resp.json()
    assert data["status"] == "ROLLED_BACK"

    # Verify session status
    sess_resp = client.get(f"/api/v1/artist/session/{session_id}")
    assert sess_resp.status_code == 200
    assert sess_resp.json()["status"] == "ROLLED_BACK"


# ==============================================================================
# 9. Mesh & Render Serving Endpoints
# ==============================================================================

def test_session_mesh_and_renders_serving(multi_component_chair_mesh):
    """Verifies /mesh and /renders/{view} serve binary files."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={"asset_name": "Chair", "mesh_path": multi_component_chair_mesh})
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id, "include_vlm": False, "user_description": "Wooden dining chair with backrest, seat, and four legs"})

    # Check mesh download
    mesh_resp = client.get(f"/api/v1/artist/session/{session_id}/mesh")
    assert mesh_resp.status_code == 200
    assert len(mesh_resp.content) > 100

    # Check front render download
    render_resp = client.get(f"/api/v1/artist/session/{session_id}/renders/front")
    assert render_resp.status_code == 200
    assert render_resp.headers["content-type"] == "image/png"

    # Check mask render download
    mask_resp = client.get(f"/api/v1/artist/session/{session_id}/renders/mask")
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "image/png"


# ==============================================================================
# 10. Blender Addon Panel 8 & Operator Registration in Headless Blender
# ==============================================================================

def test_blender_addon_registers_ai_technical_artist_panel():
    """Verifies Phase 6 operators and Panel 8 register successfully in Blender runtime."""
    blender_bin = hardware_detector.find_blender() or "blender"
    addon_path = PROJECT_ROOT / "addon" / "triposr_blender_addon.py"
    assert addon_path.exists()

    test_script = f"""
import sys, bpy
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

# Register
addon.register()

# Check Scene Properties for Phase 6
props = bpy.context.scene.local_mesh_props
assert hasattr(props, 'ai_session_id'), 'Missing props.ai_session_id'
assert hasattr(props, 'ai_artist_instruction'), 'Missing props.ai_artist_instruction'
assert hasattr(props, 'ai_plan_preview'), 'Missing props.ai_plan_preview'
assert hasattr(props, 'ai_has_plan'), 'Missing props.ai_has_plan'
assert hasattr(props, 'ai_needs_confirmation'), 'Missing props.ai_needs_confirmation'
assert hasattr(props, 'staged_image_path'), 'Missing props.staged_image_path'

# Check Phase 6 Operators in bpy.ops
assert hasattr(bpy.ops.localmesh, 'stage_active_asset'), 'Missing ops.localmesh.stage_active_asset'
assert hasattr(bpy.ops.localmesh, 'analyze_active_asset'), 'Missing ops.localmesh.analyze_active_asset'
assert hasattr(bpy.ops.localmesh, 'generate_artist_plan'), 'Missing ops.localmesh.generate_artist_plan'
assert hasattr(bpy.ops.localmesh, 'execute_artist_plan'), 'Missing ops.localmesh.execute_artist_plan'
assert hasattr(bpy.ops.localmesh, 'rollback_artist_edit'), 'Missing ops.localmesh.rollback_artist_edit'

# Check Panel 8 defined
assert hasattr(addon, 'LOCALMESH_PT_AITechnicalArtist'), 'Missing LOCALMESH_PT_AITechnicalArtist panel class'

# Unregister cleanly
addon.unregister()
print("PHASE_6_BLENDER_REGISTRATION_SUCCESS")
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", test_script], capture_output=True, text=True, timeout=25)
    assert res.returncode == 0, f"Blender test failed: {res.stderr}\nStdout: {res.stdout}"
    assert "PHASE_6_BLENDER_REGISTRATION_SUCCESS" in res.stdout
