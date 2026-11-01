# -*- coding: utf-8 -*-
"""
Phase 5 Comprehensive Verification Suite: Blender Studio Integration & Interactive 3D Workspace.
Tests:
1. Blender addon structural validity and registration in Blender 5.2 / 4.2 LTS headless runtime.
2. Blender addon 7 core panels and operator availability.
3. Image staging endpoint (/api/v1/stage-image) with format validation and dimension checks.
4. Two-way scene synchronization endpoints (/api/v1/blender/sync-scene & /api/v1/blender/scene-status).
5. Generation orchestrator Shap-E direct generation route dispatch.
6. Hardware profile live telemetry including available VRAM and memory pressure metrics.
"""

import os
import sys
import io
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.generation_orchestrator import generation_orchestrator
from core.hardware_detector import hardware_detector
from api.app import app

client = TestClient(app)


# ==============================================================================
# 1. Blender Runtime Addon Registration & Structural Inspection
# ==============================================================================

def test_blender_addon_registers_in_headless_blender():
    """Verifies triposr_blender_addon registers and exposes all properties in real Blender."""
    blender_bin = hardware_detector.find_blender() or "blender"
    addon_path = PROJECT_ROOT / "addon" / "triposr_blender_addon.py"
    assert addon_path.exists(), f"Addon file missing at {addon_path}"

    test_script = f"""
import sys, bpy
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

# Verify registration
addon.register()

# Check Scene properties
assert hasattr(bpy.context.scene, 'local_mesh_props'), 'Missing scene.local_mesh_props'
assert hasattr(bpy.context.scene, 'triposr_props'), 'Missing scene.triposr_props (backward compat)'

props = bpy.context.scene.local_mesh_props
assert hasattr(props, 'server_url'), 'Missing server_url property'
assert hasattr(props, 'input_mode'), 'Missing input_mode property'
assert hasattr(props, 'generation_route'), 'Missing generation_route property'
assert hasattr(props, 'hardware_profile'), 'Missing hardware_profile property'
assert hasattr(props, 'target_height'), 'Missing target_height property'
assert hasattr(props, 'active_job_id'), 'Missing active_job_id property'
assert hasattr(props, 'sync_status'), 'Missing sync_status property'

# Check essential operators are registered in bpy.ops
assert hasattr(bpy.ops.localmesh, 'generate_asset'), 'Missing ops.localmesh.generate_asset'
assert hasattr(bpy.ops.localmesh, 'sync_scene'), 'Missing ops.localmesh.sync_scene'
assert hasattr(bpy.ops.localmesh, 'origin_to_bottom'), 'Missing ops.localmesh.origin_to_bottom'
assert hasattr(bpy.ops.triposr, 'generate_asset'), 'Missing ops.triposr.generate_asset (backward compat)'
assert hasattr(bpy.ops.triposr, 'switch_viewport'), 'Missing ops.triposr.switch_viewport (backward compat)'

# Verify unregistration
addon.unregister()
assert not hasattr(bpy.context.scene, 'local_mesh_props'), 'Unregister failed to clean up local_mesh_props'

print("BLENDER_ADDON_REGISTRATION_SUCCESS")
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", test_script], capture_output=True, text=True, timeout=25)
    assert res.returncode == 0, f"Blender headless execution failed: {res.stderr}\nOutput: {res.stdout}"
    assert "BLENDER_ADDON_REGISTRATION_SUCCESS" in res.stdout


def test_blender_addon_contains_all_seven_panels():
    """Inspects that all 7 required studio panels are defined with category LOCAL MESH AI."""
    addon_code = (PROJECT_ROOT / "addon" / "triposr_blender_addon.py").read_text(encoding="utf-8")
    
    expected_panels = [
        "LOCALMESH_PT_Connection",
        "LOCALMESH_PT_Generate",
        "LOCALMESH_PT_JobMonitor",
        "LOCALMESH_PT_Assets",
        "LOCALMESH_PT_SceneInfo",
        "LOCALMESH_PT_SceneSync",
        "LOCALMESH_PT_MeshControls"
    ]
    for p in expected_panels:
        assert f"class {p}" in addon_code, f"Panel {p} not authored in addon"
        assert f"{p}," in addon_code or f"{p}\n" in addon_code, f"Panel {p} missing from addon registration tuple"

    assert 'bl_category = "LOCAL MESH AI"' in addon_code, "Addon sidebar tab must be 'LOCAL MESH AI'"


# ==============================================================================
# 2. Image Staging Endpoint (/api/v1/stage-image)
# ==============================================================================

def test_api_stage_image_valid_png():
    """Tests /api/v1/stage-image accepts valid reference image and returns metadata."""
    img = Image.new("RGBA", (128, 64), color=(200, 50, 50, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/v1/stage-image",
        files={"file": ("test_ref.png", buf, "image/png")}
    )
    assert res.status_code == 200, f"Expected 200, got: {res.text}"
    data = res.json()
    assert data["filename"].endswith("test_ref.png")
    assert data["format"] == "PNG"
    assert data["width"] == 128
    assert data["height"] == 64
    assert data["file_size_bytes"] > 0
    assert os.path.exists(data["staged_path"])


def test_api_stage_image_invalid_file_rejected():
    """Tests /api/v1/stage-image rejects corrupt or non-image files."""
    corrupt_buf = io.BytesIO(b"This is completely invalid binary garbage not an image")
    res = client.post(
        "/api/v1/stage-image",
        files={"file": ("corrupt.png", corrupt_buf, "image/png")}
    )
    assert res.status_code == 400
    assert "Corrupt or invalid image file" in res.json()["detail"]


# ==============================================================================
# 3. Two-Way Scene Synchronization Endpoints
# ==============================================================================

def test_api_blender_scene_synchronization_lifecycle():
    """Verifies scene sync pushes object graph and scene-status reflects state."""
    sync_payload = {
        "scene_name": "StudioScene_01",
        "objects": [
            {
                "name": "Hero_Asset",
                "type": "MESH",
                "location": [0.0, 0.0, 0.5],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "dimensions": [1.2, 0.8, 1.0],
                "poly_count": 4500,
                "material_names": ["M_Wood", "M_Leather"]
            },
            {
                "name": "Floor_Plane",
                "type": "MESH",
                "location": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [5.0, 5.0, 1.0],
                "dimensions": [10.0, 10.0, 0.0],
                "poly_count": 2,
                "material_names": ["M_Concrete"]
            }
        ],
        "units": "METRIC"
    }

    # 1. Push Scene Sync
    res = client.post("/api/v1/blender/sync-scene", json=sync_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "synchronized"
    assert data["scene_name"] == "StudioScene_01"
    assert data["synced_objects"] == 2
    assert data["scene_version"] >= 1
    assert "last_sync" in data or "timestamp" in data

    # 2. Query Scene Status
    status_res = client.get("/api/v1/blender/scene-status")
    assert status_res.status_code == 200
    s_data = status_res.json()
    assert s_data["scene_name"] == "StudioScene_01"
    assert s_data["object_count"] == 2
    assert len(s_data["objects"]) == 2
    assert s_data["objects"][0]["name"] == "Hero_Asset"
    assert s_data["objects"][0]["poly_count"] == 4500


# ==============================================================================
# 4. Generation Orchestrator Direct Shap-E Route Dispatch
# ==============================================================================

def test_orchestrator_shape_direct_route_dispatch(tmp_path):
    """Verifies that route_id == text_to_3d_direct dispatches directly to Shap-E engine."""
    mock_mesh_glb = tmp_path / "shape_direct_output.glb"
    mock_mesh_glb.write_bytes(b"glTF_shap_e_direct_content")

    mock_asset = MagicMock()
    mock_asset.mesh_path = str(mock_mesh_glb)
    mock_asset.generation_time_seconds = 2.5

    mock_shape = MagicMock()
    mock_shape.get_status.return_value.value = "ready"
    mock_shape.generate.return_value = mock_asset
    generation_orchestrator._cached_shape_status = "ready" 

    import uuid
    from services.job_manager import Job
    job_id = f"test_shape_{uuid.uuid4().hex[:8]}"
    req = {
        "prompt": "futuristic mechanical gear",
        "strategy": "direct_text_to_3d",
        "route_override": "text_to_3d_direct",
        "target_height": 0.5,
        "hardware_profile": "balanced"
    }
    job = Job(job_id=job_id, params=req)

    from core.model_paths import ModelStatus
    with patch("core.engines.registry.engine_registry.get_engine", return_value=mock_shape), \
         patch("core.model_paths.model_path_resolver.get_status", return_value=ModelStatus.RUNTIME_READY), \
         patch("core.gpu_broker.gpu_broker.acquire", return_value="test_lease"), \
         patch("core.gpu_broker.gpu_broker.release"):

        res = generation_orchestrator.execute_pipeline(job)

        assert job.status == "COMPLETED"
        assert res["route"] == "text_to_3d_direct"
        assert res["execution_provenance"] == "shape_implicit_direct"
        assert res["assets"]["primary_mesh_glb"].endswith("mesh.glb")
        mock_shape.generate.assert_called_once()


# ==============================================================================
# 5. Hardware Profile Telemetry & Memory Pressure Detection
# ==============================================================================

def test_hardware_profiles_telemetry_and_pressure_fields():
    """Verifies get_hardware_profiles() includes live telemetry and memory pressure metrics."""
    profiles = generation_orchestrator.get_hardware_profiles()

    assert "low_memory" in profiles
    assert "balanced" in profiles
    assert "quality" in profiles

    telemetry = profiles.get("_telemetry")
    assert telemetry is not None, "Hardware profiles must include '_telemetry' metadata"
    assert "device_name" in telemetry
    assert "available_vram_gb" in telemetry
    assert "memory_pressure" in telemetry
    assert isinstance(telemetry["memory_pressure"], bool)
    assert telemetry["recommended_profile"] in ("low_memory", "balanced", "quality")
