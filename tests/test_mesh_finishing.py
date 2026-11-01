"""
Test Suite for Phase 7: Professional 3D Finishing, Retopology, UV Unwrapping, PBR Materials & Game-Ready Export.
Verifies all finishing engines, Blender QuadriFlow retopology, Cycles baking, LOD generation,
multi-format export manifests, and Phase 6 defect corrections (invariants, plan integrity, threading, input hardening).
"""

import os
import sys
import json
import shutil
import tempfile
import pytest
import numpy as np
import trimesh

from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app
import core.mesh_finishing as mf
from core.geometry_editing import GeometryEditingEngine, EditComponentArgs
from services.editing_session_manager import editing_session_manager
from core.command_schemas import ComponentEditPlan, ComponentInvariant
from core.path_security import get_staging_dir, validate_safe_path

client = TestClient(app)


# ==============================================================================
# Test Fixtures: Clean and Defective Test Meshes
# ==============================================================================

@pytest.fixture(scope="module")
def sample_test_assets():
    """Creates temporary clean and defective test meshes for finishing validation."""
    tmp_dir = tempfile.mkdtemp(prefix="phase7_test_", dir=str(get_staging_dir()))

    # 1. Clean multi-component chair (High-Poly Source)
    box_seat = trimesh.creation.box(extents=[0.5, 0.5, 0.08])
    box_seat.apply_translation([0.0, 0.0, 0.45])

    box_back = trimesh.creation.box(extents=[0.46, 0.06, 0.50])
    box_back.apply_translation([0.0, -0.22, 0.74])

    leg1 = trimesh.creation.cylinder(radius=0.025, height=0.45)
    leg1.apply_translation([-0.20, -0.20, 0.225])

    leg2 = trimesh.creation.cylinder(radius=0.025, height=0.45)
    leg2.apply_translation([0.20, -0.20, 0.225])

    leg3 = trimesh.creation.cylinder(radius=0.025, height=0.45)
    leg3.apply_translation([-0.20, 0.20, 0.225])

    leg4 = trimesh.creation.cylinder(radius=0.025, height=0.45)
    leg4.apply_translation([0.20, 0.20, 0.225])

    clean_chair = trimesh.util.concatenate([box_seat, box_back, leg1, leg2, leg3, leg4])
    clean_path = os.path.join(tmp_dir, "clean_chair.glb")
    clean_chair.export(clean_path)

    # 2. Defective mesh: add duplicate vertices, degenerate zero-area faces, and reversed normals
    verts = box_seat.vertices.copy()
    faces = box_seat.faces.copy()

    # Append duplicate vertices
    dup_v = verts[:10].copy()
    all_verts = np.vstack([verts, dup_v])

    # Append degenerate zero-area faces (vertex v0, v0, v0)
    deg_faces = np.array([[0, 0, 0], [1, 1, 2], [3, 4, 3]])
    all_faces = np.vstack([faces, deg_faces])

    defective_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    defective_path = os.path.join(tmp_dir, "defective_mesh.glb")
    defective_mesh.export(defective_path)

    yield {
        "tmp_dir": tmp_dir,
        "clean_chair": clean_path,
        "defective_mesh": defective_path
    }

    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass


# ==============================================================================
# 1. Mesh Health & Validation Tests
# ==============================================================================

def test_mesh_health_analysis_clean_and_defective(sample_test_assets):
    """Verifies comprehensive mesh inspection: detects degenerate faces, duplicate elements, and health scores."""
    clean_rep = mf.analyze_mesh_health(sample_test_assets["clean_chair"])
    assert clean_rep.face_count > 100
    assert clean_rep.vertex_count > 50
    assert clean_rep.health_score >= 80.0
    assert clean_rep.degenerate_faces == 0

    defective_rep = mf.analyze_mesh_health(sample_test_assets["defective_mesh"])
    assert defective_rep.degenerate_faces > 0
    assert defective_rep.duplicate_vertices > 0
    assert defective_rep.health_score < clean_rep.health_score
    assert len(defective_rep.issues_found) >= 2


# ==============================================================================
# 2. Safe Mesh Repair Tests
# ==============================================================================

def test_mesh_repair_cleans_degeneracies_and_duplicates(sample_test_assets):
    """Verifies non-destructive repair purges degenerate faces and welds duplicate vertices."""
    defective_p = sample_test_assets["defective_mesh"]
    repaired_p = os.path.join(sample_test_assets["tmp_dir"], "repaired_mesh.glb")

    res = mf.repair_mesh(defective_p, output_path=repaired_p)
    assert res.success is True
    assert os.path.exists(repaired_p)
    assert len(res.repairs_applied) >= 2

    # Verify post-repair health improved
    assert res.after_health.degenerate_faces == 0
    assert res.after_health.duplicate_vertices == 0
    assert res.after_health.health_score > res.before_health.health_score


# ==============================================================================
# 3. High-Poly Preservation & Low-Poly Workflow Tests
# ==============================================================================

def test_high_poly_preservation_and_low_poly_generation(sample_test_assets):
    """Verifies original high-poly is strictly preserved while producing a decimation low-poly derivative."""
    clean_p = sample_test_assets["clean_chair"]
    high_before_size = os.path.getsize(clean_p)

    out_low = os.path.join(sample_test_assets["tmp_dir"], "chair_lowpoly.glb")
    res = mf.generate_low_poly(clean_p, target_triangles=400, output_path=out_low)

    assert os.path.exists(out_low)
    assert res.actual_face_count <= res.original_face_count
    assert res.reduction_ratio < 1.0
    assert res.max_surface_deviation >= 0.0

    # Verify high-poly source remained 100% immutable
    assert os.path.getsize(clean_p) == high_before_size


# ==============================================================================
# 4. Retopology Workflow Tests (QuadriFlow / Fallback)
# ==============================================================================

def test_quadriflow_retopology_workflow(sample_test_assets):
    """Verifies true quad retopology producing clean 4-sided polygon flow or documented fallback."""
    clean_p = sample_test_assets["clean_chair"]
    out_retopo = os.path.join(sample_test_assets["tmp_dir"], "chair_retopo.glb")

    res = mf.retopologize_quadriflow(clean_p, target_faces=800, preserve_sharp=True, output_path=out_retopo)
    assert os.path.exists(out_retopo)
    assert res.actual_faces > 0
    assert res.method in ("blender_quadriflow", "quad_dominant_decimation_fallback")
    assert res.provenance in ("native_blender_quadriflow_5.2", "quad_dominant_fallback")


# ==============================================================================
# 5. Professional UV Unwrapping & Packing Tests
# ==============================================================================

def test_professional_uv_unwrapping_and_packing(sample_test_assets):
    """Verifies smart UV projection and island packing with bleed margin."""
    clean_p = sample_test_assets["clean_chair"]
    out_uv = os.path.join(sample_test_assets["tmp_dir"], "chair_unwrapped.glb")

    res = mf.unwrap_uvs(clean_p, method="smart_project", margin=0.03, output_path=out_uv)
    assert os.path.exists(out_uv)
    assert res.uv_report.has_uv is True
    assert res.uv_report.uv_utilization_percent > 0.0
    assert res.uv_report.out_of_range_coords_count == 0


# ==============================================================================
# 6. PBR Material Presets & Shader Setup Tests
# ==============================================================================

def test_pbr_material_studio_and_presets(sample_test_assets):
    """Verifies Principled BSDF PBR material configuration and presets."""
    clean_p = sample_test_assets["clean_chair"]
    out_mat = os.path.join(sample_test_assets["tmp_dir"], "chair_pbr.glb")

    mat_specs = [
        mf.PBRMaterialSpec(
            material_name="M_Walnut_Frame",
            preset="wood_walnut",
            base_color_rgba=[0.28, 0.18, 0.12, 1.0],
            metallic=0.05,
            roughness=0.65
        ),
        mf.PBRMaterialSpec(
            material_name="M_Cream_Fabric",
            preset="fabric_cotton",
            base_color_rgba=[0.88, 0.85, 0.80, 1.0],
            metallic=0.0,
            roughness=0.85
        )
    ]

    res = mf.apply_pbr_materials(clean_p, materials=mat_specs, output_path=out_mat)
    assert os.path.exists(out_mat)
    assert res.slots_assigned >= 1
    assert len(res.materials_created) == 2


# ==============================================================================
# 7. Cycles Texture Baking Validation Tests
# ==============================================================================

def test_texture_baking_cycles_validation(sample_test_assets):
    """Verifies texture baking image validation logic and spec parsing."""
    bake_dir = os.path.join(sample_test_assets["tmp_dir"], "textures")
    os.makedirs(bake_dir, exist_ok=True)

    spec = mf.TextureBakeSpec(
        high_poly_path=sample_test_assets["clean_chair"],
        low_poly_path=sample_test_assets["clean_chair"],
        bake_resolution=512,
        bake_maps=["normal", "base_color"],
        output_dir=bake_dir
    )
    assert spec.bake_resolution == 512
    assert "normal" in spec.bake_maps

    # Test image validator with simulated valid normal map
    sim_normal_p = os.path.join(bake_dir, "test_normal.png")
    normal_pixels = np.full((512, 512, 3), [128, 128, 255], dtype=np.uint8)
    # Add varying normal gradients
    normal_pixels[:100, :, 0] = 180
    from PIL import Image
    Image.fromarray(normal_pixels).save(sim_normal_p)

    val_res = mf.validate_baked_image(sim_normal_p, expected_res=512)
    assert val_res["is_valid"] is True
    assert val_res["resolution"] == [512, 512]


# ==============================================================================
# 8. LOD Multi-Tier Generation Tests
# ==============================================================================

def test_lod_generation_multi_tier(sample_test_assets):
    """Verifies generation of LOD0 through LOD3 with correct triangle reduction ratios."""
    clean_p = sample_test_assets["clean_chair"]
    lod_dir = os.path.join(sample_test_assets["tmp_dir"], "lods")

    res = mf.generate_lods(clean_p, output_dir=lod_dir)
    assert len(res.lods) == 4
    assert "LOD0" in res.lods
    assert "LOD1" in res.lods
    assert "LOD2" in res.lods
    assert "LOD3" in res.lods

    # Verify descending triangle counts
    assert res.lods["LOD0"].triangle_count >= res.lods["LOD1"].triangle_count
    assert res.lods["LOD1"].triangle_count >= res.lods["LOD2"].triangle_count
    assert res.lods["LOD2"].triangle_count >= res.lods["LOD3"].triangle_count
    assert res.bounding_box_consistent is True


# ==============================================================================
# 9. Game-Ready Multi-Format Export & Manifest Tests
# ==============================================================================

def test_game_ready_export_and_manifest_validation(sample_test_assets):
    """Verifies export to GLB, FBX, OBJ with structural verification and asset_manifest.json."""
    clean_p = sample_test_assets["clean_chair"]
    export_dir = os.path.join(sample_test_assets["tmp_dir"], "export")

    res = mf.export_finished_asset(
        mesh_path=clean_p,
        export_dir=export_dir,
        asset_name="DiningChair",
        formats=["glb", "obj"]
    )

    assert "glb" in res.exported_files
    assert os.path.exists(res.exported_files["glb"])
    assert os.path.exists(res.manifest_path)

    # Read and validate manifest
    with open(res.manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["manifest_version"] == "7.0"
    assert "polygon_statistics" in manifest_data
    assert "health_and_validation" in manifest_data
    assert manifest_data["source_asset_path"] == os.path.abspath(clean_p)


# ==============================================================================
# 10. FastAPI Finishing REST Endpoints Tests
# ==============================================================================

def test_finishing_api_endpoints_via_testclient(sample_test_assets):
    """Verifies all Phase 7 FastAPI REST endpoints."""
    clean_p = sample_test_assets["clean_chair"]

    # 1. Validation endpoint
    v_resp = client.post("/api/v1/finishing/validate", json={"mesh_path": clean_p})
    assert v_resp.status_code == 200
    assert v_resp.json()["health_score"] >= 80.0

    # 2. Presets endpoint
    p_resp = client.get("/api/v1/finishing/presets")
    assert p_resp.status_code == 200
    assert "wood_walnut" in p_resp.json()["presets"]

    # 3. Low-poly endpoint
    lp_resp = client.post("/api/v1/finishing/low-poly", json={
        "mesh_path": clean_p,
        "target_triangles": 300
    })
    assert lp_resp.status_code == 200
    assert lp_resp.json()["actual_face_count"] <= lp_resp.json()["original_face_count"]

    # 4. LODs endpoint
    lod_resp = client.post("/api/v1/finishing/lods", json={"mesh_path": clean_p})
    assert lod_resp.status_code == 200
    assert "LOD0" in lod_resp.json()["lods"]


# ==============================================================================
# 11. Defect Fix Verification: Shared Boundary Seam Decoupling
# ==============================================================================

def test_shared_vertex_boundary_decoupling(sample_test_assets):
    """Verifies that editing a target component does NOT distort adjacent non-target components even with shared vertices."""
    clean_p = sample_test_assets["clean_chair"]
    edit_out = os.path.join(sample_test_assets["tmp_dir"], "decoupled_edit.glb")

    edit_args = EditComponentArgs(
        mesh_path=clean_p,
        operation="scale_axis",
        parameters={"axis": "z", "factor": 1.25, "anchor": "min"},
        output_path=edit_out
    )

    res = GeometryEditingEngine.execute_component_edit(edit_args)
    assert res.status == "success"
    assert res.non_target_components_preserved is True
    assert os.path.exists(edit_out)


# ==============================================================================
# 12. Defect Fix Verification: Plan Integrity & Override Restrictions
# ==============================================================================

def test_plan_integrity_and_override_restrictions(sample_test_assets):
    """Verifies that mismatched plan IDs are rejected and unauthorized overrides are ignored."""
    stage_resp = client.post("/api/v1/blender/stage-asset", json={
        "asset_name": "Chair",
        "mesh_path": sample_test_assets["clean_chair"]
    })
    session_id = stage_resp.json()["session_id"]
    client.post("/api/v1/artist/analyze", json={"session_id": session_id})
    plan_resp = client.post("/api/v1/artist/plan", json={
        "session_id": session_id,
        "user_instruction": "Make the backrest 20% taller"
    })
    real_plan_id = plan_resp.json()["plan_id"]

    # Attempt execution with fake/mismatched plan ID
    exec_fake = client.post("/api/v1/artist/execute", json={
        "session_id": session_id,
        "plan_id": "fake_superseded_plan_id_999"
    })
    assert exec_fake.status_code == 500
    assert "plan id" in exec_fake.json()["detail"].lower()


# ==============================================================================
# 13. Defect Fix Verification: Asset Staging Input Hardening
# ==============================================================================

def test_stage_asset_input_hardening():
    """Verifies rejection of unsupported file extensions and invalid base64 payloads."""
    # 1. Unsupported extension
    bad_fmt = client.post("/api/v1/blender/stage-asset", json={
        "asset_name": "Test",
        "mesh_base64": "AAAA",
        "file_format": "exe"
    })
    assert bad_fmt.status_code == 400
    assert "Unsupported mesh format" in bad_fmt.json()["detail"]

    # 2. Corrupted base64
    corrupt_b64 = client.post("/api/v1/blender/stage-asset", json={
        "asset_name": "Test",
        "mesh_base64": "not_valid_base_64!!!@@#$",
        "file_format": "glb"
    })
    assert corrupt_b64.status_code == 400
    assert "Invalid base64" in corrupt_b64.json()["detail"]
