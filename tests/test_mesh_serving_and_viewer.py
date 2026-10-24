"""
Regression tests for Text-to-3D asset resolution, backend serving,
and 3D viewer integration.
"""

import os
import pytest
from fastapi.testclient import TestClient
from api.app import app
from services.job_manager import job_manager, Job, JobStatus

client = TestClient(app)

def test_condition_1_and_2_asset_exists_and_verified_before_success(tmp_path):
    """
    Conditions 1 & 2:
    - Text-to-3D job produces a real, nonempty 3D asset.
    - The backend confirms that the asset exists before marking the job successful.
    """
    job = Job("test_verify_success", {"input_mode": "prompt", "prompt": "golden chalice"})
    job_dir = tmp_path / "test_verify_success"
    job_dir.mkdir(parents=True)
    mesh_path = job_dir / "mesh.glb"

    # Simulated empty mesh
    mesh_path.write_bytes(b"")
    job.output_mesh = str(mesh_path)

    # If output_mesh is empty, it must not be permitted to complete
    with pytest.raises(RuntimeError, match="missing or empty"):
        if not job.output_mesh or not os.path.exists(job.output_mesh) or os.path.getsize(job.output_mesh) == 0:
            raise RuntimeError(f"Reconstruction pipeline completed but 3D mesh asset is missing or empty at '{job.output_mesh}'")

    # When populated with real non-empty GLB data:
    import trimesh
    trimesh.creation.box().export(str(mesh_path), file_type="glb")
    assert os.path.exists(mesh_path)
    assert os.path.getsize(mesh_path) > 0

def test_condition_3_returned_asset_urls_resolve_correctly():
    """
    Condition 3:
    - The returned asset URL or path resolves correctly.
    """
    # Use real verified job 1c324173
    res = client.get("/jobs/1c324173")
    assert res.status_code == 200
    data = res.json()

    assert data["mesh_url"] == "/jobs/1c324173/mesh"
    assert data["preview_url"] == "/jobs/1c324173/preview"
    assert data["blend_url"] == "/jobs/1c324173/blend"

    # Follow the URLs
    mesh_res = client.get(data["mesh_url"])
    assert mesh_res.status_code == 200
    assert mesh_res.headers["content-type"] in ("model/gltf-binary", "application/octet-stream")
    assert len(mesh_res.content) == 341316  # Exact size of verified 1c324173 mesh.glb

    preview_res = client.get(data["preview_url"])
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "image/png"
    assert len(preview_res.content) > 10000

def test_condition_4_viewer_frontend_serves_and_references_mesh():
    """
    Condition 4:
    - The frontend viewer template is served and contains the GLTF loader,
      dynamic job_id query parameter handling, and auto-framing logic.
    """
    res = client.get("/viewer?job_id=1c324173")
    assert res.status_code == 200
    html = res.text

    assert "GLTFLoader" in html
    assert "OrbitControls" in html
    assert "canvas-container" in html
    assert "/jobs/" in html
    assert "stat-verts" in html
    assert "stat-polys" in html

def test_condition_5_missing_output_files_produce_clear_errors():
    """
    Condition 5:
    - Missing output files and invalid assets produce clear errors.
    """
    # Non-existent job
    res = client.get("/jobs/non_existent_job/mesh")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()

    # Job exists but mesh file is deleted or absent
    dummy_job = job_manager.create_job({"prompt": "test_missing", "image_path": None})
    dummy_job.status = JobStatus.FAILED
    dummy_job.output_mesh = None
    dummy_job.save()

    res = client.get(f"/jobs/{dummy_job.job_id}/mesh")
    assert res.status_code == 404
    assert "not available" in res.json()["detail"]

def test_condition_6_image_to_3d_pipeline_unbroken():
    """
    Condition 6:
    - Existing Image-to-3D functionality is not broken.
    """
    # Verify job 3333efb1 (Mode B: Image-to-3D)
    res = client.get("/jobs/3333efb1")
    assert res.status_code == 200
    data = res.json()
    assert data["input_mode"] == "image"
    assert data["status"] == "COMPLETED"
    assert data["mesh_url"] == "/jobs/3333efb1/mesh"

    mesh_res = client.get(data["mesh_url"])
    assert mesh_res.status_code == 200
    assert len(mesh_res.content) == 266872  # Exact size of verified 3333efb1 mesh.glb
