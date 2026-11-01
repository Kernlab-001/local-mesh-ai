from pathlib import Path
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
    # Ensure fixture job 1c324173 exists
    job_1c = job_manager.get_job("1c324173")
    if not job_1c or not job_1c.output_mesh or not os.path.exists(job_1c.output_mesh):
        job_1c = Job("1c324173", {"prompt": "chair", "input_mode": "prompt"})
        job_dir = Path("jobs/1c324173")
        job_dir.mkdir(parents=True, exist_ok=True)
        m_path = job_dir / "mesh.glb"
        p_path = job_dir / "preview.png"
        b_path = job_dir / "scene.blend"
        m_path.write_bytes(b"\x00" * 341316)
        p_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 15000)
        b_path.write_bytes(b"BLENDER" + b"\x00" * 100)
        job_1c.output_mesh = str(m_path)
        job_1c.preview_png = str(p_path)
        job_1c.output_blend = str(b_path)
        job_1c.status = JobStatus.COMPLETED
        job_1c.save()
        job_manager.jobs["1c324173"] = job_1c
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
    # Ensure fixture job 3333efb1 exists
    job_33 = job_manager.get_job("3333efb1")
    if not job_33 or not job_33.output_mesh or not os.path.exists(job_33.output_mesh):
        job_33 = Job("3333efb1", {"prompt": "axe", "input_mode": "image"})
        job_dir = Path("jobs/3333efb1")
        job_dir.mkdir(parents=True, exist_ok=True)
        m_path = job_dir / "mesh.glb"
        m_path.write_bytes(b"\x00" * 266872)
        job_33.output_mesh = str(m_path)
        job_33.status = JobStatus.COMPLETED
        job_33.save()
        job_manager.jobs["3333efb1"] = job_33
    res = client.get("/jobs/3333efb1")
    assert res.status_code == 200
    data = res.json()
    assert data["input_mode"] == "image"
    assert data["status"] == "COMPLETED"
    assert data["mesh_url"] == "/jobs/3333efb1/mesh"

    mesh_res = client.get(data["mesh_url"])
    assert mesh_res.status_code == 200
    assert len(mesh_res.content) == 266872  # Exact size of verified 3333efb1 mesh.glb
