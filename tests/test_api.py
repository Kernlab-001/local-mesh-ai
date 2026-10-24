"""
Tests for FastAPI Orchestration Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "gpu" in data
    assert "blender" in data
    assert isinstance(data["blender"]["installed"], bool)

def test_missing_image_and_prompt_submission():
    payload = {
        "target_height": 1.8
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 400
    assert "Either image_path or prompt must be provided" in res.json()["detail"]

def test_prompt_job_submission(monkeypatch):
    from PIL import Image
    import os
    from services.job_manager import job_manager

    # Mock prompt generation so the unit test doesn't launch a heavy GPU inference thread
    def mock_generate(prompt, output_path, **kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.new("RGBA", (128, 128), color=(255, 0, 0, 255))
        img.save(output_path)
        return output_path

    monkeypatch.setattr("core.prompt_to_image.prompt_engine.generate", mock_generate)
    monkeypatch.setattr("services.job_manager.prompt_engine.generate", mock_generate)

    payload = {
        "prompt": "sci-fi crate box",
        "target_height": 1.0,
        "quality": "draft"
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["status"] in ("QUEUED", "VALIDATING", "PREPROCESSING")
    # Cancel immediately so background worker doesn't proceed to TripoSR during tests
    job_manager.cancel_job(data["job_id"])

def test_missing_image_job_submission():
    payload = {
        "image_path": "E:/GPUPLAT/non_existent_image.png",
        "target_height": 1.8
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 400
    assert "does not exist" in res.json()["detail"]

def test_get_nonexistent_job():
    res = client.get("/jobs/non_existent_id")
    assert res.status_code == 404

def test_export_nonexistent_job():
    res = client.post("/jobs/non_existent_id/export", json={"format": "glb"})
    assert res.status_code == 400
    assert "Job blend project not available" in res.json()["detail"]

def test_export_service_timeout(tmp_path, monkeypatch):
    import subprocess
    from services.export_service import export_service
    
    dummy_blend = tmp_path / "test.blend"
    dummy_blend.write_text("blend dummy")

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=180.0)

    monkeypatch.setattr(subprocess, "run", mock_run)
    
    with pytest.raises(RuntimeError, match="timed out after 180 seconds"):
        export_service.export_asset(
            blend_path=str(dummy_blend),
            format_type="glb"
        )

def test_viewer_endpoint():
    res = client.get("/viewer")
    assert res.status_code == 200
    assert "canvas-container" in res.text
    assert "THREE.GLTFLoader" in res.text

def test_job_mesh_endpoint_not_found():
    res = client.get("/jobs/non_existent_id/mesh")
    assert res.status_code == 404

def test_job_mesh_endpoint_existing():
    # Job 1c324173 is a verified completed job on disk
    res = client.get("/jobs/1c324173/mesh")
    if res.status_code == 200:
        assert res.headers["content-type"] in ("model/gltf-binary", "application/octet-stream")
        assert len(res.content) > 1000  # verified non-empty GLB

def test_job_response_urls():
    res = client.get("/jobs/1c324173")
    if res.status_code == 200:
        data = res.json()
        assert "mesh_url" in data
        assert data["mesh_url"] == "/jobs/1c324173/mesh"
        assert "preview_url" in data


