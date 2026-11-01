"""
Automated Test Suite for Phase 4: Unified AI 3D Asset Generation Pipeline.
Verifies route discovery, hardware profiles, request validation, lifecycle state
transitions, cancellation, honest error gating, asset cataloging, and API contracts.
"""

import os
import sys
import time
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.generation_orchestrator import (
    generation_orchestrator,
    GenerationModality,
    HardwareProfile,
    PipelineStage,
    RouteUnavailableError,
    HARDWARE_PROFILES
)
from core.model_paths import model_path_resolver, ModelStatus
from services.job_manager import job_manager, Job, JobStatus
from api.app import app

client = TestClient(app)


# ==============================================================================
# GROUP A: Route Capability & Hardware Profile Verification
# ==============================================================================

def test_route_discovery_truthfulness():
    """Verifies that route discovery truthfully reports readiness based on installed model weights."""
    routes = generation_orchestrator.discover_routes()
    assert len(routes) >= 2, "Expected at least Image-to-3D and Text-to-3D routes."

    route_map = {r["route_id"]: r for r in routes}
    assert "image_to_3d_triposr" in route_map
    assert "text_to_3d_hybrid" in route_map

    # Image-to-3D: TripoSR is physically installed on host
    img_route = route_map["image_to_3d_triposr"]
    assert img_route["modality"] == "image-to-3d"
    assert img_route["ready"] is True
    assert img_route["status"] == "READY"
    assert "triposr" in img_route["required_models"]

    # Text-to-3D: SD-Turbo weights are MISSING on host
    txt_route = route_map["text_to_3d_hybrid"]
    assert txt_route["modality"] == "text-to-3d"
    assert txt_route["ready"] is False
    assert txt_route["status"] == "NEEDS_DOWNLOAD"
    assert "sd_turbo" in txt_route["required_models"]
    assert "MISSING" in txt_route["notes"] or "not found" in txt_route["notes"]


def test_hardware_profiles_definitions_and_adaptation():
    """Verifies hardware profiles definitions, tuning parameters, and memory constraints."""
    profiles = generation_orchestrator.get_hardware_profiles()
    assert "low_memory" in profiles
    assert "balanced" in profiles
    assert "quality" in profiles

    low = profiles["low_memory"]
    assert low["mc_resolution"] == 128
    assert low["chunk_size"] == 4096
    assert low["min_vram_gb"] <= 3.5

    bal = profiles["balanced"]
    assert bal["mc_resolution"] == 256
    assert bal["chunk_size"] == 8192
    assert bal["min_vram_gb"] <= 5.0

    qual = profiles["quality"]
    assert qual["mc_resolution"] == 384
    assert qual["chunk_size"] == 4096
    assert qual["min_vram_gb"] >= 5.0


# ==============================================================================
# GROUP B: Request Validation & Error Handling
# ==============================================================================

def test_validate_request_text_to_3d():
    """Verifies input validation for text prompts."""
    # 1. Valid prompt
    valid, err, params = generation_orchestrator.validate_request(
        modality="text-to-3d",
        params={"prompt": "A medieval carved wooden throne with red velvet cushion"}
    )
    assert valid is True
    assert err is None
    assert "wooden throne" in params["prompt"]
    assert params["hardware_profile"] == "balanced"

    # 2. Empty prompt rejected
    valid_empty, err_empty, _ = generation_orchestrator.validate_request(
        modality="text-to-3d",
        params={"prompt": "   "}
    )
    assert valid_empty is False
    assert "Prompt is required" in err_empty

    # 3. Invalid modality rejected
    valid_mod, err_mod, _ = generation_orchestrator.validate_request(
        modality="audio-to-3d",
        params={"prompt": "test"}
    )
    assert valid_mod is False
    assert "Unsupported modality" in err_mod


def test_validate_request_image_to_3d(tmp_path):
    """Verifies input validation for reference images."""
    # 1. Non-existent image rejected
    valid_missing, err_missing, _ = generation_orchestrator.validate_request(
        modality="image-to-3d",
        params={"image_path": str(tmp_path / "non_existent.png")}
    )
    assert valid_missing is False
    assert "does not exist" in err_missing

    # 2. Empty 0-byte image rejected
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    valid_empty, err_empty, _ = generation_orchestrator.validate_request(
        modality="image-to-3d",
        params={"image_path": str(empty_file)}
    )
    assert valid_empty is False
    assert "empty" in err_empty.lower()

    # 3. Valid image accepted and metadata extracted
    valid_img_path = tmp_path / "sample_chair.png"
    img = Image.new("RGBA", (128, 128), color=(100, 150, 200, 255))
    img.save(str(valid_img_path))

    valid_ok, err_ok, params_ok = generation_orchestrator.validate_request(
        modality="image-to-3d",
        params={"image_path": str(valid_img_path)}
    )
    assert valid_ok is True
    assert err_ok is None
    assert params_ok["image_metadata"]["width"] == 128
    assert params_ok["image_metadata"]["height"] == 128


def test_text_to_3d_route_rejection_without_sdturbo():
    """Verifies that requesting Text-to-3D when SD-Turbo is uninstalled raises RouteUnavailableError with clear guidance."""
    with pytest.raises(RouteUnavailableError) as exc_info:
        generation_orchestrator.resolve_route(
            modality="text-to-3d",
            params={"prompt": "An elven silver dagger with sapphire hilt"}
        )
    err_msg = str(exc_info.value)
    assert "SD-Turbo" in err_msg
    assert "MISSING" in err_msg
    assert "Image-to-3D mode" in err_msg


# ==============================================================================
# GROUP C: Lifecycle State Machine & Cancellation
# ==============================================================================

def test_job_status_enum_phase4_compliance():
    """Verifies that JobStatus contains all Phase 4 lifecycle states."""
    expected_states = [
        "QUEUED",
        "VALIDATING",
        "WAITING_FOR_GPU",
        "LOADING_MODEL",
        "GENERATING_CONCEPT",
        "RUNNING_RECONSTRUCTION",
        "PROCESSING_MESH",
        "SAVING_ASSET",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    ]
    for s in expected_states:
        assert hasattr(JobStatus, s), f"JobStatus missing state '{s}'"
        assert getattr(JobStatus, s) == s


def test_job_cancellation_semantics(tmp_path):
    """Verifies clean job cancellation and transition to CANCELLED state."""
    job = Job("test_cancel_flow", {"input_mode": "image", "image_path": "dummy.png"})
    assert job.status == JobStatus.QUEUED

    # User cancels job
    job.cancel()
    assert job.status == JobStatus.CANCELLED
    assert job._cancel_requested is True
    assert "Cancelled" in job.stage


# ==============================================================================
# GROUP D: Asset Cataloging and Retrieval
# ==============================================================================

def test_asset_registration_and_listing(tmp_path):
    """Verifies asset registration in the central index and metadata queryability."""
    mock_manifest = {
        "job_id": "test_asset_001",
        "created_at": "2026-11-01 12:00:00",
        "modality": "image-to-3d",
        "route": "image_to_3d_triposr",
        "hardware_profile": "balanced",
        "execution_provenance": "triposr_neural",
        "timings": {"reconstruction_seconds": 12.5, "total_pipeline_seconds": 18.2},
        "assets": {
            "primary_mesh_glb": str(tmp_path / "mock_mesh.glb")
        }
    }
    # Create non-empty mock mesh file
    (tmp_path / "mock_mesh.glb").write_bytes(b"glTF_mock_content_123456")

    generation_orchestrator._register_asset(mock_manifest)

    # Query asset list
    assets = generation_orchestrator.list_assets()
    matching = [a for a in assets if a.get("job_id") == "test_asset_001"]
    assert len(matching) == 1
    rec = matching[0]
    assert rec["asset_id"] == "asset_test_asset_001"
    assert rec["execution_provenance"] == "triposr_neural"
    assert rec["hardware_profile"] == "balanced"
    assert rec["file_size_bytes"] > 0

    # Query single asset
    single = generation_orchestrator.get_asset("test_asset_001")
    assert single is not None
    assert single["job_id"] == "test_asset_001"


# ==============================================================================
# GROUP E: API Endpoint Contract Verification
# ==============================================================================

def test_api_v1_routes_endpoint():
    """Verifies GET /api/v1/routes returns supported routes and hardware profiles."""
    res = client.get("/api/v1/routes")
    assert res.status_code == 200
    data = res.json()
    assert "routes" in data
    assert "hardware_profiles" in data
    assert any(r["route_id"] == "image_to_3d_triposr" for r in data["routes"])
    assert "balanced" in data["hardware_profiles"]


def test_api_v1_capabilities_endpoint():
    """Verifies GET /api/v1/capabilities returns live model inventory and hardware telemetry."""
    res = client.get("/api/v1/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "routes" in data
    assert "hardware" in data
    assert "triposr" in data["models"]


def test_api_v1_generate_text_rejection_when_weights_missing():
    """Verifies POST /api/v1/generate/text returns HTTP 400 with actionable error when SD-Turbo is missing."""
    res = client.post("/api/v1/generate/text", json={"prompt": "A golden goblet with sapphire gems"})
    assert res.status_code == 400
    data = res.json()
    assert "SD-Turbo" in data["detail"]
    assert "MISSING" in data["detail"]


def test_api_v1_generate_image_validation_rejection():
    """Verifies POST /api/v1/generate/image rejects invalid or non-existent image paths."""
    res = client.post("/api/v1/generate/image", json={"image_path": "non_existent_image_123.png"})
    assert res.status_code == 400
    data = res.json()
    assert "does not exist" in data["detail"].lower()


def test_api_v1_assets_endpoint():
    """Verifies GET /api/v1/assets returns registered assets list."""
    res = client.get("/api/v1/assets")
    assert res.status_code == 200
    data = res.json()
    assert "assets" in data
    assert "total" in data
    assert isinstance(data["assets"], list)
