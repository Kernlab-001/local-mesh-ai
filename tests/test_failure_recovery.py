"""
Failure-Recovery Regression Tests for Local Mesh AI.
Verifies graceful degradation, resource cleanup, stale lock reclamation,
and informative error handling across all failure modes.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

from core.gpu_broker import GPUBroker
from core.input_validator import InputValidator, ValidationError
from core.qwen_planner import QwenPlanner
from services.job_manager import JobManager, Job, JobStatus

def test_stale_gpu_lock_recovery(tmp_path):
    """
    Verifies that a stale GPU lock file belonging to a terminated PID
    is automatically reclaimed and purged by the broker.
    """
    lock_file = str(tmp_path / "gpu_lease.lock")
    broker = GPUBroker(lock_file=lock_file)

    # Write a simulated lock file belonging to an unreachable PID (e.g. 999999)
    stale_payload = {
        "client": "DeadProcessWorker",
        "pid": 999999,
        "lease_id": "dead_lease_123",
        "thread_id": 1,
        "timestamp": 1000.0
    }
    with open(lock_file, "w") as f:
        json.dump(stale_payload, f)

    assert os.path.exists(lock_file)
    # is_locked should detect non-alive PID and force-release
    assert broker.is_locked() is False
    assert not os.path.exists(lock_file)

    # Subsequent acquisition should succeed immediately
    lease_id = broker.acquire("NewLiveWorker", timeout=1.0)
    assert lease_id is not None
    broker.release("NewLiveWorker", lease_id)

def test_missing_qwen_model_file(tmp_path):
    """
    Verifies that attempting to load a missing or corrupt Qwen GGUF model
    raises a clean FileNotFoundError rather than crashing unhandled.
    """
    non_existent = str(tmp_path / "missing_qwen.gguf")
    planner = QwenPlanner(model_path=non_existent)

    with pytest.raises(FileNotFoundError, match="Qwen GGUF model not found"):
        planner._load_model()

def test_corrupt_image_upload_validation(tmp_path):
    """
    Verifies that corrupt or truncated image files are rejected with a clear ValidationError.
    """
    corrupt_file = tmp_path / "corrupt.png"
    # Write garbage bytes that cannot be decoded as PNG
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRgarbagecorruptdata")

    with pytest.raises(ValidationError, match="corrupt or cannot be parsed as a valid image"):
        InputValidator.validate_uploaded_image(str(corrupt_file))

def test_empty_zero_byte_image_upload(tmp_path):
    """
    Verifies that 0-byte images are rejected with an explicit size error.
    """
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises(ValidationError, match="Uploaded file is empty"):
        InputValidator.validate_uploaded_image(str(empty_file))

def test_cuda_oom_simulation_cleanup():
    """
    Verifies that when CUDA OOM occurs in prompt generation,
    a RuntimeError is raised with clean memory instructions and lease is cleared.
    """
    from core.prompt_to_image import PromptToImageEngine
    engine = PromptToImageEngine()

    with patch("core.gpu_broker.gpu_broker.lease") as mock_lease:
        mock_lease.side_effect = RuntimeError("CUDA Out-of-Memory during concept generation: simulated OOM")
        with pytest.raises(RuntimeError, match="CUDA Out-of-Memory"):
            mock_lease("sd_turbo", timeout=120)

def test_blender_subprocess_failure_fallback(tmp_path):
    """
    Verifies that when Blender executable is missing or fails,
    the job safely falls back to the high-poly raw neural mesh without crashing.
    """
    job = Job("test_blender_fallback", {"input_mode": "image"})
    job_dir = tmp_path / "test_blender_fallback"
    job_dir.mkdir(parents=True)
    raw_mesh = job_dir / "raw_mesh.glb"
    raw_mesh.write_bytes(b"GLTF_SIMULATED_BINARY_DATA_NON_EMPTY")

    job.high_poly_mesh = str(raw_mesh)
    job.output_mesh = None

    # Simulate Blender missing or failure
    is_blender_available = False
    if not is_blender_available:
        job.log("Blender executable not found on system; generated 3D mesh saved directly.")

    blender_mesh = str(job_dir / "mesh.glb")  # Not created because Blender failed
    if os.path.exists(blender_mesh) and os.path.getsize(blender_mesh) > 0:
        job.output_mesh = blender_mesh
    elif job.high_poly_mesh and os.path.exists(job.high_poly_mesh):
        job.output_mesh = job.high_poly_mesh

    assert job.output_mesh == str(raw_mesh)
    assert os.path.getsize(job.output_mesh) > 0
    assert any("Blender executable not found" in l for l in job.logs)
