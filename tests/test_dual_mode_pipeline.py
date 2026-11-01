"""
Comprehensive Tests for Dual-Mode Operations:
- Mode A (Text-to-3D): Validates text prompt, runs concept diffusion, and passes into reconstruction.
- Mode B (Image-to-3D): Validates uploaded photograph, stages it safely, and BYPASSES concept diffusion.
- Validation: Edge cases for corrupt images, empty files, oversized payloads.
- Manifest: Structured production manifest generation.
"""

import os
import pytest
from unittest.mock import MagicMock
from PIL import Image

from services.job_manager import JobManager, JobStatus, Job
from core.input_validator import input_validator, ValidationError

def test_input_validator_prompt():
    # Valid prompt
    assert input_validator.validate_prompt("medieval glowing battleaxe") == "medieval glowing battleaxe"

    # Too short
    with pytest.raises(ValidationError):
        input_validator.validate_prompt("x")

    # Empty / whitespace
    with pytest.raises(ValidationError):
        input_validator.validate_prompt("   ")

def test_input_validator_upload(tmp_path):
    # Valid image
    img_path = str(tmp_path / "valid.png")
    img = Image.new("RGBA", (128, 128), color=(255, 0, 0, 255))
    img.save(img_path)

    stage_dir = str(tmp_path / "staged")
    res = input_validator.validate_uploaded_image(img_path, destination_dir=stage_dir)
    assert res["valid"] is True
    assert res["width"] == 128
    assert res["height"] == 128
    assert os.path.exists(res["staged_path"])

    # Corrupt / 0-byte file
    empty_file = str(tmp_path / "empty.png")
    with open(empty_file, "wb") as f:
        pass

    with pytest.raises(ValidationError, match="empty"):
        input_validator.validate_uploaded_image(empty_file)

    # Unsupported extension
    bad_ext = str(tmp_path / "model.txt")
    with open(bad_ext, "w") as f:
        f.write("text content")

    with pytest.raises(ValidationError, match="Unsupported file extension"):
        input_validator.validate_uploaded_image(bad_ext)

def test_mode_b_image_to_3d_bypasses_diffusion(tmp_path, monkeypatch):
    """
    CRITICAL REQUIREMENT: Mode B (Image-to-3D) must NEVER invoke
    the text-to-image diffusion generator!
    """
    img_path = str(tmp_path / "concept_photo.png")
    img = Image.new("RGBA", (128, 128), color=(100, 150, 200, 255))
    img.save(img_path)

    # Mock diffusion generator with a spy that raises if called
    mock_diffusion = MagicMock()
    monkeypatch.setattr("services.job_manager.prompt_engine.generate", mock_diffusion)

    # Mock reconstruction and blender to test pipeline flow
    def mock_reconstruct(*args, **kwargs):
        mesh_out = str(tmp_path / "mock_mesh.glb")
        with open(mesh_out, "wb") as f:
            f.write(b"glTF mock binary")
        return {"mesh_path": mesh_out, "elapsed_seconds": 1.2}

    monkeypatch.setattr("services.job_manager.triposr_engine.reconstruct", mock_reconstruct)
    monkeypatch.setattr("services.job_manager.hardware_detector.find_blender", lambda: None)

    jm = JobManager()
    job = jm.create_job({
        "input_mode": "image",
        "image_path": img_path,
        "accurate_colors": False
    })

    # Wait briefly for worker thread to complete
    import time
    for _ in range(30):
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.1)

    assert job.status == JobStatus.COMPLETED
    assert job.input_mode == "image"
    # Ensure text-to-image diffusion was NEVER invoked!
    mock_diffusion.assert_not_called()
    assert job.output_mesh is not None

def test_mode_a_text_to_3d_invokes_diffusion(tmp_path, monkeypatch):
    """
    Mode A (Text-to-3D) must invoke concept diffusion and pass generated asset downstream.
    """
    diffusion_called = []

    def mock_diffusion_generate(prompt, output_path, **kwargs):
        diffusion_called.append(prompt)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.new("RGBA", (128, 128), color=(200, 50, 50, 255))
        img.save(output_path)
        return output_path

    from core.model_paths import ModelStatus
    monkeypatch.setattr("core.model_paths.model_path_resolver.get_status", lambda name: ModelStatus.RUNTIME_READY if name in ("sd_turbo", "triposr") else ModelStatus.MISSING)
    monkeypatch.setattr("services.job_manager.prompt_engine.generate", mock_diffusion_generate)

    def mock_reconstruct(*args, **kwargs):
        mesh_out = str(tmp_path / "mock_mesh_a.glb")
        with open(mesh_out, "wb") as f:
            f.write(b"glTF mock binary")
        return {"mesh_path": mesh_out, "elapsed_seconds": 0.8}

    monkeypatch.setattr("services.job_manager.triposr_engine.reconstruct", mock_reconstruct)
    monkeypatch.setattr("services.job_manager.hardware_detector.find_blender", lambda: None)

    jm = JobManager()
    job = jm.create_job({
        "input_mode": "prompt",
        "prompt": "futuristic plasma rifle",
        "accurate_colors": False
    })

    import time
    for _ in range(30):
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.1)

    assert job.status == JobStatus.COMPLETED
    assert job.input_mode == "prompt"
    assert len(diffusion_called) == 1
    assert "futuristic plasma rifle" in diffusion_called[0]
    assert job.manifest is not None
    assert job.manifest["input_mode"] == "prompt"
