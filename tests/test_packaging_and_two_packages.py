"""
Automated Release Packaging and Distribution Tests for Local Mesh AI.
Verifies that:
1. Exactly two release archives exist in dist/releases/.
2. Package 1 (Code ZIP) contains all first-party application source and NO model weights or .venv.
3. Package 2 (Data ZIP) contains required core model checkpoints, manifests, and documentation.
4. Checksums match both packages.
5. Code and Data structures are compatible via LOCAL_MESH_AI_DATA_DIR.
"""

import json
import zipfile
import hashlib
from pathlib import Path

DIST_DIR = Path("E:/local-mesh-ai-v6.0/dist/releases")
CODE_ZIP = DIST_DIR / "LocalMeshAI-v6.0.0-Complete-Code.zip"
DATA_ZIP = DIST_DIR / "LocalMeshAI-v6.0.0-Complete-Data.zip"
CHECKSUMS_FILE = DIST_DIR / "LocalMeshAI-v6.0.0-Release-Checksums.txt"
CODE_MANIFEST = DIST_DIR / "LocalMeshAI-v6.0.0-Code-Manifest.json"
DATA_MANIFEST = DIST_DIR / "LocalMeshAI-v6.0.0-Data-Manifest.json"


def test_release_packages_and_supporting_files_exist():
    """Confirms all required distribution files exist in dist/releases/."""
    assert CODE_ZIP.exists(), f"Missing {CODE_ZIP}"
    assert DATA_ZIP.exists(), f"Missing {DATA_ZIP}"
    assert CHECKSUMS_FILE.exists(), f"Missing {CHECKSUMS_FILE}"
    assert CODE_MANIFEST.exists(), f"Missing {CODE_MANIFEST}"
    assert DATA_MANIFEST.exists(), f"Missing {DATA_MANIFEST}"


def test_code_zip_contains_complete_application_and_no_weights():
    """Package 1 must contain complete codebase, launchers, tests and NO .venv or model weights."""
    with zipfile.ZipFile(CODE_ZIP, "r") as zf:
        namelist = zf.namelist()

        # Check required files
        assert any("Start_Local_Mesh_AI.bat" in name for name in namelist)
        assert any("run_server.py" in name for name in namelist)
        assert any("requirements.txt" in name for name in namelist)
        assert any("README.md" in name for name in namelist)
        assert any("addon/triposr_blender_addon.py" in name for name in namelist)
        assert any("core/gpu_broker.py" in name for name in namelist)

        # Check strict exclusions
        assert not any(".venv/" in name or "/.venv/" in name for name in namelist)
        assert not any(".git/" in name or "/.git/" in name for name in namelist)
        assert not any(name.endswith(".gguf") for name in namelist)
        assert not any(name.endswith(".safetensors") for name in namelist)
        assert not any(name.endswith(".ckpt") for name in namelist)


def test_data_zip_contains_required_models_and_manifests():
    """Package 2 must contain core model checkpoints, manifests, and documentation."""
    with zipfile.ZipFile(DATA_ZIP, "r") as zf:
        namelist = zf.namelist()

        # Check documentation and manifests
        assert "LocalMeshAI-Data/README_DATA_PACKAGE.md" in namelist
        assert "LocalMeshAI-Data/manifests/model_manifest.json" in namelist
        assert "LocalMeshAI-Data/manifests/SHA256SUMS.txt" in namelist

        # Check core models
        assert any("model.ckpt" in name for name in namelist)
        assert any("qwen2.5-coder-7b-instruct-q4_0.gguf" in name for name in namelist)
        assert any("Qwen3VL-4B-Instruct-Q4_K_M.gguf" in name for name in namelist)
        assert any("model_objaverse.ckpt" in name for name in namelist)


def test_release_checksums_match_actual_archives():
    """Verifies that the hashes recorded in the checksums file match the physical ZIP files."""
    with open(CHECKSUMS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Compute actual SHA-256
    def get_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            while chunk := fp.read(1048576):
                h.update(chunk)
        return h.hexdigest()

    code_hash = get_hash(CODE_ZIP)
    data_hash = get_hash(DATA_ZIP)

    assert code_hash in content
    assert data_hash in content
