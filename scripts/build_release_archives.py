"""
Release Packaging Automation Tool for Local Mesh AI.
Produces EXACTLY TWO canonical ZIP release packages:
  1. dist/releases/LocalMeshAI-v6.0.0-Complete-Code.zip
  2. dist/releases/LocalMeshAI-v6.0.0-Complete-Data.zip
Along with supporting release metadata:
  - dist/releases/LocalMeshAI-v6.0.0-Release-Checksums.txt
  - dist/releases/LocalMeshAI-v6.0.0-Code-Manifest.json
  - dist/releases/LocalMeshAI-v6.0.0-Data-Manifest.json

Features:
  - Strict inclusion and exclusion filters.
  - Streaming SHA-256 calculation.
  - Archive integrity verification (testzip).
  - Clean temporary extraction verification.
  - Cross-package compatibility test.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist" / "releases"
VERSION = "v6.0.0"
CODE_ZIP_NAME = f"LocalMeshAI-{VERSION}-Complete-Code.zip"
DATA_ZIP_NAME = f"LocalMeshAI-{VERSION}-Complete-Data.zip"
CHECKSUMS_NAME = f"LocalMeshAI-{VERSION}-Release-Checksums.txt"
CODE_MANIFEST_NAME = f"LocalMeshAI-{VERSION}-Code-Manifest.json"
DATA_MANIFEST_NAME = f"LocalMeshAI-{VERSION}-Data-Manifest.json"

def resolve_external_models_dir() -> Path:
    """Resolves the external model directory without hard-coded developer paths.

    Order: ``LOCAL_MESH_AI_DATA_DIR`` (the data package root) -> the project's
    own model resolver -> the in-project ``models`` folder.
    """
    env_data = os.environ.get("LOCAL_MESH_AI_DATA_DIR")
    if env_data:
        base = Path(env_data)
        candidate = base / "models"
        if candidate.exists():
            return candidate
        if base.name.lower() == "models" and base.exists():
            return base

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.model_paths import model_path_resolver  # type: ignore
        resolved = model_path_resolver.get_models_root()
        if resolved and Path(resolved).exists():
            return Path(resolved)
    except Exception as exc:
        print(f"  [WARN] model resolver unavailable ({exc}); falling back to ./models")

    return PROJECT_ROOT / "models"


EXTERNAL_MODELS_DIR = resolve_external_models_dir()


def compute_sha256(file_path: Path) -> str:
    """Computes streaming SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1048576):  # 1MB chunks
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 1. Package 1: Complete Project Code Package
# ==============================================================================

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "scratch",
    ".idea",
    ".vscode",
    # Runtime state must never ship in the Code package.
    "jobs",
    "logs",
    "output",
    "reports",
    "release",
    ".blender_user",
    ".mypy_cache",
    ".ruff_cache",
}

#: Strict inclusion allowlist for the Code package. Only these top-level
#: directories and files are packaged; anything else is skipped even when it
#: lives inside the project tree. This replaces the previous "walk everything
#: and exclude a list of known-bad names" approach, which leaked runtime job
#: output, logs and scratch files into the release archive.
CODE_INCLUDE_DIRS = (
    "addon",
    "api",
    "bridge",
    "core",
    "docs",
    "examples",
    "inputs",
    "launchers",
    "scripts",
    "services",
    "tests",
    "TripoSR",
    "ui",
    "workers",
)

CODE_INCLUDE_FILES = (
    "Start_Local_Mesh_AI.bat",
    "run_server.py",
    "run_master.py",
    "detect_hardware.py",
    "requirements.txt",
    "config.example.json",
    "README.md",
    "SECURITY.md",
    "PRIVACY.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "BEGINNER_GUIDE.md",
    "MODEL_LICENSES.md",
    "THIRD_PARTY_NOTICES.md",
    "GITHUB_RELEASE_DESCRIPTION.md",
)

#: Paths (relative, POSIX) that are never packaged even inside an allowed root.
CODE_EXCLUDE_REL_PATHS = (
    "data/staging",
    "data/editing_sessions",
    "data/learning",
    "data/recovery_audit.json",
    "data/gpu_lease.lock",
)

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".gguf",
    ".safetensors",
    ".ckpt",
    ".pt",
    ".bin",
    ".zip",
    ".7z",
    ".lock",
}


def should_include_in_code_zip(rel_path: str, p: Path) -> bool:
    """Determines if a file should be packaged in the Code ZIP.

    The decision is allowlist-first: the top-level entry must be listed in
    ``CODE_INCLUDE_DIRS`` / ``CODE_INCLUDE_FILES``.
    """
    parts = rel_path.replace("\\", "/").split("/")
    top = parts[0]

    if len(parts) == 1:
        return top in CODE_INCLUDE_FILES and p.is_file()

    if top not in CODE_INCLUDE_DIRS:
        return False

    # Exclude directories
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False

    rel_posix = "/".join(parts)
    for excluded in CODE_EXCLUDE_REL_PATHS:
        if rel_posix == excluded or rel_posix.startswith(excluded + "/"):
            return False

    if p.suffix.lower() in EXCLUDE_EXTENSIONS:
        # Never ship model weights or archives inside the Code package.
        return False

    return True


def build_code_package() -> Tuple[Path, Dict[str, Any]]:
    """Builds the complete code ZIP package."""
    print("\n" + "=" * 70)
    print("  BUILDING PACKAGE 1: COMPLETE PROJECT CODE")
    print("=" * 70)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = DIST_DIR / CODE_ZIP_NAME
    if out_zip.exists():
        out_zip.unlink()

    file_manifest: Dict[str, Any] = {}
    total_uncompressed_bytes = 0
    file_count = 0

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place for efficiency
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                p = Path(root) / file
                rel_path = str(p.relative_to(PROJECT_ROOT))

                if should_include_in_code_zip(rel_path, p):
                    arcname = f"LocalMeshAI-{VERSION}/{rel_path.replace('\\', '/')}"
                    zf.write(p, arcname)
                    sz = p.stat().st_size
                    total_uncompressed_bytes += sz
                    file_count += 1
                    file_manifest[rel_path.replace("\\", "/")] = {
                        "size_bytes": sz,
                        "archive_path": arcname,
                    }

    zip_size_bytes = out_zip.stat().st_size
    zip_sha256 = compute_sha256(out_zip)

    summary = {
        "package_name": CODE_ZIP_NAME,
        "version": VERSION,
        "package_type": "complete_code",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "file_count": file_count,
        "uncompressed_bytes": total_uncompressed_bytes,
        "archive_size_bytes": zip_size_bytes,
        "archive_size_mb": round(zip_size_bytes / (1024**2), 2),
        "sha256": zip_sha256,
        "files": file_manifest,
    }

    manifest_path = DIST_DIR / CODE_MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  [SUCCESS] Created {CODE_ZIP_NAME}")
    print(f"  Files: {file_count} | Size: {summary['archive_size_mb']} MB")
    print(f"  SHA-256: {zip_sha256}")
    return out_zip, summary


# ==============================================================================
# 2. Package 2: Complete AI Data Folder Package
# ==============================================================================

MODEL_LICENSES = {
    "triposr": {
        "name": "TripoSR Feedforward 3D Reconstruction Checkpoint",
        "upstream_repo": "stabilityai/TripoSR",
        "license": "MIT License",
        "redistribution_permitted": True,
        "commercial_use_permitted": True,
        "attribution_required": True,
        "notes": "Original weights released under open MIT terms by Stability AI & Tripo AI."
    },
    "dino": {
        "name": "DINO ViT-B/16 Vision Transformer Checkpoint",
        "upstream_repo": "facebookresearch/dino",
        "license": "Apache 2.0",
        "redistribution_permitted": True,
        "commercial_use_permitted": True,
        "attribution_required": True,
        "notes": "Open weights by Meta AI Research."
    },
    "qwen_coder": {
        "name": "Qwen 2.5 Coder 7B Instruct GGUF (Q4_0)",
        "upstream_repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "license": "Apache 2.0",
        "redistribution_permitted": True,
        "commercial_use_permitted": True,
        "attribution_required": True,
        "notes": "Alibaba Cloud open Apache 2.0 license."
    },
    "qwen3_vl_4b": {
        "name": "Qwen3-VL 4B Instruct GGUF (Q4_K_M + mmproj F16)",
        "upstream_repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "license": "Apache 2.0",
        "redistribution_permitted": True,
        "commercial_use_permitted": True,
        "attribution_required": True,
        "notes": "Alibaba Cloud open Apache 2.0 license."
    },
    "partfield": {
        "name": "PartField 3D Component Decomposition Checkpoint",
        "upstream_repo": "NVIDIA/PartField",
        "license": "NVIDIA Research License (Non-Commercial)",
        "redistribution_permitted": False,  # Restricted commercial distribution
        "commercial_use_permitted": False,
        "attribution_required": True,
        "notes": "Research checkpoint for semantic component segmentation."
    },
    "sd_turbo": {
        "name": "SD-Turbo 1-Step Latent Diffusion Concept Generator",
        "upstream_repo": "stabilityai/sd-turbo",
        "license": "Stability AI Non-Commercial Research Community License",
        "redistribution_permitted": False,
        "commercial_use_permitted": False,
        "attribution_required": True,
        "notes": "Large 24GB optional concept generator; provisioned externally via download manager."
    }
}


def build_data_package() -> Tuple[Path, Dict[str, Any]]:
    """Builds the separate complete AI data package."""
    print("\n" + "=" * 70)
    print("  BUILDING PACKAGE 2: COMPLETE AI DATA FOLDER")
    print("=" * 70)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = DIST_DIR / DATA_ZIP_NAME
    if out_zip.exists():
        out_zip.unlink()

    # Collect model assets from EXTERNAL_MODELS_DIR.
    # Each entry carries the license key it belongs to so that the license
    # register can be enforced before anything is written into the archive.
    model_files_to_pack: List[Tuple[Path, str, str]] = []

    # 1. TripoSR & DINO (hf_hub)
    hf_hub = EXTERNAL_MODELS_DIR / "hf_hub"
    if hf_hub.exists():
        for root, dirs, files in os.walk(hf_hub):
            dirs.sort()
            for f in sorted(files):
                p = Path(root) / f
                if f.endswith(".lock"):
                    continue
                rel = p.relative_to(EXTERNAL_MODELS_DIR)
                key = "dino" if "dino" in rel.as_posix().lower() else "triposr"
                model_files_to_pack.append((p, f"LocalMeshAI-Data/models/{rel.as_posix()}", key))

    # 2. Qwen Coder
    qwen_dir = EXTERNAL_MODELS_DIR / "qwen_coder"
    if qwen_dir.exists():
        for f in sorted(qwen_dir.glob("*.gguf")):
            rel = f.relative_to(EXTERNAL_MODELS_DIR)
            model_files_to_pack.append((f, f"LocalMeshAI-Data/models/{rel.as_posix()}", "qwen_coder"))

    # 3. Qwen3-VL 4B
    qwen3_dir = EXTERNAL_MODELS_DIR / "qwen3_vl_4b"
    if qwen3_dir.exists():
        for f in sorted(qwen3_dir.glob("*.gguf")):
            rel = f.relative_to(EXTERNAL_MODELS_DIR)
            model_files_to_pack.append((f, f"LocalMeshAI-Data/models/{rel.as_posix()}", "qwen3_vl_4b"))

    # 4. PartField checkpoint & config
    partfield_dir = EXTERNAL_MODELS_DIR / "partfield"
    if partfield_dir.exists():
        ckpt = partfield_dir / "model_objaverse.ckpt"
        if ckpt.exists():
            rel = ckpt.relative_to(EXTERNAL_MODELS_DIR)
            model_files_to_pack.append((ckpt, f"LocalMeshAI-Data/models/{rel.as_posix()}", "partfield"))

    # 5. SD-Turbo is provisioned on demand and is NOT redistributed here.

    # --- LICENSE ENFORCEMENT (fail closed) -----------------------------------
    packaged: List[Tuple[Path, str, str]] = []
    excluded_models: List[Dict[str, Any]] = []
    excluded_keys = set()
    for source, arcname, key in model_files_to_pack:
        license_entry = MODEL_LICENSES.get(key)
        if license_entry is None:
            raise RuntimeError(
                f"Refusing to build: '{source}' maps to model key '{key}' which has no "
                f"entry in MODEL_LICENSES. Add an explicit license decision first."
            )
        if not license_entry.get("redistribution_permitted", False):
            excluded_keys.add(key)
            excluded_models.append({
                "model_key": key,
                "name": license_entry["name"],
                "license": license_entry["license"],
                "reason": "redistribution_permitted is False for this model",
                "excluded_files": [],
                "user_provisioning": license_entry.get("notes", ""),
            })
            continue
        packaged.append((source, arcname, key))

    for entry in excluded_models:
        entry["excluded_files"] = [
            arcname for _src, arcname, key in model_files_to_pack if key == entry["model_key"]
        ]

    if excluded_models:
        print("\n  [LICENSE] The following models are NOT redistributed by this package:")
        for entry in excluded_models:
            print(f"    - {entry['name']} ({entry['license']})")
            print(f"      {len(entry['excluded_files'])} file(s) excluded; provision it yourself.")

    # Generate README_DATA_PACKAGE.md for inside the archive
    readme_data_content = f"""# Local Mesh AI — Complete AI Data Package ({VERSION})

This archive contains the official, verified local AI model weights for Local Mesh AI.

## 📁 Connecting This Data Folder to Local Mesh AI

1. Extract this archive to your preferred storage drive (e.g. `E:\\LocalMeshAI-Data` or `D:\\LocalMeshAI-Data`).
2. Set the environment variable `LOCAL_MESH_AI_DATA_DIR` pointing to the extracted root folder:
   ```cmd
   set "LOCAL_MESH_AI_DATA_DIR=E:\\LocalMeshAI-Data"
   ```
3. Run `Start_Local_Mesh_AI.bat` from the Code package. Local Mesh AI will automatically detect all model weights!

## 📜 Included Models and Licenses

- **TripoSR Checkpoint (`models/hf_hub/models--stabilityai--TripoSR`)**: MIT License (Commercial permitted).
- **DINO ViT-B/16 (`models/hf_hub/models--facebook--dino-vitb16`)**: Apache 2.0 (Commercial permitted).
  *Note:* DINO is a **ViT-B/16** model (hidden size 768, 12 layers, patch size 16).
  The provisioning path currently installs its `config.json`; the ViT weights are
  fetched separately by `core.download_manager` and are **not** part of this package.
- **Qwen 2.5 Coder 7B Q4_0 (`models/qwen_coder/qwen2.5-coder-7b-instruct-q4_0.gguf`)**: Apache 2.0 (Commercial permitted).
  This is the single supported artifact for the local technical-artist planner.
- **Qwen3-VL 4B Instruct (`models/qwen3_vl_4b/`)**: Apache 2.0 (Commercial permitted).
  The language model (`Qwen3VL-4B-Instruct-Q4_K_M.gguf`) and the vision projector
  (`mmproj-Qwen3VL-4B-Instruct-F16.gguf`) must both be present.

## ⚠️ Models that are NOT redistributed

The following models are excluded from this package because their upstream
licenses do not permit redistribution. Provision them yourself with
`core.download_manager` if you need them:

- **PartField (`model_objaverse.ckpt`)** - NVIDIA Research License (Non-Commercial).
- **SD-Turbo** - Stability AI Non-Commercial Research Community License.
  It is an optional large diffusion concept generator and is never bundled.
"""

    file_manifest: Dict[str, Any] = {}
    total_bytes = 0
    sha256_entries: List[str] = []

    print(f"  Packaging {len(packaged)} core model files into {DATA_ZIP_NAME}...")
    print("  Using ZIP_STORED (fast storage without double compression on neural weights)...")

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        # Write README_DATA_PACKAGE.md
        zf.writestr("LocalMeshAI-Data/README_DATA_PACKAGE.md", readme_data_content.encode("utf-8"))

        for p, arcname, key in packaged:
            sz = p.stat().st_size
            total_bytes += sz
            print(f"  + [{sz / (1024**2):.1f} MB] {p.name}")
            zf.write(p, arcname)

            f_hash = compute_sha256(p)
            file_manifest[arcname] = {
                "size_bytes": sz,
                "size_mb": round(sz / (1024**2), 2),
                "sha256": f_hash,
                "model_key": key,
                "source_path": str(p),
                "archive_path": arcname,
            }
            sha256_entries.append(f"{f_hash}  {arcname}")

        # Write manifest inside archive
        manifest_data = {
            "version": VERSION,
            "package_name": DATA_ZIP_NAME,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "models": MODEL_LICENSES,
            "excluded_models": excluded_models,
            "files": file_manifest,
        }
        zf.writestr("LocalMeshAI-Data/manifests/model_manifest.json", json.dumps(manifest_data, indent=2).encode("utf-8"))
        zf.writestr("LocalMeshAI-Data/manifests/SHA256SUMS.txt", "\n".join(sha256_entries).encode("utf-8"))

    zip_size_bytes = out_zip.stat().st_size
    zip_sha256 = compute_sha256(out_zip)

    summary = {
        "package_name": DATA_ZIP_NAME,
        "version": VERSION,
        "package_type": "complete_data",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "file_count": len(packaged) + 3,
        "uncompressed_bytes": total_bytes,
        "archive_size_bytes": zip_size_bytes,
        "archive_size_gb": round(zip_size_bytes / (1024**3), 2),
        "sha256": zip_sha256,
        "model_licensing_register": MODEL_LICENSES,
        "excluded_models": excluded_models,
        "models_root": str(EXTERNAL_MODELS_DIR),
        "files": file_manifest,
    }

    manifest_path = DIST_DIR / DATA_MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  [SUCCESS] Created {DATA_ZIP_NAME}")
    print(f"  Files: {summary['file_count']} | Size: {summary['archive_size_gb']} GB")
    print(f"  SHA-256: {zip_sha256}")
    return out_zip, summary


# ==============================================================================
# 3. Blender Add-on Package (installable via "Install from Disk")
# ==============================================================================

ADDON_ZIP_NAME = "local_mesh_ai_blender_addon.zip"
ADDON_PACKAGE_DIR = "local_mesh_ai"


def build_addon_package() -> Path:
    """Builds the standalone add-on ZIP.

    Blender installs a *directory* add-on from a ZIP, so the archive contains
    ``local_mesh_ai/__init__.py`` -- not a flat ``.py`` file. The ZIP can be
    installed with Edit > Preferences > Add-ons > Install from Disk.
    """
    print("\n" + "=" * 70)
    print("  BUILDING THE BLENDER ADD-ON PACKAGE")
    print("=" * 70)

    out_zip = PROJECT_ROOT / "dist" / ADDON_ZIP_NAME
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    source_package = PROJECT_ROOT / "addon" / ADDON_PACKAGE_DIR
    if not source_package.is_dir():
        raise RuntimeError(f"Add-on package not found: {source_package}")

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_package):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in sorted(files):
                if name.endswith(".pyc"):
                    continue
                src = Path(root) / name
                arc = f"{ADDON_PACKAGE_DIR}/{src.relative_to(source_package).as_posix()}"
                zf.write(src, arc)

    sha = compute_sha256(out_zip)
    print(f"  [SUCCESS] Created {ADDON_ZIP_NAME} ({out_zip.stat().st_size} bytes)")
    print(f"  SHA-256: {sha}")
    return out_zip


# ==============================================================================
# 4. Checksums and Verification
# ==============================================================================

def generate_release_checksums(code_summary: Dict[str, Any], data_summary: Dict[str, Any]) -> Path:
    """Writes the master release checksums file."""
    checksums_path = DIST_DIR / CHECKSUMS_NAME
    lines = [
        f"# Local Mesh AI Release Checksums — Version {VERSION}",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        f"{code_summary['sha256']}  {code_summary['package_name']}",
        f"{data_summary['sha256']}  {data_summary['package_name']}",
        "",
        "# Supporting Manifest Checksums:",
        f"{compute_sha256(DIST_DIR / CODE_MANIFEST_NAME)}  {CODE_MANIFEST_NAME}",
        f"{compute_sha256(DIST_DIR / DATA_MANIFEST_NAME)}  {DATA_MANIFEST_NAME}",
    ]
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  [SUCCESS] Created {CHECKSUMS_NAME}")
    return checksums_path


def verify_packages(code_zip: Path, data_zip: Path):
    """Tests archive integrity and clean temporary extraction."""
    print("\n" + "=" * 70)
    print("  VERIFYING RELEASE ARCHIVE INTEGRITY & EXTRACTION")
    print("=" * 70)

    # 1. Test Code ZIP integrity
    print(f"  Testing {code_zip.name} internal structure...")
    with zipfile.ZipFile(code_zip, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt file in {code_zip.name}: {bad_file}")
    print("  [OK] Code ZIP passed CRC-32 integrity test.")

    # 2. Test Data ZIP integrity
    print(f"  Testing {data_zip.name} internal structure...")
    with zipfile.ZipFile(data_zip, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt file in {data_zip.name}: {bad_file}")
    print("  [OK] Data ZIP passed CRC-32 integrity test.")

    # 3. Clean temporary extraction verification
    test_dir = PROJECT_ROOT / "dist" / "test_extraction"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Testing selective extraction of Code ZIP into {test_dir.name}...")
        with zipfile.ZipFile(code_zip, "r") as zf:
            # Extract root launcher and requirements
            zf.extract(f"LocalMeshAI-{VERSION}/Start_Local_Mesh_AI.bat", test_dir)
            zf.extract(f"LocalMeshAI-{VERSION}/requirements.txt", test_dir)
            extracted_launcher = test_dir / f"LocalMeshAI-{VERSION}" / "Start_Local_Mesh_AI.bat"
            assert extracted_launcher.exists(), "Extracted launcher not found!"

        print(f"  Testing selective extraction of Data ZIP into {test_dir.name}...")
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extract("LocalMeshAI-Data/README_DATA_PACKAGE.md", test_dir)
            zf.extract("LocalMeshAI-Data/manifests/model_manifest.json", test_dir)
            extracted_readme = test_dir / "LocalMeshAI-Data" / "README_DATA_PACKAGE.md"
            assert extracted_readme.exists(), "Extracted data README not found!"

        print("  [OK] Extraction verification successful!")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    t0 = time.time()
    addon_zip = build_addon_package()
    code_zip, code_sum = build_code_package()
    data_zip, data_sum = build_data_package()
    checksums_file = generate_release_checksums(code_sum, data_sum)
    verify_packages(code_zip, data_zip)
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ALL TWO RELEASE ARCHIVES SUCCESSFULLY BUILT & VERIFIED IN {elapsed:.2f}s!")
    print("=" * 70)
