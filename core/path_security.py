"""
Path Security and Controlled Staging Utilities for Local Mesh AI.
Enforces containment within authorized project and data directories,
prevents path traversal attacks, rejects malicious symlinks, and validates file formats and sizes.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_data_dir() -> Path:
    """Returns the base data storage directory for Local Mesh AI."""
    env_dir = os.environ.get("LOCAL_MESH_DATA_DIR")
    if env_dir:
        d = Path(env_dir).resolve()
    else:
        d = (PROJECT_ROOT / "data").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_staging_dir() -> Path:
    """Returns the controlled staging root for temporary mesh transfers and Blender operations."""
    staging = get_data_dir() / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def get_allowed_roots() -> List[Path]:
    """
    Returns the list of authorized directory roots that file accesses may target.
    Restricted strictly to application-owned project and data directories.
    General system temporary directory is explicitly excluded.
    """
    roots = [
        PROJECT_ROOT.resolve(),
        get_data_dir().resolve(),
        get_staging_dir().resolve(),
    ]
    unique_roots = []
    for r in roots:
        if r not in unique_roots:
            unique_roots.append(r)
    return unique_roots


DEFAULT_ALLOWED_EXTENSIONS = {
    ".glb", ".gltf", ".obj", ".mtl", ".fbx", ".ply", ".stl",
    ".png", ".jpg", ".jpeg", ".json", ".txt", ".bin", ".md", ".pt", ".safetensors"
}


def validate_safe_path(
    file_path: Union[str, Path],
    allowed_extensions: Optional[List[str]] = None,
    must_exist: bool = True,
    max_size_bytes: int = 500 * 1024 * 1024  # 500 MB
) -> Path:
    """
    Validates that a file path is safe, contained within authorized project roots,
    is not an unauthorized symlink, has a valid extension, and conforms to size limits.

    Raises ValueError or PermissionError on violation.
    """
    if not file_path or not str(file_path).strip():
        raise ValueError("File path cannot be empty or null.")

    p_str = str(file_path).strip()

    # Reject null bytes and path traversal patterns
    if "\0" in p_str:
        raise ValueError("Path contains invalid null byte.")

    path_obj = Path(p_str)

    # Resolve symlinks and canonical path
    try:
        resolved = path_obj.resolve()
    except Exception as e:
        raise ValueError(f"Failed to resolve path '{file_path}': {str(e)}")

    # Check for symlink traversal
    if path_obj.is_symlink():
        target = path_obj.readlink().resolve()
        # Ensure symlink target also stays within allowed roots
        allowed_roots = get_allowed_roots()
        if not any(target == r or r in target.parents for r in allowed_roots):
            raise PermissionError(f"Symlink target '{target}' escapes authorized directory roots.")

    # Containment check
    allowed_roots = get_allowed_roots()
    is_contained = any(resolved == r or r in resolved.parents for r in allowed_roots)
    if not is_contained:
        raise PermissionError(
            f"Access denied: Path '{resolved}' is not contained within authorized roots: "
            f"{[str(r) for r in allowed_roots]}"
        )

    # Extension check
    valid_exts = set(ext.lower() for ext in allowed_extensions) if allowed_extensions else DEFAULT_ALLOWED_EXTENSIONS
    suffix = resolved.suffix.lower()
    if valid_exts and suffix not in valid_exts:
        raise ValueError(f"Disallowed file extension '{suffix}'. Allowed: {sorted(list(valid_exts))}")

    if must_exist:
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: '{resolved}'")
        if not resolved.is_file():
            raise ValueError(f"Path is not a regular file: '{resolved}'")
        file_size = resolved.stat().st_size
        if file_size > max_size_bytes:
            raise ValueError(f"File size {file_size} bytes exceeds maximum allowed {max_size_bytes} bytes.")

    return resolved
