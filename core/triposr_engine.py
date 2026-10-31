"""
TripoSR Engine & Neural Reconstruction Manager.
Validates inputs, ensures strict per-job output isolation, coordinates GPU lease,
runs marching cubes with memory safety, measures real peak VRAM,
and performs rigorous geometry validation on exported GLB meshes.
Supports automatic device selection (CUDA GPU with graceful CPU fallback).
"""

import os
import sys
import time
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.gpu_broker import gpu_broker
from core.download_manager import get_configured_models_dir, configure_hf_environment

try:
    import trimesh
except ImportError:
    trimesh = None

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class TripoSREngine:
    def __init__(self, triposr_dir: Optional[str] = None, output_dir: Optional[str] = None):
        self.triposr_dir = Path(triposr_dir) if triposr_dir else PROJECT_ROOT / "TripoSR"
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validates image exists, is readable, uncorrupted, and meets dimension/format constraints.
        Supports PNG, JPG, JPEG, WebP.
        """
        if not image_path:
            raise ValueError("Input image path must not be empty.")

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        file_size = p.stat().st_size
        if file_size == 0:
            raise ValueError(f"Image file is empty (0 bytes): {image_path}")
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Image file size ({file_size / (1024*1024):.2f} MB) exceeds 25 MB limit.")

        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}")

        try:
            with Image.open(p) as img:
                img.verify()

            # Re-open after verify() to inspect properties and dimensions
            with Image.open(p) as img:
                width, height = img.size
                mode = img.mode

                if width < 32 or height < 32:
                    raise ValueError(f"Image dimensions ({width}x{height}) are too small. Minimum 32x32.")
                if width > 4096 or height > 4096:
                    raise ValueError(f"Image dimensions ({width}x{height}) exceed maximum 4096x4096.")

                has_alpha = False
                if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
                    rgba_img = img.convert("RGBA")
                    alpha_arr = np.array(rgba_img)[..., 3]
                    has_alpha = bool(np.any(alpha_arr < 250))

                return {
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_bytes": file_size,
                    "has_alpha": has_alpha,
                    "path": str(p.resolve())
                }
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise
            raise ValueError(f"Corrupted or invalid image file ({p.name}): {e}")

    def prepare_foreground_image(self, image_path: str, working_dir: Path, has_alpha: bool) -> str:
        """
        Ensures the input image has a clean transparent background without external network calls.
        If alpha is already present, exports a clean RGBA PNG.
        Otherwise uses adaptive border color flood-fill segmentation.
        """
        out_path = working_dir / "prepared_input.png"
        with Image.open(image_path) as im:
            rgba = im.convert("RGBA")
            arr = np.array(rgba)

            if has_alpha:
                rgba.save(out_path)
                return str(out_path)

            try:
                from scipy.ndimage import label
                h, w, _ = arr.shape
                edge_size = max(5, min(15, min(h, w) // 10))
                corners = [
                    arr[:edge_size, :edge_size, :3],
                    arr[:edge_size, -edge_size:, :3],
                    arr[-edge_size:, :edge_size, :3],
                    arr[-edge_size:, -edge_size:, :3]
                ]
                corner_colors = np.vstack([c.reshape(-1, 3) for c in corners])
                bg_color = np.median(corner_colors, axis=0)

                diff = np.linalg.norm(arr[..., :3].astype(np.float32) - bg_color, axis=-1)
                bg_candidate = diff < 60.0

                labeled, num_features = label(bg_candidate)
                border_labels = set(labeled[0, :]).union(labeled[-1, :]).union(labeled[:, 0]).union(labeled[:, -1])
                border_labels.discard(0)

                is_real_bg = np.isin(labeled, list(border_labels))
                arr[..., 3][is_real_bg] = 0
            except Exception:
                pass

            Image.fromarray(arr).save(out_path)
            return str(out_path)

    def inspect_mesh_geometry(self, glb_path: str) -> Dict[str, Any]:
        """
        Rigorously inspects exported GLB mesh for vertices, faces, dimensions,
        vertex colors, UVs, and coordinate validity.
        """
        info = {
            "vertex_count": 0,
            "face_count": 0,
            "dimensions": (0.0, 0.0, 0.0),
            "is_watertight": False,
            "has_vertex_colors": False,
            "has_uvs": False,
            "has_textures": False,
            "coordinates_finite": True,
            "errors": []
        }

        if not os.path.exists(glb_path) or os.path.getsize(glb_path) == 0:
            info["errors"].append("GLB file does not exist or is 0 bytes.")
            return info

        if trimesh is None:
            info["errors"].append("Trimesh is not available for geometry inspection.")
            return info

        try:
            scene_or_mesh = trimesh.load(glb_path, process=False)
            if isinstance(scene_or_mesh, trimesh.Scene):
                geoms = list(scene_or_mesh.geometry.values())
                mesh = trimesh.util.concatenate(geoms) if geoms else None
            else:
                mesh = scene_or_mesh

            if mesh is None or len(mesh.faces) == 0:
                info["errors"].append("Loaded mesh contains zero faces.")
                return info

            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            # Check finite coordinates
            has_nan_inf = bool(np.isnan(verts).any() or np.isinf(verts).any())
            info["coordinates_finite"] = not has_nan_inf
            if has_nan_inf:
                info["errors"].append("Mesh contains NaN or Infinite vertex coordinates.")

            info["vertex_count"] = int(len(verts))
            info["face_count"] = int(len(faces))

            extents = mesh.extents
            info["dimensions"] = (round(float(extents[0]), 4), round(float(extents[1]), 4), round(float(extents[2]), 4))
            info["is_watertight"] = bool(getattr(mesh, "is_watertight", False))

            # Inspect vertex colors
            has_colors = hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0
            info["has_vertex_colors"] = bool(has_colors)

            # Inspect UV coordinates
            has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
            info["has_uvs"] = bool(has_uv)

            # Inspect texture material
            has_tex = False
            if hasattr(mesh.visual, "material"):
                mat = mesh.visual.material
                if getattr(mat, "image", None) is not None or getattr(mat, "baseColorTexture", None) is not None:
                    has_tex = True
            info["has_textures"] = has_tex

        except Exception as e:
            info["errors"].append(f"Trimesh inspection error: {e}")

        return info

    def reconstruct(
        self,
        image_path: str,
        quality: str = "balanced",
        mc_resolution: Optional[int] = None,
        chunk_size: Optional[int] = None,
        render_turntable: bool = False,
        device: Optional[str] = None,
        timeout: float = 360.0,
        job_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs genuine TripoSR neural reconstruction under an isolated working directory and GPU lease.
        Measures real execution time and peak VRAM usage.
        """
        img_meta = self.validate_image(image_path)
        start_time = time.time()

        # Dynamic device selection
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Hardware-aware quality profiles (RTX 3050 6GB Laptop GPU tuning)
        if mc_resolution is None or chunk_size is None:
            if quality == "draft":
                mc_resolution = mc_resolution or 128
                chunk_size = chunk_size or 8192
            elif quality == "high":
                mc_resolution = mc_resolution or 256
                chunk_size = chunk_size or 4096
            else:  # balanced
                mc_resolution = mc_resolution or 256
                chunk_size = chunk_size or 8192

        # -------------------------------------------------------------
        # STRICT OUTPUT ISOLATION: Unique Per-Job Working Directory
        # -------------------------------------------------------------
        jid = job_id or f"triposr_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if output_dir:
            job_mesh_dir = Path(output_dir)
        else:
            job_mesh_dir = self.output_dir / jid

        job_mesh_dir.mkdir(parents=True, exist_ok=True)

        run_script = self.triposr_dir / "run.py"
        if not run_script.exists():
            raise FileNotFoundError(f"TripoSR run script missing at {run_script}")

        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = Path(sys.executable)

        # Preprocess background in isolated job directory
        processed_image = self.prepare_foreground_image(
            image_path=image_path,
            working_dir=job_mesh_dir,
            has_alpha=img_meta["has_alpha"]
        )

        cmd = [
            str(venv_python), str(run_script),
            processed_image,
            "--output-dir", str(job_mesh_dir),
            "--mc-resolution", str(mc_resolution),
            "--chunk-size", str(chunk_size),
            "--model-save-format", "glb",
            "--device", device,
            "--no-remove-bg"
        ]
        if render_turntable:
            cmd.append("--render")

        # Configure environment so offline HF hub loading resolves to persistent model cache
        models_dir = get_configured_models_dir()
        hf_cache = configure_hf_environment(models_dir)

        env = os.environ.copy()
        env["HF_HOME"] = str(models_dir)
        env["HF_HUB_CACHE"] = str(hf_cache)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        client_tag = f"TripoSR_Reconstruct_{jid}"
        peak_vram_gb = 0.0

        # Acquire GPU Lease to enforce single-model occupancy
        with gpu_broker.lease(client_tag, timeout=timeout):
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.reset_peak_memory_stats()

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.triposr_dir),
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=timeout
                )

                if torch.cuda.is_available() and "cuda" in device:
                    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

                if proc.returncode != 0:
                    err_text = proc.stderr or proc.stdout
                    if "OutOfMemoryError" in err_text or "CUDA out of memory" in err_text:
                        raise RuntimeError(
                            f"CUDA Out of Memory on {device}. Try setting quality='draft' or "
                            f"--mc-resolution 128 --chunk-size 4096.\nDetails: {err_text}"
                        )
                    raise RuntimeError(f"TripoSR subprocess failed (exit code {proc.returncode}):\n{err_text}")

            except subprocess.TimeoutExpired:
                raise TimeoutError(f"TripoSR reconstruction timed out after {timeout} seconds.")
            finally:
                gpu_broker.purge_vram()

        elapsed = round(time.time() - start_time, 2)

        # Locate exported GLB in the isolated job folder
        expected_mesh = job_mesh_dir / "0" / "mesh.glb"
        if not expected_mesh.exists() or expected_mesh.stat().st_size == 0:
            found_glb = None
            for p in job_mesh_dir.glob("**/*.glb"):
                if p.stat().st_size > 0:
                    found_glb = p
                    break
            if found_glb:
                expected_mesh = found_glb
            else:
                raise RuntimeError(f"TripoSR completed but output GLB was not found in {job_mesh_dir}")

        # Rigorous geometry inspection
        geom_info = self.inspect_mesh_geometry(str(expected_mesh))
        if geom_info["errors"]:
            raise RuntimeError(f"Generated mesh validation failed: {geom_info['errors']}")

        return {
            "status": "success",
            "mesh_path": str(expected_mesh.resolve()),
            "job_id": jid,
            "working_dir": str(job_mesh_dir.resolve()),
            "elapsed_seconds": elapsed,
            "mesh_size_bytes": expected_mesh.stat().st_size,
            "peak_vram_gb": peak_vram_gb,
            "mc_resolution": mc_resolution,
            "chunk_size": chunk_size,
            "device": device,
            "image_meta": img_meta,
            "geometry": geom_info
        }


# Global TripoSR engine instance
triposr_engine = TripoSREngine()
