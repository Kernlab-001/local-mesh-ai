"""
TripoSR Engine & Neural Reconstruction Manager.
Validates inputs, coordinates GPU lease, runs marching cubes with memory safety,
and exports intermediate GLB meshes.
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional
from PIL import Image
from core.gpu_broker import gpu_broker

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TripoSREngine:
    def __init__(self, triposr_dir: Optional[str] = None, output_dir: Optional[str] = None):
        self.triposr_dir = triposr_dir or os.path.join(PROJECT_ROOT, "TripoSR")
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, "output")

    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """Validates image exists, is accessible, and meets dimension/format constraints."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        file_size = os.path.getsize(image_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Image file size ({file_size / (1024*1024):.2f} MB) exceeds 25 MB limit.")

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                if width < 64 or height < 64:
                    raise ValueError(f"Image dimensions ({width}x{height}) are too small. Minimum 64x64.")
                if width > 4096 or height > 4096:
                    raise ValueError(f"Image dimensions ({width}x{height}) exceed maximum 4096x4096.")
                return {
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_bytes": file_size,
                    "has_alpha": mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info)
                }
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {e}")

    def reconstruct(
        self,
        image_path: str,
        quality: str = "balanced",
        mc_resolution: int = 256,
        chunk_size: int = 8192,
        render_turntable: bool = False,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Runs TripoSR under global GPU lease and records benchmark metrics.
        """
        img_meta = self.validate_image(image_path)
        start_time = time.time()
        
        # Configure quality preset
        if quality == "draft":
            mc_resolution = 128
            chunk_size = 8192
        elif quality == "high":
            mc_resolution = 256
            chunk_size = 4096

        run_script = os.path.join(self.triposr_dir, "run.py")
        venv_python = os.path.join(os.path.dirname(self.output_dir), ".venv", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            venv_python = sys.executable

        job_mesh_dir = os.path.join(self.output_dir, "meshes")
        os.makedirs(job_mesh_dir, exist_ok=True)

        cmd = [
            venv_python, run_script,
            os.path.abspath(image_path),
            "--output-dir", job_mesh_dir,
            "--mc-resolution", str(mc_resolution),
            "--chunk-size", str(chunk_size),
            "--model-save-format", "glb",
            "--device", "cuda:0"
        ]
        if render_turntable:
            cmd.append("--render")

        # Acquire GPU Lease
        with gpu_broker.lease("TripoSR_Reconstruction", timeout=timeout):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.triposr_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"TripoSR subprocess failed:\n{proc.stderr}")
            finally:
                gpu_broker.purge_vram()

        elapsed = round(time.time() - start_time, 2)
        mesh_path = os.path.join(job_mesh_dir, "0", "mesh.glb")
        
        if not os.path.exists(mesh_path) or os.path.getsize(mesh_path) == 0:
            raise RuntimeError(f"Expected output mesh not created at {mesh_path}")

        return {
            "status": "success",
            "mesh_path": os.path.abspath(mesh_path),
            "elapsed_seconds": elapsed,
            "mesh_size_bytes": os.path.getsize(mesh_path),
            "mc_resolution": mc_resolution,
            "chunk_size": chunk_size,
            "image_meta": img_meta
        }

# Global engine instance
triposr_engine = TripoSREngine()
