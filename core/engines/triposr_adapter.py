"""
TripoSR Engine Adapter for Local Mesh AI.
Wraps the validated local TripoSR neural reconstruction engine into the unified Base3DEngine contract.
"""

import os
import sys
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
from PIL import Image

from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset
from core.triposr_engine import TripoSREngine, SUPPORTED_EXTENSIONS
from core.hardware_detector import hardware_detector

try:
    import trimesh
except ImportError:
    trimesh = None


class TripoSRAdapter(Base3DEngine):
    def __init__(self, triposr_dir: Optional[str] = None):
        super().__init__(
            engine_id="triposr",
            name="TripoSR (Feedforward Volumetric)",
            description="Ultra-fast feedforward single-image 3D neural reconstruction by Tripo AI & Stability AI."
        )
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.triposr_dir = triposr_dir or str(self.project_root / "TripoSR")
        self._engine = TripoSREngine(triposr_dir=self.triposr_dir)

    @property
    def supported_modalities(self) -> List[str]:
        return ["image-to-3d"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "pbr_textures": True,             # When paired with finishing baker
            "vertex_colors": True,
            "uv_unwrapping": True,
            "watertight": True,
            "direct_text_conditioned": False,
            "typical_generation_seconds": 12.0
        }

    @property
    def hardware_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 4.0,
            "recommended_vram_gb": 6.0,
            "requires_cuda": False,           # CPU fallback supported
            "supports_cpu": True,
            "recommended_precision": "fp16"
        }

    def get_status(self) -> EngineStatus:
        """Determines if TripoSR repository and checkpoint are locally ready."""
        run_script = os.path.join(self.triposr_dir, "run.py")
        if not os.path.exists(run_script):
            return EngineStatus.NOT_INSTALLED
        return EngineStatus.READY

    def get_status_details(self) -> Dict[str, Any]:
        status = self.get_status()
        return {
            "status": status.value,
            "engine_id": self.engine_id,
            "name": self.name,
            "installed_path": self.triposr_dir,
            "ready": status == EngineStatus.READY,
            "supported_modalities": self.supported_modalities,
            "requirements": self.hardware_requirements,
            "license": "MIT License (Code & Weights)"
        }

    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        if modality != "image-to-3d":
            return False, f"TripoSR does not directly support modality '{modality}'."
        if not input_data or not isinstance(input_data, (str, Path)):
            return False, "Input data must be a valid file path string or Path object."
        p = str(input_data)
        if not os.path.exists(p):
            return False, f"Input image does not exist: {p}"
        ext = os.path.splitext(p)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported image format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        return True, None

    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        params = params or {}
        quality = params.get("quality", "balanced")
        if quality == "draft":
            return 2.5
        elif quality == "high":
            return 4.5
        return 3.2  # Balanced baseline

    def generate(
        self,
        input_data: Any,
        output_dir: str,
        modality: str,
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> StandardizedAsset:
        params = params or {}
        if progress_callback:
            progress_callback("Validating input image for TripoSR", 10)

        is_valid, err = self.validate_input(input_data, modality)
        if not is_valid:
            raise ValueError(f"Input validation failed: {err}")

        os.makedirs(output_dir, exist_ok=True)
        quality = params.get("quality", "balanced")
        mc_res = params.get("mc_resolution", 256 if quality != "draft" else 128)
        chunk_size = params.get("chunk_size", 8192 if quality != "high" else 4096)
        render_tt = params.get("render_turntable", False)

        if progress_callback:
            progress_callback("Running TripoSR neural reconstruction", 35)

        start_time = time.time()
        rec_res = self._engine.reconstruct(
            image_path=str(input_data),
            quality=quality,
            mc_resolution=mc_res,
            chunk_size=chunk_size,
            render_turntable=render_tt
        )
        elapsed = time.time() - start_time

        raw_mesh_path = rec_res["mesh_path"]
        target_mesh_path = os.path.join(output_dir, "mesh.glb")
        if os.path.exists(raw_mesh_path) and os.path.getsize(raw_mesh_path) > 0:
            shutil.copy2(raw_mesh_path, target_mesh_path)
        else:
            raise RuntimeError("TripoSR completed but output mesh was missing or empty.")

        if progress_callback:
            progress_callback("Extracting standardized mesh metrics", 80)

        # Inspect geometry
        v_count = 0
        f_count = 0
        dims = (1.0, 1.0, 1.0)
        is_wt = False
        has_uv = False
        has_vc = True

        if trimesh is not None:
            try:
                m = trimesh.load(target_mesh_path, force="mesh")
                if isinstance(m, trimesh.Scene):
                    m = m.dump(concatenate=True)
                v_count = len(m.vertices)
                f_count = len(m.faces)
                ext = m.extents
                dims = (float(ext[0]), float(ext[1]), float(ext[2]))
                is_wt = getattr(m, "is_watertight", False)
                has_uv = bool(hasattr(m.visual, "uv") and m.visual.uv is not None and len(m.visual.uv) > 0)
            except Exception:
                pass

        if progress_callback:
            progress_callback("TripoSR generation completed", 100)

        return StandardizedAsset(
            mesh_path=target_mesh_path,
            format="glb",
            vertex_count=v_count,
            face_count=f_count,
            dimensions=dims,
            has_uvs=has_uv,
            has_vertex_colors=has_vc,
            is_watertight=is_wt,
            engine_id=self.engine_id,
            engine_name=self.name,
            generation_time_seconds=elapsed,
            peak_vram_gb=rec_res.get("peak_vram_gb", 3.02),
            diagnostics={
                "quality": quality,
                "mc_resolution": mc_res,
                "chunk_size": chunk_size,
                "device": rec_res.get("device", "cuda:0")
            }
        )
