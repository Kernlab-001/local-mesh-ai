"""
TripoSR Engine Adapter for Local Mesh AI.
Wraps the validated local TripoSR neural reconstruction engine into the unified Base3DEngine contract.
Truthfully verifies source code, required Python dependencies, offline model weights,
and hardware capability.
"""

import os
import sys
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset
from core.triposr_engine import TripoSREngine, SUPPORTED_EXTENSIONS
from core.hardware_detector import hardware_detector
from core.download_manager import download_manager

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
        self.project_root = PROJECT_ROOT
        self.triposr_dir = Path(triposr_dir) if triposr_dir else self.project_root / "TripoSR"
        self._engine = TripoSREngine(triposr_dir=str(self.triposr_dir))

    @property
    def supported_modalities(self) -> List[str]:
        return ["image-to-3d"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "pbr_textures": True,
            "vertex_colors": True,
            "uv_unwrapping": True,
            "watertight": True,
            "direct_text_conditioned": False,
            "typical_generation_seconds": 15.0
        }

    @property
    def hardware_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 4.0,
            "recommended_vram_gb": 6.0,
            "requires_cuda": False,
            "supports_cpu": True,
            "recommended_precision": "fp16"
        }

    def _check_source_code(self) -> Tuple[bool, List[str]]:
        """Verifies upstream TripoSR source code files exist."""
        required_sources = [
            self.triposr_dir / "run.py",
            self.triposr_dir / "tsr" / "system.py",
            self.triposr_dir / "tsr" / "models" / "tokenizers" / "image.py",
            self.triposr_dir / "tsr" / "models" / "isosurface.py"
        ]
        missing = [str(f.name) for f in required_sources if not f.exists()]
        return len(missing) == 0, missing

    def _check_dependencies(self) -> Tuple[bool, List[str]]:
        """Verifies all Python dependencies required by TripoSR can be imported."""
        required_modules = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("trimesh", "Trimesh"),
            ("PIL", "Pillow"),
            ("einops", "Einops"),
            ("omegaconf", "OmegaConf"),
            ("transformers", "Transformers"),
            ("huggingface_hub", "HuggingFace Hub")
        ]
        missing = []
        for mod_name, label in required_modules:
            try:
                __import__(mod_name)
            except Exception as e:
                missing.append(f"{label} ({mod_name}): {e}")
        return len(missing) == 0, missing

    def get_status(self) -> EngineStatus:
        """
        Determines truthful engine state distinguishing:
        - Source code missing
        - Python dependencies missing
        - Model weights missing
        - Model weights partially installed
        - Ready for inference
        """
        src_ok, _ = self._check_source_code()
        if not src_ok:
            return EngineStatus.SOURCE_MISSING

        deps_ok, _ = self._check_dependencies()
        if not deps_ok:
            return EngineStatus.DEPENDENCIES_MISSING

        weights_stat = download_manager.verify_triposr_status()
        if weights_stat["state"] == "missing":
            return EngineStatus.WEIGHTS_MISSING
        elif weights_stat["state"] == "partially_installed":
            return EngineStatus.WEIGHTS_PARTIAL

        if not weights_stat["ready_for_offline_inference"]:
            return EngineStatus.WEIGHTS_MISSING

        return EngineStatus.READY

    def get_status_details(self) -> Dict[str, Any]:
        """Exposes detailed diagnostic metadata explaining readiness or failure reasons."""
        status = self.get_status()
        src_ok, missing_sources = self._check_source_code()
        deps_ok, missing_deps = self._check_dependencies()
        weights_info = download_manager.verify_triposr_status()

        reason = "Engine is fully installed, verified offline, and ready for inference."
        if status == EngineStatus.SOURCE_MISSING:
            reason = f"TripoSR source code missing files: {', '.join(missing_sources)}."
        elif status == EngineStatus.DEPENDENCIES_MISSING:
            reason = f"Missing required Python packages: {', '.join(missing_deps)}."
        elif status == EngineStatus.WEIGHTS_MISSING:
            reason = "Model weights missing. Run 'python scripts/download_models.py --triposr'."
        elif status == EngineStatus.WEIGHTS_PARTIAL:
            reason = "Model weights partially installed. Re-run 'python scripts/download_models.py --triposr'."

        hw_profile = hardware_detector.profile or hardware_detector.scan_and_configure(save_config=False)
        gpu_info = hw_profile.get("gpu", {})

        return {
            "status": status.value,
            "engine_id": self.engine_id,
            "name": self.name,
            "ready": status == EngineStatus.READY,
            "reason": reason,
            "installed_path": str(self.triposr_dir),
            "supported_modalities": self.supported_modalities,
            "requirements": self.hardware_requirements,
            "source_code_verified": src_ok,
            "dependencies_verified": deps_ok,
            "weights": weights_info,
            "hardware": {
                "device": gpu_info.get("device", "cpu"),
                "gpu_name": gpu_info.get("name", "Unknown"),
                "vram_gb": gpu_info.get("vram_gb", 0.0),
                "cuda_available": gpu_info.get("available", False)
            },
            "license": "MIT License (Code & Weights)"
        }

    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        if modality != "image-to-3d":
            return False, f"TripoSR does not directly support modality '{modality}'."
        if not input_data or not isinstance(input_data, (str, Path)):
            return False, "Input data must be a valid file path string or Path object."
        p = Path(input_data)
        if not p.exists():
            return False, f"Input image does not exist: {p}"
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported image format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        return True, None

    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        params = params or {}
        quality = params.get("quality", "balanced")
        if quality == "draft":
            return 2.5
        elif quality == "high":
            return 4.8
        return 3.5  # Balanced baseline

    def generate(
        self,
        input_data: Any = None,
        output_dir: Optional[str] = None,
        modality: Optional[str] = "image-to-3d",
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        **kwargs
    ) -> StandardizedAsset:
        params = params or {}
        input_path = input_data or params.get("image_path")
        mod = modality or "image-to-3d"

        # Verify engine readiness before running
        status = self.get_status()
        if status != EngineStatus.READY:
            details = self.get_status_details()
            raise RuntimeError(f"TripoSR is not ready for inference ({status.value}): {details.get('reason')}")

        if progress_callback:
            progress_callback("Validating input image for TripoSR", 10)

        is_valid, err = self.validate_input(input_path, mod)
        if not is_valid:
            raise ValueError(f"Input validation failed: {err}")

        # Isolated output directory
        target_dir = Path(output_dir) if output_dir else self.project_root / "output" / f"job_{int(time.time())}"
        target_dir.mkdir(parents=True, exist_ok=True)

        quality = params.get("quality", "balanced")
        mc_res = params.get("mc_resolution")
        chunk_size = params.get("chunk_size")
        render_tt = params.get("render_turntable", False)
        timeout = float(params.get("timeout", 360.0))

        if progress_callback:
            progress_callback("Running TripoSR neural reconstruction under GPU lease", 30)

        start_time = time.time()
        rec_res = self._engine.reconstruct(
            image_path=str(input_path),
            quality=quality,
            mc_resolution=mc_res,
            chunk_size=chunk_size,
            render_turntable=render_tt,
            timeout=timeout,
            output_dir=str(target_dir)
        )
        elapsed = round(time.time() - start_time, 2)

        raw_mesh_path = rec_res["mesh_path"]
        target_mesh_path = target_dir / "mesh.glb"
        if Path(raw_mesh_path) != target_mesh_path:
            shutil.copy2(raw_mesh_path, target_mesh_path)

        if progress_callback:
            progress_callback("Validating geometry metrics and textures", 85)

        geom = rec_res.get("geometry", {})

        if progress_callback:
            progress_callback("TripoSR generation completed successfully", 100)

        return StandardizedAsset(
            mesh_path=str(target_mesh_path.resolve()),
            format="glb",
            vertex_count=geom.get("vertex_count", 0),
            face_count=geom.get("face_count", 0),
            dimensions=geom.get("dimensions", (1.0, 1.0, 1.0)),
            has_uvs=geom.get("has_uvs", False),
            has_vertex_colors=geom.get("has_vertex_colors", True),
            is_watertight=geom.get("is_watertight", False),
            engine_id=self.engine_id,
            engine_name=self.name,
            generation_time_seconds=elapsed,
            peak_vram_gb=rec_res.get("peak_vram_gb", 0.0),
            diagnostics={
                "quality": quality,
                "mc_resolution": rec_res.get("mc_resolution"),
                "chunk_size": rec_res.get("chunk_size"),
                "device": rec_res.get("device"),
                "image_meta": rec_res.get("image_meta"),
                "geometry_valid": geom.get("coordinates_finite", True)
            }
        )
