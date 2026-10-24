"""
Stable Fast 3D (SF3D) Engine Adapter for Local Mesh AI.
Evaluates hardware compatibility, dependencies, and gating requirements.
Requires at least 8GB VRAM and C++ CUDA extensions.
"""

import os
import torch
from typing import Dict, Any, List, Optional, Tuple, Callable
from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset


class StableFast3DAdapter(Base3DEngine):
    def __init__(self):
        super().__init__(
            engine_id="stable_fast_3d",
            name="Stable Fast 3D (Stability AI)",
            description="Rapid 3D asset generation with UV unwrapping and material baking by Stability AI (Requires >=8GB VRAM)."
        )

    @property
    def supported_modalities(self) -> List[str]:
        return ["image-to-3d"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "pbr_textures": True,
            "vertex_colors": False,
            "uv_unwrapping": True,
            "watertight": True,
            "direct_text_conditioned": False,
            "typical_generation_seconds": 1.5
        }

    @property
    def hardware_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 8.0,
            "recommended_vram_gb": 12.0,
            "requires_cuda": True,
            "supports_cpu": False,
            "recommended_precision": "fp16"
        }

    def get_status(self) -> EngineStatus:
        if not torch.cuda.is_available():
            return EngineStatus.UNSUPPORTED_HARDWARE

        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram < 7.5:
            return EngineStatus.UNSUPPORTED_HARDWARE

        # Check if sf3d module is importable
        try:
            import sf3d
            return EngineStatus.READY
        except ImportError:
            return EngineStatus.NOT_INSTALLED

    def get_status_details(self) -> Dict[str, Any]:
        status = self.get_status()
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0
        reason = None
        if status == EngineStatus.UNSUPPORTED_HARDWARE:
            reason = f"Requires NVIDIA GPU with at least 8.0 GB VRAM. Detected: {vram:.1f} GB."
        elif status == EngineStatus.NOT_INSTALLED:
            reason = "sf3d package not installed. Requires MSVC C++ Build Tools and gated Hugging Face token."

        return {
            "status": status.value,
            "engine_id": self.engine_id,
            "name": self.name,
            "ready": status == EngineStatus.READY,
            "detected_vram_gb": round(vram, 1),
            "unsupported_reason": reason,
            "requirements": self.hardware_requirements,
            "upstream_repo": "https://github.com/Stability-AI/stable-fast-3d",
            "license": "Stability AI Community License"
        }

    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        if modality != "image-to-3d":
            return False, "SF3D only supports image-to-3d."
        if not input_data or not os.path.exists(str(input_data)):
            return False, "Input image file not found."
        return True, None

    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        return 7.8

    def generate(
        self,
        input_data: Any,
        output_dir: str,
        modality: str,
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> StandardizedAsset:
        status = self.get_status()
        if status != EngineStatus.READY:
            details = self.get_status_details()
            raise RuntimeError(f"Cannot execute Stable Fast 3D: {details.get('unsupported_reason', 'Engine not ready')}")
        raise NotImplementedError("Stable Fast 3D execution requires >=8GB VRAM and sf3d compiled extensions.")
