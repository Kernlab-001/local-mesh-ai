"""
Microsoft TRELLIS Engine Adapter for Local Mesh AI.
Evaluates high-end structured latent diffusion 3D generation.
Requires 16GB-24GB VRAM and custom sparse CUDA extensions.
"""

import os
import torch
from typing import Dict, Any, List, Optional, Tuple, Callable
from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset


class TrellisAdapter(Base3DEngine):
    def __init__(self):
        super().__init__(
            engine_id="trellis",
            name="Microsoft TRELLIS (Structured Latent 3D)",
            description="State-of-the-art structured latent 3D diffusion for high-fidelity assets (Requires 16GB-24GB VRAM)."
        )

    @property
    def supported_modalities(self) -> List[str]:
        return ["image-to-3d", "text-to-3d"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "pbr_textures": True,
            "vertex_colors": True,
            "uv_unwrapping": True,
            "watertight": True,
            "direct_text_conditioned": True,
            "typical_generation_seconds": 30.0
        }

    @property
    def hardware_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 12.0,
            "recommended_vram_gb": 16.0,
            "requires_cuda": True,
            "supports_cpu": False,
            "recommended_precision": "fp16"
        }

    def get_status(self) -> EngineStatus:
        if not torch.cuda.is_available():
            return EngineStatus.UNSUPPORTED_HARDWARE

        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram < 11.5:
            return EngineStatus.UNSUPPORTED_HARDWARE

        try:
            import trellis
            return EngineStatus.READY
        except ImportError:
            return EngineStatus.NOT_INSTALLED

    def get_status_details(self) -> Dict[str, Any]:
        status = self.get_status()
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0
        reason = None
        if status == EngineStatus.UNSUPPORTED_HARDWARE:
            reason = f"Requires high-end GPU with at least 12.0 GB - 16.0 GB VRAM. Detected: {vram:.1f} GB."
        elif status == EngineStatus.NOT_INSTALLED:
            reason = "trellis package not installed. Requires Linux/WSL2 and custom CUDA sparse kernels (spconv, diffoctreerast)."

        return {
            "status": status.value,
            "engine_id": self.engine_id,
            "name": self.name,
            "ready": status == EngineStatus.READY,
            "detected_vram_gb": round(vram, 1),
            "unsupported_reason": reason,
            "requirements": self.hardware_requirements,
            "upstream_repo": "https://github.com/microsoft/TRELLIS",
            "license": "MIT License (Code) / Research weights"
        }

    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        if modality not in self.supported_modalities:
            return False, f"Modality '{modality}' not supported by TRELLIS."
        if modality == "image-to-3d" and (not input_data or not os.path.exists(str(input_data))):
            return False, "Input image not found."
        if modality == "text-to-3d" and (not input_data or len(str(input_data).strip()) < 2):
            return False, "Text prompt must be at least 2 characters."
        return True, None

    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        return 14.5

    def generate(
        self,
        input_data: Any = None,
        output_dir: Optional[str] = None,
        modality: Optional[str] = "image-to-3d",
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        **kwargs
    ) -> StandardizedAsset:
        status = self.get_status()
        if status != EngineStatus.READY:
            details = self.get_status_details()
            raise RuntimeError(f"Cannot execute TRELLIS: {details.get('unsupported_reason', 'Engine not ready')}")
        raise NotImplementedError("TRELLIS execution requires high-end 16GB+ VRAM GPU.")
