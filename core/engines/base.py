"""
Local Mesh AI - Modular Multi-Engine Architecture
Base class and standard asset contracts for all 3D neural generation engines.
"""

import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path


class EngineStatus(str, Enum):
    READY = "ready"
    NEEDS_DOWNLOAD = "needs_download"
    UNSUPPORTED_HARDWARE = "unsupported_hardware"
    NOT_INSTALLED = "not_installed"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class StandardizedAsset:
    """Standardized internal representation of a generated 3D asset."""
    mesh_path: str
    format: str = "glb"
    vertex_count: int = 0
    face_count: int = 0
    dimensions: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (Width, Depth, Height) in meters
    materials: List[Dict[str, Any]] = field(default_factory=list)
    textures: Dict[str, str] = field(default_factory=dict)     # diffuse, normal, roughness, metallic
    has_uvs: bool = False
    has_vertex_colors: bool = False
    is_watertight: bool = False
    engine_id: str = "unknown"
    engine_name: str = "Unknown Engine"
    generation_time_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    blend_path: Optional[str] = None
    preview_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mesh_path": self.mesh_path,
            "format": self.format,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "dimensions_meters": list(self.dimensions),
            "materials": self.materials,
            "textures": self.textures,
            "has_uvs": self.has_uvs,
            "has_vertex_colors": self.has_vertex_colors,
            "is_watertight": self.is_watertight,
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "generation_time_seconds": round(self.generation_time_seconds, 2),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "blend_path": self.blend_path,
            "preview_image_path": self.preview_image_path,
            "diagnostics": self.diagnostics
        }


class Base3DEngine(ABC):
    """Abstract Base Class for all 3D neural generation and texturing engines."""

    def __init__(self, engine_id: str, name: str, description: str):
        self.engine_id = engine_id
        self.name = name
        self.description = description
        self._is_cancelled = False

    @property
    @abstractmethod
    def supported_modalities(self) -> List[str]:
        """Returns list of supported modalities: 'text-to-3d', 'image-to-3d', 'texturing'."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """
        Returns capability dict:
        {
            "pbr_textures": bool,
            "vertex_colors": bool,
            "uv_unwrapping": bool,
            "watertight": bool,
            "direct_text_conditioned": bool
        }
        """
        pass

    @property
    @abstractmethod
    def hardware_requirements(self) -> Dict[str, Any]:
        """
        Returns hardware requirements:
        {
            "min_vram_gb": float,
            "recommended_vram_gb": float,
            "requires_cuda": bool,
            "supports_cpu": bool,
            "recommended_precision": str  # 'fp16', 'bf16', 'fp32'
        }
        """
        pass

    @abstractmethod
    def get_status(self) -> EngineStatus:
        """Determines current engine availability based on dependencies, weights, and hardware."""
        pass

    @abstractmethod
    def get_status_details(self) -> Dict[str, Any]:
        """Returns detailed explanation of status (missing packages, weights path, hardware limits)."""
        pass

    @abstractmethod
    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        """Validates input data (file path or prompt string). Returns (is_valid, error_msg)."""
        pass

    @abstractmethod
    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        """Estimates peak VRAM required for the requested task and parameters."""
        pass

    @abstractmethod
    def generate(
        self,
        input_data: Any = None,
        output_dir: Optional[str] = None,
        modality: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        **kwargs
    ) -> StandardizedAsset:
        """Executes 3D generation and returns a StandardizedAsset."""
        pass

    def cancel(self):
        """Signals cancellation to active generation."""
        self._is_cancelled = True

    def cleanup(self):
        """Frees model weights from VRAM and host RAM."""
        self._is_cancelled = False
