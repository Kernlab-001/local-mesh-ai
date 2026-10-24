"""
Engine Registry and Intelligent Hardware-Adaptive Router for Local Mesh AI.
Maintains the registry of supported 3D engines, tracks installation and availability,
and dynamically routes requests to the optimal engine based on hardware capabilities.
"""

import os
import torch
from typing import Dict, Any, List, Optional, Tuple
from core.engines.base import Base3DEngine, EngineStatus
from core.engines.triposr_adapter import TripoSRAdapter
from core.engines.shape_adapter import ShapEAdapter
from core.engines.hybrid_text_adapter import HybridTextTo3DAdapter
from core.engines.sf3d_adapter import StableFast3DAdapter
from core.engines.trellis_adapter import TrellisAdapter
from core.hardware_detector import hardware_detector


class EngineRegistry:
    def __init__(self):
        self._engines: Dict[str, Base3DEngine] = {}
        self._register_default_engines()

    def _register_default_engines(self):
        """Registers all built-in and evaluated 3D neural engines."""
        triposr = TripoSRAdapter()
        self.register_engine(triposr)
        self.register_engine(ShapEAdapter())
        self.register_engine(HybridTextTo3DAdapter(secondary_engine=triposr))
        self.register_engine(StableFast3DAdapter())
        self.register_engine(TrellisAdapter())

    def register_engine(self, engine: Base3DEngine):
        self._engines[engine.engine_id] = engine

    def get_engine(self, engine_id: str) -> Optional[Base3DEngine]:
        return self._engines.get(engine_id)

    def list_engines(self, modality: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns metadata and live status for all registered engines."""
        result = []
        for eng in self._engines.values():
            if modality and modality not in eng.supported_modalities:
                continue
            details = eng.get_status_details()
            details["capabilities"] = eng.capabilities
            details["hardware_requirements"] = eng.hardware_requirements
            result.append(details)
        return result

    def get_dynamic_engine_choices(self) -> Tuple[List[str], Dict[str, str]]:
        """Dynamically queries engine registry and builds truthful label -> engine_id map."""
        choices = ["Auto-Detect (Adaptive Hardware Router)"]
        mapping = {"Auto-Detect (Adaptive Hardware Router)": "auto"}

        for eng in self.list_engines():
            status = eng["status"]
            if status == "ready":
                tag = "[READY]"
            elif status == "needs_download":
                tag = "[NEEDS DOWNLOAD]"
            elif status == "unsupported_hardware":
                req_vram = eng["hardware_requirements"].get("min_vram_gb", "HIGH")
                tag = f"[REQUIRES {req_vram}GB VRAM]"
            else:
                tag = f"[{status.upper()}]"

            label = f"{eng['name']} {tag}"
            choices.append(label)
            mapping[label] = eng["engine_id"]

        return choices, mapping

    def get_hardware_telemetry(self) -> Dict[str, Any]:
        """Probes live hardware VRAM and RAM availability."""
        has_cuda = torch.cuda.is_available()
        total_vram = 0.0
        avail_vram = 0.0
        device_name = "CPU Only"

        if has_cuda:
            props = torch.cuda.get_device_properties(0)
            total_vram = round(props.total_memory / (1024 ** 3), 2)
            device_name = torch.cuda.get_device_name(0)
            try:
                # Query free memory
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                avail_vram = round(free_bytes / (1024 ** 3), 2)
            except Exception:
                avail_vram = total_vram

        sys_ram = 8.0
        avail_ram = 4.0
        try:
            import psutil
            vm = psutil.virtual_memory()
            sys_ram = round(vm.total / (1024 ** 3), 2)
            avail_ram = round(vm.available / (1024 ** 3), 2)
        except Exception:
            pass

        return {
            "has_cuda": has_cuda,
            "device_name": device_name,
            "total_vram_gb": total_vram,
            "available_vram_gb": avail_vram,
            "total_ram_gb": sys_ram,
            "available_ram_gb": avail_ram
        }

    def resolve_best_engine(
        self,
        modality: str,
        profile: str = "balanced",
        user_override_engine: Optional[str] = None
    ) -> Tuple[Base3DEngine, Dict[str, Any]]:
        """
        Dynamically selects the optimal engine based on task, hardware profile, and available VRAM.
        Returns (selected_engine, decision_metadata).
        """
        hw = self.get_hardware_telemetry()
        avail_vram = hw["available_vram_gb"]

        # If user explicitly requested an engine, validate its viability
        if user_override_engine and user_override_engine in self._engines:
            eng = self._engines[user_override_engine]
            status = eng.get_status()
            req = eng.hardware_requirements.get("min_vram_gb", 4.0)

            warning = None
            if status == EngineStatus.UNSUPPORTED_HARDWARE:
                warning = f"Selected engine '{eng.name}' is marked unsupported on current hardware ({avail_vram} GB VRAM available, {req} GB required)."
            elif status == EngineStatus.NOT_INSTALLED:
                warning = f"Selected engine '{eng.name}' is missing dependencies."
            elif status == EngineStatus.NEEDS_DOWNLOAD:
                warning = f"Selected engine '{eng.name}' requires model weight download."

            return eng, {
                "engine_id": eng.engine_id,
                "name": eng.name,
                "reason": "User Manual Override",
                "warning": warning,
                "estimated_vram_gb": eng.estimate_vram_gb(modality),
                "hardware": hw
            }

        # Dynamic Engine Selection based on Hardware Profile & Task
        if modality == "image-to-3d":
            # On 16GB+ workstations, TRELLIS or SF3D could be chosen if ready
            if avail_vram >= 12.0 and profile == "high_quality" and self._engines["trellis"].get_status() == EngineStatus.READY:
                return self._engines["trellis"], {
                    "engine_id": "trellis",
                    "name": self._engines["trellis"].name,
                    "reason": "High-Quality Profile with >=12GB VRAM headroom.",
                    "hardware": hw
                }
            # Standard & Consumer NVIDIA GPUs (4GB - 12GB): TripoSR is the proven robust baseline
            return self._engines["triposr"], {
                "engine_id": "triposr",
                "name": self._engines["triposr"].name,
                "reason": "Optimal feedforward volumetric reconstruction calibrated for consumer GPU memory (3.0GB-6.0GB VRAM).",
                "hardware": hw
            }

        elif modality == "text-to-3d":
            if profile == "direct_latent":
                shap_eng = self._engines["shap_e"]
                status = shap_eng.get_status()
                warning = None
                if status == EngineStatus.NEEDS_DOWNLOAD:
                    warning = "Shap-E requires model weights download (~1.1GB openai/shap-e)."
                elif status != EngineStatus.READY:
                    warning = f"Shap-E engine status is '{status.value}'."

                return shap_eng, {
                    "engine_id": "shap_e",
                    "name": shap_eng.name,
                    "reason": "Direct text-conditioned 3D implicit surface generation (Strategy A).",
                    "status": status.value,
                    "warning": warning,
                    "hardware": hw
                }
            else:
                # Default Strategy B: Hybrid SD-Turbo concept art + TripoSR volumetric reconstruction
                return self._engines["hybrid_sd_triposr"], {
                    "engine_id": "hybrid_sd_triposr",
                    "name": self._engines["hybrid_sd_triposr"].name,
                    "reason": "Multi-stage visual concept synthesis with 4-view canonical projections and PBR appearance transfer (Strategy B).",
                    "hardware": hw
                }

        # Safe fallback
        fallback = self._engines.get("triposr") or list(self._engines.values())[0]
        return fallback, {
            "engine_id": fallback.engine_id,
            "name": fallback.name,
            "reason": "Default safe baseline fallback.",
            "hardware": hw
        }


engine_registry = EngineRegistry()
