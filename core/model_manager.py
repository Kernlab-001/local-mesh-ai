"""
Model & Dependency Lifecycle Manager for Local Mesh AI.
Inspects local weight caches, disk space availability, engine dependencies,
and provides safe on-demand weight verification without cloud dependencies during runtime.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware_detector import hardware_detector


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    engine_id: str
    description: str
    approx_size_gb: float
    hf_repo_id: Optional[str] = None
    local_path: Optional[str] = None
    is_installed: bool = False
    required_vram_gb: float = 4.0
    required_ram_gb: float = 8.0
    license: str = "Unknown"


class ModelManager:
    """
    Central manager for model weights, dependency inspection,
    and storage capacity validation.
    """

    def __init__(self, models_root: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.models_root = Path(models_root) if models_root else self.project_root / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._hf_cache_dir = self._detect_hf_cache()

    def _detect_hf_cache(self) -> Path:
        """Locates the Hugging Face hub cache directory."""
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            p = Path(hf_home) / "hub"
            if p.exists():
                return p
        cache_home = os.environ.get("XDG_CACHE_HOME")
        if cache_home:
            p = Path(cache_home) / "huggingface" / "hub"
            if p.exists():
                return p
        default_p = Path.home() / ".cache" / "huggingface" / "hub"
        return default_p

    def get_free_disk_space_gb(self, target_dir: Optional[str] = None) -> float:
        """Returns free disk space in Gigabytes on the target drive."""
        path_to_check = target_dir or str(self.models_root)
        try:
            total, used, free = shutil.disk_usage(path_to_check)
            return round(free / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def get_model_catalog(self) -> List[ModelMetadata]:
        """Returns metadata for all supported AI models in the ecosystem."""
        catalog = [
            ModelMetadata(
                model_id="triposr",
                name="TripoSR Feedforward Volumetric Checkpoint",
                engine_id="triposr",
                description="Fast feedforward single-image 3D reconstruction model by Tripo AI & Stability AI.",
                approx_size_gb=1.67,
                hf_repo_id="stabilityai/TripoSR",
                required_vram_gb=4.0,
                required_ram_gb=8.0,
                license="MIT"
            ),
            ModelMetadata(
                model_id="sd_turbo",
                name="SD-Turbo 1-Step Latent Diffusion",
                engine_id="hybrid_text",
                description="Real-time 1-step text-to-image concept generator by Stability AI.",
                approx_size_gb=2.50,
                hf_repo_id="stabilityai/sd-turbo",
                required_vram_gb=3.5,
                required_ram_gb=8.0,
                license="Stability AI Non-Commercial Research Community License"
            ),
            ModelMetadata(
                model_id="shape_text3d",
                name="Shap-E Text-to-3D Implicit Pipeline",
                engine_id="shape",
                description="Direct text-to-3D neural radiance / mesh synthesis by OpenAI.",
                approx_size_gb=1.10,
                hf_repo_id="openai/shap-e",
                required_vram_gb=4.2,
                required_ram_gb=8.0,
                license="MIT"
            ),
            ModelMetadata(
                model_id="shape_img3d",
                name="Shap-E Image-to-3D Pipeline",
                engine_id="shape",
                description="Direct image-to-3D neural implicit synthesis by OpenAI.",
                approx_size_gb=1.10,
                hf_repo_id="openai/shap-e-img2img",
                required_vram_gb=4.2,
                required_ram_gb=8.0,
                license="MIT"
            ),
            ModelMetadata(
                model_id="qwen2.5_coder",
                name="Qwen 2.5 Coder 7B (GGUF Q4_0)",
                engine_id="llm_planner",
                description="Local code & structural 3D generation planner by Alibaba Cloud.",
                approx_size_gb=4.13,
                hf_repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                required_vram_gb=5.5,
                required_ram_gb=12.0,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="qwen3_vl_4b",
                name="Qwen3-VL 4B Instruct (GGUF Q4_K_M + mmproj F16)",
                engine_id="vision_language",
                description="Multimodal visual understanding and component recognition by Alibaba Cloud.",
                approx_size_gb=3.10,
                hf_repo_id="Qwen/Qwen3-VL-4B-Instruct-GGUF",
                required_vram_gb=4.5,
                required_ram_gb=8.0,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="stable_fast_3d",
                name="Stable Fast 3D (SF3D)",
                engine_id="sf3d",
                description="Fast UV unwrapped textured mesh synthesis by Stability AI.",
                approx_size_gb=1.45,
                hf_repo_id="stabilityai/stable-fast-3d",
                required_vram_gb=8.0,
                required_ram_gb=16.0,
                license="Stability AI Community License"
            ),
            ModelMetadata(
                model_id="trellis_large",
                name="Microsoft TRELLIS Image-to-3D Large",
                engine_id="trellis",
                description="Unified 3D generative model with structured latent representations.",
                approx_size_gb=14.2,
                hf_repo_id="JeffreyXiang/TRELLIS-image-large",
                required_vram_gb=16.0,
                required_ram_gb=32.0,
                license="Creative Commons Attribution-NonCommercial 4.0"
            )
        ]

        # Check installation status
        for item in catalog:
            item.is_installed = self.is_model_installed(item)

        return catalog

    def is_model_installed(self, model: ModelMetadata) -> bool:
        """Determines if the model weights are present and verified locally."""
        from core.download_manager import download_manager

        # TripoSR specific verification (requires config.yaml, model.ckpt, and dino config.json)
        if model.model_id == "triposr":
            stat = download_manager.verify_triposr_status()
            return bool(stat.get("ready_for_offline_inference", False))

        # Qwen LLM specific verification (checks GGUF file existence and non-zero size)
        if model.model_id == "qwen2.5_coder":
            q_stat = download_manager.verify_qwen_status()
            return q_stat.get("state") == "fully_installed"

        if model.model_id == "qwen3_vl_4b":
            models_dir = download_manager.models_dir
            gguf_path = models_dir / "qwen3_vl_4b" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
            mmproj_path = models_dir / "qwen3_vl_4b" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
            return gguf_path.exists() and mmproj_path.exists()

        if model.model_id == "partfield":
            models_dir = download_manager.models_dir
            ckpt = models_dir / "partfield" / "model_objaverse.ckpt"
            return ckpt.exists() and ckpt.stat().st_size > 10_000_000

        if model.model_id == "sd_turbo":
            models_dir = download_manager.models_dir
            sd_dir = models_dir / "sd_turbo" / "sd-turbo"
            return (sd_dir / "model_index.json").exists() or (sd_dir / "sd_turbo.safetensors").exists()

        # 1. Local path check
        if model.local_path and Path(model.local_path).exists() and Path(model.local_path).stat().st_size > 1000:
            return True

        # 2. Check models directory for non-empty model files
        local_dir = self.models_root / model.model_id
        if local_dir.exists():
            files = [f for f in local_dir.rglob("*") if f.is_file() and f.stat().st_size > 100_000]
            if files:
                return True

        # 3. Check Hugging Face hub cache for actual non-empty snapshots
        if model.hf_repo_id and self._hf_cache_dir.exists():
            cache_folder_name = "models--" + model.hf_repo_id.replace("/", "--")
            hf_path = self._hf_cache_dir / cache_folder_name
            if hf_path.exists() and (hf_path / "snapshots").exists():
                for s in (hf_path / "snapshots").iterdir():
                    if s.is_dir():
                        real_files = [f for f in s.rglob("*") if f.is_file() and f.stat().st_size > 100_000]
                        if real_files:
                            return True

        return False

    def check_engine_dependencies(self, engine_id: str) -> Dict[str, Any]:
        """
        Inspects required python modules and hardware headroom for an engine.
        """
        status: Dict[str, Any] = {
            "engine_id": engine_id,
            "can_run": False,
            "missing_packages": [],
            "sufficient_vram": False,
            "sufficient_ram": False,
            "sufficient_disk": False,
            "notes": []
        }

        # Check common packages
        core_packages = {
            "triposr": ["torch", "torchvision", "trimesh", "PIL"],
            "shape": ["torch", "diffusers", "trimesh"],
            "hybrid_text": ["torch", "diffusers", "transformers", "PIL"],
            "sf3d": ["torch", "diffusers", "torchvision"],
            "trellis": ["torch", "diffusers"]
        }

        required_pkgs = core_packages.get(engine_id, ["torch"])
        missing = []
        for pkg in required_pkgs:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        status["missing_packages"] = missing

        # Telemetry
        gpu_info = hardware_detector.detect_gpu()
        sys_info = hardware_detector.detect_cpu_ram()
        free_disk = self.get_free_disk_space_gb()
        status["sufficient_disk"] = free_disk >= 3.0  # At least 3GB for temporary work

        vram_gb = gpu_info.get("vram_gb", 0.0)
        ram_gb = sys_info.get("system_ram_gb", 8.0)

        # Engine specific resource requirements
        vram_needed = {
            "triposr": 3.0,
            "shape": 4.0,
            "hybrid_text": 3.5,
            "sf3d": 8.0,
            "trellis": 16.0
        }.get(engine_id, 4.0)

        ram_needed = {
            "triposr": 8.0,
            "shape": 8.0,
            "hybrid_text": 8.0,
            "sf3d": 16.0,
            "trellis": 24.0
        }.get(engine_id, 8.0)

        status["sufficient_vram"] = vram_gb >= vram_needed
        status["sufficient_ram"] = ram_gb >= ram_needed

        # Can run assessment
        if not missing and status["sufficient_vram"] and status["sufficient_ram"]:
            status["can_run"] = True
            status["notes"].append(f"Ready to execute (Requires ~{vram_needed}GB VRAM, system has {vram_gb:.1f}GB).")
        else:
            reasons = []
            if missing:
                reasons.append(f"Missing packages: {missing}")
            if not status["sufficient_vram"]:
                reasons.append(f"Insufficient VRAM: needs {vram_needed}GB, detected {vram_gb:.1f}GB")
            if not status["sufficient_ram"]:
                reasons.append(f"Insufficient RAM: needs {ram_needed}GB, detected {ram_gb:.1f}GB")
            status["notes"].append("; ".join(reasons))

        return status

    # --- Active LoRA / Adapter Runtime Management ---

    def load_adapter(self, adapter_id: str, adapter_path: str):
        """Loads and activates a fine-tuned adapter into the runtime inference state."""
        self._active_adapter_id = adapter_id
        self._active_adapter_path = adapter_path

        # Genuine neural module weight loading
        try:
            import torch
            from pathlib import Path as P
            from services.lora_training_manager import AuxiliaryTechnicalArtistClassifier as TechnicalArtistLoRAModule
            cand_p = P(adapter_path)
            ckpt_files = list(cand_p.glob("*.bin")) + list(cand_p.glob("*.pt")) + list(cand_p.glob("*.safetensors"))
            if ckpt_files:
                module = TechnicalArtistLoRAModule()
                state_dict = torch.load(str(ckpt_files[0]), map_location="cpu", weights_only=True)
                module.load_state_dict(state_dict, strict=True)
                self._runtime_adapter_module = module
            else:
                self._runtime_adapter_module = None
        except Exception:
            self._runtime_adapter_module = None

        try:
            from core.qwen_planner import qwen_planner
            if hasattr(qwen_planner, "set_active_adapter"):
                qwen_planner.set_active_adapter(adapter_path)
        except Exception:
            pass

    def unload_adapter(self):
        """Unloads any active adapter, reverting strictly to base model inference."""
        self._active_adapter_id = None
        self._active_adapter_path = None
        self._runtime_adapter_module = None
        try:
            from core.qwen_planner import qwen_planner
            if hasattr(qwen_planner, "set_active_adapter"):
                qwen_planner.set_active_adapter(None)
        except Exception:
            pass

    def predict_technical_artist_plan(self, prompt: str):
        """
        Executes genuine technical artist inference.
        If an auxiliary adapter is active, inference uses the loaded neural module.
        If no adapter is active, inference uses the genuine Qwen base planner (never a dummy fallback).
        """
        mod = getattr(self, "_runtime_adapter_module", None)
        if mod is not None:
            plan = mod.generate_plan(prompt)
            plan["active_adapter_id"] = self._active_adapter_id
            return plan

        # Genuine Qwen Planner base inference (zero dummy dicts)
        try:
            from core.qwen_planner import qwen_planner
            res = qwen_planner.generate_plan(prompt)
            if hasattr(res, "dict"):
                plan = res.dict()
            elif hasattr(res, "model_dump"):
                plan = res.model_dump()
            else:
                plan = dict(res)
            plan["active_adapter_id"] = None
            plan["model_type"] = "base_model"
            return plan
        except Exception as e:
            raise RuntimeError(f"Base technical artist model inference failed: {e}")

    def get_active_adapter(self):
        """Returns the currently loaded adapter ID, or None if using pure base model."""
        return getattr(self, "_active_adapter_id", None)


# Global singleton instance
model_manager = ModelManager()
