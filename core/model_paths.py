"""Central Unified Model Path Resolver for Local Mesh AI.

Provides single-source-of-truth path resolution across all adapters, CLI tools,
and the model catalog. Supports local user config (.local_models_config.json),
environment variables (LOCAL_MESH_MODELS_ROOT, LOCAL_MESH_AI_MODELS_DIR, etc.),
and graceful automatic discovery.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ModelPathResolver")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ModelPathResolver:
    """Resolves and validates paths for all neural and geometric models."""

    _instance: Optional[ModelPathResolver] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self.config_file = config_path or (PROJECT_ROOT / ".local_models_config.json")
        self._local_config: Dict[str, Any] = self._load_local_config()
        self._initialized = True

    def _load_local_config(self) -> Dict[str, Any]:
        """Loads optional gitignored user configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse local models config: {e}")
        return {}

    def save_local_config(self, overrides: Dict[str, Any]) -> None:
        """Saves user path overrides to .local_models_config.json."""
        self._local_config.update(overrides)
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._local_config, f, indent=2)
            logger.info(f"Saved local models configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save local models config: {e}")

    def get_models_root(self) -> Path:
        """Resolves the external models storage directory."""
        if "models_root" in self._local_config:
            p = Path(self._local_config["models_root"])
            if p.exists():
                return p.resolve()

        for env_var in ["LOCAL_MESH_MODELS_ROOT", "LOCAL_MESH_AI_MODELS_DIR"]:
            if env_var in os.environ:
                p = Path(os.environ[env_var])
                if p.exists():
                    return p.resolve()

        candidates = [
            Path(r"E:\LocalMeshAI-Data\models"),
            Path.home() / ".local-mesh-ai" / "models",
            PROJECT_ROOT / "models",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()

        # Fallback to standard location
        fallback = Path(r"E:\LocalMeshAI-Data\models")
        return fallback

    def resolve_qwen_coder_path(self) -> Optional[Path]:
        """Resolves the GGUF file for Qwen 2.5 Coder."""
        if "qwen_coder_path" in self._local_config:
            p = Path(self._local_config["qwen_coder_path"])
            if p.exists() and p.is_file():
                return p.resolve()

        if "QWEN_CODER_PATH" in os.environ:
            p = Path(os.environ["QWEN_CODER_PATH"])
            if p.exists() and p.is_file():
                return p.resolve()

        candidates = [
            Path(r"E:\Game\Models\qwen2.5-coder-7b-instruct-q5_k_m.gguf"),
            Path(r"E:\New folder\qwen2.5-coder-3b-instruct-q6_k.gguf"),
            self.get_models_root() / "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
            self.get_models_root() / "qwen_coder" / "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            self.get_models_root() / "qwen_coder" / "qwen2.5-coder-3b-instruct-q6_k.gguf",
            PROJECT_ROOT / "models" / "qwen_coder" / "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def resolve_qwen3_vl_model_path(self) -> Optional[Path]:
        """Resolves the language model GGUF for Qwen3-VL 4B."""
        if "qwen3_vl_model_path" in self._local_config:
            p = Path(self._local_config["qwen3_vl_model_path"])
            if p.exists() and p.is_file():
                return p.resolve()

        if "QWEN3_VL_MODEL_PATH" in os.environ:
            p = Path(os.environ["QWEN3_VL_MODEL_PATH"])
            if p.exists() and p.is_file():
                return p.resolve()

        candidates = [
            self.get_models_root() / "qwen3_vl_4b" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            self.get_models_root() / "qwen3_vl_4b" / "qwen3-vl-4b-instruct-q4_k_m.gguf",
            Path(r"E:\LocalMeshAI-Data\models\qwen3_vl_4b\Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
            PROJECT_ROOT / "models" / "qwen3_vl_4b" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def resolve_qwen3_vl_mmproj_path(self) -> Optional[Path]:
        """Resolves the multimodal vision projector (mmproj) for Qwen3-VL."""
        if "qwen3_vl_mmproj_path" in self._local_config:
            p = Path(self._local_config["qwen3_vl_mmproj_path"])
            if p.exists() and p.is_file():
                return p.resolve()

        if "QWEN3_VL_MMPROJ_PATH" in os.environ:
            p = Path(os.environ["QWEN3_VL_MMPROJ_PATH"])
            if p.exists() and p.is_file():
                return p.resolve()

        candidates = [
            self.get_models_root() / "qwen3_vl_4b" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
            self.get_models_root() / "qwen3_vl_4b" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
            self.get_models_root() / "qwen3_vl_4b" / "mmproj-model-f16.gguf",
            Path(r"E:\LocalMeshAI-Data\models\qwen3_vl_4b\mmproj-Qwen3VL-4B-Instruct-F16.gguf"),
            PROJECT_ROOT / "models" / "qwen3_vl_4b" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def resolve_partfield_checkpoint(self) -> Optional[Path]:
        """Resolves the PartField checkpoint (model_objaverse.ckpt or partfield_objaverse.pt)."""
        if "partfield_checkpoint" in self._local_config:
            p = Path(self._local_config["partfield_checkpoint"])
            if p.exists() and p.is_file():
                return p.resolve()

        if "PARTFIELD_CHECKPOINT" in os.environ:
            p = Path(os.environ["PARTFIELD_CHECKPOINT"])
            if p.exists() and p.is_file():
                return p.resolve()

        candidates = [
            self.get_models_root() / "partfield" / "model_objaverse.ckpt",
            self.get_models_root() / "partfield" / "partfield_objaverse.pt",
            Path(r"E:\LocalMeshAI-Data\models\partfield\model_objaverse.ckpt"),
            Path(r"C:\Users\diyaj\Downloads\model_objaverse.ckpt"),
            PROJECT_ROOT / "workers" / "partfield" / "model_objaverse.ckpt",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def resolve_partfield_repo(self) -> Optional[Path]:
        """Resolves upstream PartField source directory containing partfield package."""
        if "partfield_repo_dir" in self._local_config:
            p = Path(self._local_config["partfield_repo_dir"])
            if p.exists() and (p / "partfield").exists():
                return p.resolve()

        if "PARTFIELD_REPO_DIR" in os.environ:
            p = Path(os.environ["PARTFIELD_REPO_DIR"])
            if p.exists() and (p / "partfield").exists():
                return p.resolve()

        candidates = [
            self.get_models_root() / "partfield" / "source" / "PartField-main" / "PartField-main",
            self.get_models_root() / "partfield" / "source" / "PartField-main",
            self.get_models_root() / "partfield" / "source",
            Path(r"E:\LocalMeshAI-Data\models\partfield\source\PartField-main\PartField-main"),
            PROJECT_ROOT / "workers" / "partfield" / "source" / "PartField-main" / "PartField-main",
        ]
        for c in candidates:
            if c.exists() and (c / "partfield").exists():
                return c.resolve()
        return None

    def resolve_partfield_python(self) -> str:
        """Resolves Python executable for PartField worker."""
        if "partfield_python_bin" in self._local_config:
            p = Path(self._local_config["partfield_python_bin"])
            if p.exists():
                return str(p.resolve())

        if "PARTFIELD_PYTHON_BIN" in os.environ:
            p = Path(os.environ["PARTFIELD_PYTHON_BIN"])
            if p.exists():
                return str(p.resolve())

        return sys.executable

    def resolve_triposr_checkpoint(self) -> Optional[Path]:
        """Resolves the TripoSR model checkpoint."""
        if "triposr_checkpoint" in self._local_config:
            p = Path(self._local_config["triposr_checkpoint"])
            if p.exists() and p.is_file():
                return p.resolve()

        if "TRIPOSR_CHECKPOINT" in os.environ:
            p = Path(os.environ["TRIPOSR_CHECKPOINT"])
            if p.exists() and p.is_file():
                return p.resolve()

        candidates = [
            self.get_models_root() / "hf_hub" / "models--stabilityai--TripoSR" / "snapshots" / "5b521936b01fbe1890f6f9baed0254ab6351c04a" / "model.ckpt",
            self.get_models_root() / "triposr" / "model.ckpt",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--TripoSR" / "snapshots" / "5b521936b01fbe1890f6f9baed0254ab6351c04a" / "model.ckpt",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def resolve_sd_turbo_path(self) -> Optional[Path]:
        """Resolves SD-Turbo weights directory or checkpoint."""
        if "sd_turbo_path" in self._local_config:
            p = Path(self._local_config["sd_turbo_path"])
            if p.exists():
                return p.resolve()

        if "SD_TURBO_PATH" in os.environ:
            p = Path(os.environ["SD_TURBO_PATH"])
            if p.exists():
                return p.resolve()

        candidates = [
            self.get_models_root() / "sd_turbo",
            self.get_models_root() / "stabilityai--sd-turbo",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--sd-turbo",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    def get_model_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Constructs an accurate inventory of all models with physical status."""
        coder_p = self.resolve_qwen_coder_path()
        triposr_p = self.resolve_triposr_checkpoint()
        sdturbo_p = self.resolve_sd_turbo_path()
        pf_repo = self.resolve_partfield_repo()
        pf_ckpt = self.resolve_partfield_checkpoint()
        vlm_model = self.resolve_qwen3_vl_model_path()
        vlm_proj = self.resolve_qwen3_vl_mmproj_path()

        def _file_info(p: Optional[Path]) -> Dict[str, Any]:
            if p and p.exists() and p.is_file():
                return {"path": str(p), "exists": True, "size_mb": round(p.stat().st_size / (1024 * 1024), 2)}
            return {"path": str(p) if p else None, "exists": False, "size_mb": 0.0}

        def _dir_info(p: Optional[Path]) -> Dict[str, Any]:
            if p and p.exists() and p.is_dir():
                return {"path": str(p), "exists": True}
            return {"path": str(p) if p else None, "exists": False}

        # Check llama_cpp availability
        llama_available = False
        try:
            lib_dir = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib"
            if hasattr(os, "add_dll_directory") and lib_dir.exists():
                os.add_dll_directory(str(lib_dir))
            import llama_cpp
            llama_available = True
        except Exception:
            pass

        return {
            "qwen_coder": {
                "name": "Qwen 2.5 Coder (Primary Planner)",
                "role": "primary_planner",
                **_file_info(coder_p),
                "runtime_available": llama_available,
                "status": "RUNTIME_READY" if (coder_p and llama_available) else ("VERIFIED_OFFLINE" if coder_p else "WEIGHTS_MISSING"),
            },
            "triposr": {
                "name": "TripoSR (Image-to-3D)",
                "role": "reconstruction",
                **_file_info(triposr_p),
                "runtime_available": True,
                "status": "RUNTIME_READY" if triposr_p else "WEIGHTS_MISSING",
            },
            "sd_turbo": {
                "name": "SD-Turbo (Text-to-Image Concept)",
                "role": "concept_generation",
                "path": str(sdturbo_p) if sdturbo_p else None,
                "exists": bool(sdturbo_p and sdturbo_p.exists()),
                "runtime_available": True,
                "status": "RUNTIME_READY" if (sdturbo_p and sdturbo_p.exists()) else "WEIGHTS_MISSING",
            },
            "partfield_source": {
                "name": "NVIDIA PartField Source Tree",
                "role": "specialist_3d_segmentation",
                **_dir_info(pf_repo),
                "runtime_available": True,
                "status": "FOUND" if pf_repo else "MISSING",
            },
            "partfield_checkpoint": {
                "name": "NVIDIA PartField Checkpoint (Objaverse)",
                "role": "specialist_3d_segmentation",
                **_file_info(pf_ckpt),
                "runtime_available": True,
                "status": "RUNTIME_READY" if (pf_ckpt and pf_repo) else ("VERIFIED_OFFLINE" if pf_ckpt else "WEIGHTS_MISSING"),
            },
            "qwen3_vl_model": {
                "name": "Qwen3-VL 4B Language GGUF",
                "role": "specialist_visual_perception",
                **_file_info(vlm_model),
                "runtime_available": llama_available,
                "status": "RUNTIME_READY" if (vlm_model and llama_available) else ("VERIFIED_OFFLINE" if vlm_model else "WEIGHTS_MISSING"),
            },
            "qwen3_vl_mmproj": {
                "name": "Qwen3-VL 4B Vision Projector (mmproj)",
                "role": "specialist_visual_perception",
                **_file_info(vlm_proj),
                "runtime_available": llama_available,
                "status": "RUNTIME_READY" if (vlm_proj and llama_available) else ("VERIFIED_OFFLINE" if vlm_proj else "WEIGHTS_MISSING"),
            },
        }


    resolve_qwen3_vl_path = resolve_qwen3_vl_model_path
    resolve_sdturbo_weights = resolve_sd_turbo_path

model_path_resolver = ModelPathResolver()
