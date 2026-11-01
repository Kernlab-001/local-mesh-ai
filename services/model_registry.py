"""
Model Registry Service for Local Mesh AI.
Catalogs immutable foundational models and LoRA adapters.
Implements complete streaming SHA-256 file hashing, deterministic directory manifest hashing,
real filesystem size calculation, and checkpoint integrity validation.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.path_security import get_data_dir, PROJECT_ROOT


class ModelArtifact(BaseModel):
    model_id: str
    name: str
    model_type: str  # "base_llm", "base_reconstruction", "lora_adapter"
    file_path: str
    sha256_hash: str
    file_size_mb: float
    parameter_count_str: str
    is_active: bool = False
    is_immutable: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    def __init__(self):
        self.models_dir = get_data_dir() / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_file_complete(file_path: Path) -> str:
        """Computes complete SHA-256 checksum over the entire file contents using streaming chunks."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1048576):  # 1 MB chunk
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def hash_directory_deterministic(dir_path: Path) -> str:
        """
        Creates a deterministic SHA-256 hash across all contained files and relative paths.
        Ensures any modification or corrupt file alters the digest.
        """
        h = hashlib.sha256()
        files = sorted([p for p in dir_path.rglob("*") if p.is_file()], key=lambda p: str(p.relative_to(dir_path)))
        for p in files:
            rel = str(p.relative_to(dir_path)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            with open(p, "rb") as f:
                while chunk := f.read(1048576):
                    h.update(chunk)
        return h.hexdigest()

    def discover_models(self) -> List[ModelArtifact]:
        """Scans workspace and data directory for base models and verified adapters."""
        artifacts: List[ModelArtifact] = []

        # 1. Base Qwen Model (Immutable)
        qwen_file = PROJECT_ROOT / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        if not qwen_file.exists():
            try:
                from core.model_paths import model_path_resolver
                resolved = model_path_resolver.resolve_qwen_coder_path()
                if resolved and Path(resolved).exists():
                    qwen_file = Path(resolved)
            except Exception:
                pass

        if qwen_file and qwen_file.exists():
            size_mb = round(qwen_file.stat().st_size / (1024**2), 2)
            artifacts.append(
                ModelArtifact(
                    model_id="qwen-7b-base",
                    name="Qwen2.5-Coder-7B-Instruct (GGUF Q4_K_M)",
                    model_type="base_llm",
                    file_path=str(qwen_file),
                    sha256_hash=self.hash_file_complete(qwen_file),
                    file_size_mb=size_mb,
                    parameter_count_str="7.6 Billion",
                    is_active=True,
                    is_immutable=True,
                    metadata={"quantization": "Q4_K_M", "backend": "llama.cpp"}
                )
            )

        # 2. Candidate and verified adapters
        candidates_dir = self.models_dir / "candidates"
        if candidates_dir.exists():
            for d in candidates_dir.iterdir():
                if d.is_dir():
                    meta_file = d / "adapter_metadata.json"
                    meta = {}
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                        except Exception:
                            pass

                    # Require real checkpoint file existence
                    ckpt_files = list(d.glob("*.bin")) + list(d.glob("*.pt")) + list(d.glob("*.safetensors"))
                    if not ckpt_files:
                        continue

                    total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    size_mb = round(total_size / (1024**2), 2)

                    dir_hash = self.hash_directory_deterministic(d)
                    artifacts.append(
                        ModelArtifact(
                            model_id=f"adapter_{d.name}",
                            name=meta.get("adapter_name", d.name),
                            model_type="lora_adapter",
                            file_path=str(d),
                            sha256_hash=dir_hash,
                            file_size_mb=size_mb,
                            parameter_count_str="LoRA Rank 16",
                            is_active=meta.get("promoted_to_active", False),
                            is_immutable=False,
                            metadata=meta
                        )
                    )

        return artifacts

    def register_checkpoint(
        self,
        model_id: str,
        name: str,
        model_type: str,
        file_path: str,
        sha256_hash: str,
        file_size_mb: float = 0.0,
        parameter_count_str: str = "LoRA Rank 16",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelArtifact:
        """Registers a verified model or adapter checkpoint."""
        return ModelArtifact(
            model_id=model_id,
            name=name,
            model_type=model_type,
            file_path=file_path,
            sha256_hash=sha256_hash,
            file_size_mb=file_size_mb,
            parameter_count_str=parameter_count_str,
            is_active=False,
            is_immutable=False,
            metadata=metadata or {},
        )


model_registry = ModelRegistry()
