"""
Model Provisioning and Download Manager for Local Mesh AI.
Provides robust, verifiable model weight downloads using huggingface_hub with:
1. Configurable persistent model cache outside Git repository.
2. Genuine download progress and resume capability.
3. Disk space safety checks before downloading.
4. Offline verification using local_files_only=True.
5. Provenance manifest tracking installed model files, sizes, and revisions.
6. Zero repeated downloads for already verified models.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default external data directory for models
DEFAULT_MODELS_DIR = Path("E:/LocalMeshAI-Data/models") if Path("E:/").exists() else Path.home() / ".local_mesh_ai" / "models"


def get_configured_models_dir() -> Path:
    """Resolves model cache directory from env var, config.json, or default external path."""
    env_models = os.environ.get("LOCAL_MESH_AI_MODELS_DIR")
    if env_models:
        p = Path(env_models)
        p.mkdir(parents=True, exist_ok=True)
        return p

    data_dir = os.environ.get("LOCAL_MESH_AI_DATA_DIR") or os.environ.get("LOCAL_MESH_DATA_DIR")
    if data_dir:
        candidate = Path(data_dir) / "models"
        p = candidate if candidate.exists() else Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            custom_dir = cfg.get("models_cache_dir")
            if custom_dir:
                p = Path(custom_dir)
                p.mkdir(parents=True, exist_ok=True)
                return p
        except Exception:
            pass

    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_MODELS_DIR


def configure_hf_environment(models_dir: Optional[Path] = None):
    """Sets Hugging Face cache environment variables to point to persistent cache."""
    target_dir = models_dir or get_configured_models_dir()
    hf_cache = target_dir / "hf_hub"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(target_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    return hf_cache


class DownloadManager:
    """
    Manages downloading, verifying, and provisioning weights for TripoSR, DINO, and Qwen.
    """

    TRIPOSR_FILES = [
        {"repo_id": "stabilityai/TripoSR", "filename": "config.yaml", "approx_bytes": 2048},
        {"repo_id": "stabilityai/TripoSR", "filename": "model.ckpt", "approx_bytes": 1_790_000_000},
        {"repo_id": "facebook/dino-vitb16", "filename": "config.json", "approx_bytes": 2048}
    ]

    QWEN_GGUF_FILE = {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "approx_bytes": 4_680_000_000
    }

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or get_configured_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache = configure_hf_environment(self.models_dir)
        self.manifest_path = self.models_dir / "provisioned_models.json"

    def get_free_disk_space_gb(self) -> float:
        """Returns free disk space in GB on the model storage drive."""
        try:
            total, used, free = shutil.disk_usage(self.models_dir)
            return round(free / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def load_manifest(self) -> Dict[str, Any]:
        """Loads persistent model installation manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_manifest(self, manifest: Dict[str, Any]):
        """Saves persistent model installation manifest."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[Warning] Could not save model manifest: {e}")

    def verify_file_offline(self, repo_id: str, filename: str) -> Optional[Path]:
        """
        Verifies whether a file can be resolved strictly offline from cache.
        Returns the resolved Path if valid and non-empty, otherwise None.
        """
        try:
            from huggingface_hub import hf_hub_download
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.hf_cache),
                local_files_only=True
            )
            p = Path(resolved)
            if p.exists() and p.stat().st_size > 0:
                return p
        except Exception:
            pass
        return None

    def verify_triposr_status(self) -> Dict[str, Any]:
        """
        Truthfully checks TripoSR and DINO weights installation status.
        Distinguishes missing, partial, and fully installed states.
        """
        results = {}
        all_present = True
        any_present = False

        for item in self.TRIPOSR_FILES:
            repo = item["repo_id"]
            fn = item["filename"]
            key = f"{repo}/{fn}"
            resolved = self.verify_file_offline(repo, fn)
            if resolved:
                any_present = True
                results[key] = {
                    "installed": True,
                    "path": str(resolved),
                    "size_bytes": resolved.stat().st_size
                }
            else:
                all_present = False
                results[key] = {
                    "installed": False,
                    "path": None,
                    "size_bytes": 0
                }

        if all_present:
            state = "fully_installed"
        elif any_present:
            state = "partially_installed"
        else:
            state = "missing"

        return {
            "state": state,
            "ready_for_offline_inference": all_present,
            "cache_dir": str(self.hf_cache),
            "files": results
        }

    def verify_qwen_status(self) -> Dict[str, Any]:
        """
        Checks local Qwen GGUF model availability across known locations and cache.
        """
        # 1. Check configured path in config.json
        config_path = PROJECT_ROOT / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                custom_qwen = cfg.get("qwen_model_path")
                if custom_qwen:
                    p = Path(custom_qwen)
                    if not p.is_absolute():
                        p = PROJECT_ROOT / custom_qwen
                    if p.exists() and p.stat().st_size > 100_000_000:
                        return {
                            "state": "fully_installed",
                            "model_path": str(p),
                            "size_bytes": p.stat().st_size,
                            "source": "config_custom_path"
                        }
            except Exception:
                pass

        # 2. Check known locations in models_dir and configured external paths
        candidate_paths = [
            self.models_dir / "qwen_coder" / "qwen2.5-coder-7b-instruct-q4_0.gguf",
            self.models_dir / "qwen_coder" / "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            self.models_dir / "qwen" / "qwen2.5-coder-7b-instruct-q4_0.gguf",
            self.models_dir / "qwen" / "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            self.models_dir / "qwen2.5-coder-7b-instruct-q4_0.gguf",
            self.models_dir / "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            Path("E:/LocalMeshAI-Data/models/qwen_coder/qwen2.5-coder-7b-instruct-q4_0.gguf"),
            Path("E:/Game/Models/qwen2.5-coder-7b-instruct-q5_k_m.gguf"),
            Path("E:/Game/Models/Spark-X2.6-4B-Q8_0.gguf"),
            PROJECT_ROOT / "Spark-X2.6-4B-Q8_0.gguf"
        ]
        for cp in candidate_paths:
            if cp.exists() and cp.stat().st_size > 100_000_000:
                return {
                    "state": "fully_installed",
                    "model_path": str(cp),
                    "size_bytes": cp.stat().st_size,
                    "source": "existing_local_file"
                }

        # 3. Check Hugging Face hub cache
        resolved = self.verify_file_offline(
            self.QWEN_GGUF_FILE["repo_id"],
            self.QWEN_GGUF_FILE["filename"]
        )
        if resolved:
            return {
                "state": "fully_installed",
                "model_path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "source": "hf_cache"
            }

        return {
            "state": "missing",
            "model_path": None,
            "size_bytes": 0,
            "source": None
        }

    def download_triposr_weights(self, force: bool = False, progress_callback: Optional[callable] = None) -> bool:
        """
        Downloads TripoSR (config.yaml, model.ckpt) and DINO ViT (config.json)
        using huggingface_hub with resume support and disk space checking.
        """
        configure_hf_environment(self.models_dir)
        status = self.verify_triposr_status()
        if status["ready_for_offline_inference"] and not force:
            print("[DownloadManager] TripoSR and DINO weights are already fully installed and verified.")
            return True

        # Check required space (~2 GB)
        req_gb = 2.5
        free_gb = self.get_free_disk_space_gb()
        if free_gb < req_gb:
            raise RuntimeError(
                f"Insufficient disk space on {self.models_dir.drive or self.models_dir}. "
                f"Required: {req_gb} GB, Available: {free_gb} GB."
            )

        from huggingface_hub import hf_hub_download

        print("=" * 68)
        print("  PROVISIONING TRIPOSR & DINO MODEL WEIGHTS")
        print(f"  Target Cache Directory: {self.hf_cache}")
        print("=" * 68)

        manifest = self.load_manifest()
        manifest.setdefault("models", {})

        for item in self.TRIPOSR_FILES:
            repo = item["repo_id"]
            fn = item["filename"]
            print(f"-> Provisioning {repo}/{fn}...")
            if progress_callback:
                progress_callback(f"Downloading {repo}/{fn}...")

            t0 = time.time()
            try:
                local_path = hf_hub_download(
                    repo_id=repo,
                    filename=fn,
                    cache_dir=str(self.hf_cache),
                    resume_download=True,
                    local_files_only=False
                )
                p = Path(local_path)
                elapsed = round(time.time() - t0, 2)
                size_mb = round(p.stat().st_size / (1024 * 1024), 2)
                print(f"   [OK] Downloaded {fn} ({size_mb} MB in {elapsed}s)")

                manifest["models"][f"{repo}/{fn}"] = {
                    "repo_id": repo,
                    "filename": fn,
                    "local_path": str(p),
                    "size_bytes": p.stat().st_size,
                    "installed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                print(f"   [ERROR] Failed to download {repo}/{fn}: {e}")
                self.save_manifest(manifest)
                raise RuntimeError(f"Connection or download failure for {repo}/{fn}: {e}")

        # Post-download offline verification
        verify_status = self.verify_triposr_status()
        if not verify_status["ready_for_offline_inference"]:
            raise RuntimeError("Post-download offline verification failed. Some files could not be resolved offline.")

        manifest["triposr_provisioned"] = True
        manifest["triposr_last_verified"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_manifest(manifest)

        print("=" * 68)
        print("  [SUCCESS] TripoSR and DINO model weights verified and ready for offline inference!")
        print("=" * 68)
        return True

    def download_qwen_weights(self, force: bool = False, progress_callback: Optional[callable] = None) -> str:
        """
        Downloads Qwen 2.5 Coder 7B Instruct GGUF (Q4_K_M) with resume support.
        Returns the resolved local file path.
        """
        configure_hf_environment(self.models_dir)
        q_status = self.verify_qwen_status()
        if q_status["state"] == "fully_installed" and not force:
            print(f"[DownloadManager] Qwen model already installed at {q_status['model_path']}")
            return q_status["model_path"]

        # Check required space (~5 GB)
        req_gb = 5.5
        free_gb = self.get_free_disk_space_gb()
        if free_gb < req_gb:
            raise RuntimeError(
                f"Insufficient disk space on {self.models_dir.drive or self.models_dir}. "
                f"Required: {req_gb} GB, Available: {free_gb} GB."
            )

        from huggingface_hub import hf_hub_download

        repo = self.QWEN_GGUF_FILE["repo_id"]
        fn = self.QWEN_GGUF_FILE["filename"]

        print("=" * 68)
        print(f"  PROVISIONING LOCAL LLM: {repo}/{fn}")
        print(f"  Target Cache Directory: {self.hf_cache}")
        print("=" * 68)

        if progress_callback:
            progress_callback(f"Downloading {fn} (~4.7 GB)...")

        t0 = time.time()
        try:
            local_path = hf_hub_download(
                repo_id=repo,
                filename=fn,
                cache_dir=str(self.hf_cache),
                resume_download=True,
                local_files_only=False
            )
            p = Path(local_path)
            elapsed = round(time.time() - t0, 2)
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            print(f"   [OK] Downloaded {fn} ({size_mb} MB in {elapsed}s)")

            manifest = self.load_manifest()
            manifest.setdefault("models", {})
            manifest["models"][f"{repo}/{fn}"] = {
                "repo_id": repo,
                "filename": fn,
                "local_path": str(p),
                "size_bytes": p.stat().st_size,
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            manifest["qwen_provisioned"] = True
            manifest["qwen_model_path"] = str(p)
            self.save_manifest(manifest)

            # Update config.json with resolved path
            config_path = PROJECT_ROOT / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    cfg["qwen_model_path"] = str(p)
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
                except Exception:
                    pass

            return str(p)
        except Exception as e:
            raise RuntimeError(f"Failed to download Qwen model {repo}/{fn}: {e}")


download_manager = DownloadManager()
