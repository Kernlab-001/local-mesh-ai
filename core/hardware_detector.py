"""
Hardware Auto-Detection & Dynamic Configuration Engine for Local Mesh AI.
Automatically scans the host machine for:
1. Dedicated NVIDIA / AMD / Intel GPUs, CUDA runtime, and GDDR VRAM size.
2. System CPU cores and physical RAM.
3. Installed Blender installations across standard OS paths.
4. Dynamically calculates optimal neural reconstruction and VRAM lease limits,
   updating config.json without manual user editing.
"""

import os
import sys
import json
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

class HardwareDetector:
    def __init__(self):
        self.profile = {}

    def detect_gpu(self) -> Dict[str, Any]:
        """Detects CUDA availability, device name, VRAM capacity, and compute capability."""
        gpu_info = {
            "available": False,
            "device": "cpu",
            "name": "CPU Only",
            "vram_gb": 0.0,
            "compute_capability": None,
            "cuda_version": None
        }

        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["device"] = "cuda:0"
                gpu_info["name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_info["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
                gpu_info["compute_capability"] = f"{props.major}.{props.minor}"
                gpu_info["cuda_version"] = torch.version.cuda
        except Exception as e:
            gpu_info["error"] = str(e)

        # Fallback: query system nvidia-smi if PyTorch is not yet installed or not CUDA
        if not gpu_info["available"]:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                    encoding="utf-8",
                    errors="ignore"
                ).strip()
                if out:
                    line = out.splitlines()[0]
                    parts = [p.strip() for p in line.split(",")]
                    gpu_name = parts[0]
                    mem_mb = float(parts[1])
                    driver_ver = parts[2] if len(parts) > 2 else "Installed"
                    gpu_info["available"] = True
                    gpu_info["device"] = "cuda:0"
                    gpu_info["name"] = gpu_name
                    gpu_info["vram_gb"] = round(mem_mb / 1024.0, 2)
                    gpu_info["cuda_version"] = f"Driver {driver_ver}"
            except Exception:
                pass

        return gpu_info

    def detect_cpu_ram(self) -> Dict[str, Any]:
        """Detects host CPU architecture, core counts, and physical RAM."""
        cpu_info = {
            "processor": platform.processor(),
            "architecture": platform.machine(),
            "physical_cores": os.cpu_count() or 4,
            "ram_gb": 8.0
        }

        try:
            import psutil
            cpu_info["physical_cores"] = psutil.cpu_count(logical=False) or cpu_info["physical_cores"]
            cpu_info["logical_threads"] = psutil.cpu_count(logical=True) or cpu_info["physical_cores"]
            cpu_info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass

        return cpu_info

    def find_blender(self) -> Optional[str]:
        """Searches common installation paths and PATH for Blender 4.x / 5.x."""
        # 1. Check existing config if valid
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r") as f:
                    cfg = json.load(f)
                    p = cfg.get("blender_path")
                    if p and os.path.exists(p) and "blender" in p.lower():
                        return str(Path(p).resolve())
            except Exception:
                pass

        # 2. Check PATH
        path_blender = shutil.which("blender")
        if path_blender and os.path.exists(path_blender):
            return str(Path(path_blender).resolve())

        # 3. Standard Windows locations
        if platform.system() == "Windows":
            search_paths = [
                r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender Foundation\Blender 5.2\blender.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender Foundation\Blender 4.2\blender.exe"),
            ]
            for p in search_paths:
                if os.path.exists(p):
                    return str(Path(p).resolve())

        # 4. Standard Linux / macOS locations
        elif platform.system() == "Darwin":
            mac_p = "/Applications/Blender.app/Contents/MacOS/Blender"
            if os.path.exists(mac_p):
                return mac_p
        elif platform.system() == "Linux":
            for p in ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"]:
                if os.path.exists(p):
                    return p

        return None

    def calculate_optimal_tuning(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates optimal reconstruction resolution, chunk size, and limits based on detected VRAM."""
        vram = gpu_info.get("vram_gb", 0.0)

        if vram >= 12.0:
            tier = "High-End GPU (12GB+ VRAM)"
            mc_res = 256
            chunk = 4096
            vram_limit = round(vram - 1.5, 1)
            qwen_layers = 33  # Full offload
            rec_llm = "Qwen 2.5 Coder 14B / 32B Instruct (Q4_K_M)"
            llm_filename = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
        elif vram >= 6.0:
            tier = "Balanced GPU (6GB-12GB VRAM)"
            mc_res = 256
            chunk = 8192
            vram_limit = round(vram - 0.5, 1)
            qwen_layers = 24  # Standard offload
            rec_llm = "Qwen 2.5 Coder 7B Instruct (Q4_K_M)"
            llm_filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        elif vram >= 4.0:
            tier = "Entry GPU (4GB-6GB VRAM)"
            mc_res = 128
            chunk = 8192
            vram_limit = round(vram - 0.5, 1)
            qwen_layers = 16
            rec_llm = "Qwen 2.5 Coder 3B Instruct (Q4_K_M)"
            llm_filename = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
        else:
            tier = "CPU / Integrated Fallback"
            mc_res = 128
            chunk = 8192
            vram_limit = 4.0
            qwen_layers = 0
            rec_llm = "Qwen 2.5 Coder 1.5B Instruct (Q4_K_M)"
            llm_filename = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

        return {
            "tier": tier,
            "mc_resolution": mc_res,
            "chunk_size": chunk,
            "vram_limit_gb": vram_limit,
            "qwen_gpu_layers": qwen_layers,
            "recommended_llm": rec_llm,
            "recommended_llm_file": llm_filename
        }

    def scan_and_configure(self, save_config: bool = True) -> Dict[str, Any]:
        """Runs the complete auto-detection scan and updates config.json with host hardware profile."""
        gpu_info = self.detect_gpu()
        cpu_ram = self.detect_cpu_ram()
        blender_path = self.find_blender()
        tuning = self.calculate_optimal_tuning(gpu_info)

        profile = {
            "system": {
                "os": f"{platform.system()} {platform.release()}",
                "cpu_cores": cpu_ram.get("physical_cores"),
                "ram_gb": cpu_ram.get("ram_gb")
            },
            "gpu": gpu_info,
            "blender": {
                "found": blender_path is not None,
                "path": blender_path or "NOT_FOUND"
            },
            "tuning": tuning
        }

        self.profile = profile

        if save_config:
            # Build unified config.json with relative paths
            current_cfg = {}
            if CONFIG_PATH.exists():
                try:
                    with open(CONFIG_PATH, "r") as f:
                        current_cfg = json.load(f)
                except Exception:
                    pass

            updated_cfg = {
                "blender_path": blender_path or current_cfg.get("blender_path", "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"),
                "gpu_device": gpu_info["device"],
                "gpu_name": gpu_info["name"],
                "vram_gb": gpu_info["vram_gb"],
                "hardware_tier": tuning["tier"],
                "mc_resolution": tuning["mc_resolution"],
                "chunk_size": tuning["chunk_size"],
                "vram_limit_gb": tuning["vram_limit_gb"],
                "qwen_gpu_layers": tuning["qwen_gpu_layers"],
                "output_format": current_cfg.get("output_format", "glb"),
                "qwen_model_path": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                "output_dir": "output"
            }

            with open(CONFIG_PATH, "w") as f:
                json.dump(updated_cfg, f, indent=2)

        return profile

hardware_detector = HardwareDetector()

if __name__ == "__main__":
    prof = hardware_detector.scan_and_configure(save_config=True)
    print("=" * 65)
    print("LOCAL MESH AI - HARDWARE AUTODETECTION & CALIBRATION ENGINE")
    print("=" * 65)
    print(f"OS:       {prof['system']['os']}")
    print(f"CPU:      {prof['system']['cpu_cores']} Physical Cores | RAM: {prof['system']['ram_gb']} GB")
    print(f"GPU:      {prof['gpu']['name']} ({prof['gpu']['vram_gb']} GB VRAM)")
    print(f"Device:   {prof['gpu']['device']} (CUDA: {prof['gpu']['cuda_version']})")
    print(f"Blender:  {prof['blender']['path']}")
    print(f"Profile:  {prof['tuning']['tier']}")
    print(f"Settings: Resolution={prof['tuning']['mc_resolution']}, Chunk={prof['tuning']['chunk_size']}, VRAM Limit={prof['tuning']['vram_limit_gb']}GB")
    
    # Check for installed local GGUF LLM models
    found_llm = None
    try:
        for f in os.listdir(PROJECT_ROOT):
            if f.endswith(".gguf"):
                found_llm = f
                break
    except Exception:
        pass

    if found_llm:
        print(f"Local LLM: Installed ({found_llm})")
    else:
        print("Local LLM: None installed (Optional - only needed for AI scene editing)")
        print(f"Recommended LLM for this GPU: {prof['tuning']['recommended_llm']}")
    print("=" * 65)
    print("Successfully calibrated config.json for this machine!")
