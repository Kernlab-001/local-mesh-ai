"""
Hardware Preflight Check for LoRA Training in Local Mesh AI.
Enforces physical hardware safety boundaries for NVIDIA RTX 3050 Laptop GPU (6 GB VRAM).
Rejects training workloads that exceed VRAM capacity, preventing CUDA OOM crashes.
"""

from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

from core.gpu_broker import gpu_broker
import psutil


class PreflightReport(BaseModel):
    passed: bool
    can_train_on_gpu: bool
    can_train_on_cpu: bool
    target_model: str
    parameter_count_str: str
    available_vram_mb: float
    required_vram_mb: float
    available_ram_gb: float
    required_ram_gb: float
    summary: str
    rejection_reason: str = ""
    recommended_alternative: str = ""


class TrainingPreflight:
    # Approximate minimum VRAM requirements for 4-bit LoRA training
    MODEL_VRAM_REQUIREMENTS = {
        "qwen2.5-coder-7b": 9500.0,   # ~9.5 GB minimum with batch_size=1, gradient checkpointing
        "qwen2.5-coder-1.5b": 3800.0, # ~3.8 GB fits in 6 GB VRAM
        "qwen2.5-coder-0.5b": 2200.0, # ~2.2 GB fits comfortably
        "triposr-adapter": 4500.0,     # ~4.5 GB fits
    }

    @classmethod
    def evaluate(
        cls,
        model_name: str = "qwen2.5-coder-7b",
        force_cpu: bool = False
    ) -> PreflightReport:
        """
        Evaluates system hardware against the target model training requirements.
        Strictly enforces 6 GB VRAM limitation.
        """
        gpu_status = gpu_broker.get_status()
        available_vram = float(gpu_status.get("total_vram_mb", 6144.0))

        vm = psutil.virtual_memory()
        available_ram = round(vm.available / (1024**3), 2)

        model_key = model_name.lower().replace("_", "-")
        required_vram = cls.MODEL_VRAM_REQUIREMENTS.get(model_key, 9500.0)
        required_ram = 2.0  # At least 2.0 GB free RAM recommended for training data processing

        can_train_on_gpu = False
        rejection_reason = ""
        recommended_alt = ""

        if force_cpu:
            can_train_on_cpu = available_ram >= required_ram
            passed = can_train_on_cpu
            summary = "CPU training mode selected. GPU preflight bypassed."
        else:
            if available_vram < required_vram:
                can_train_on_gpu = False
                passed = False
                rejection_reason = (
                    f"Target model '{model_name}' requires minimum {required_vram:.0f} MB VRAM for LoRA training, "
                    f"but detected GPU (RTX 3050 Laptop) provides only {available_vram:.0f} MB VRAM."
                )
                recommended_alt = (
                    "Export prepared training dataset via CPU for remote/cloud fine-tuning, "
                    "or train smaller 1.5B/0.5B adapter models locally."
                )
                summary = f"Preflight REJECTED: Insufficient VRAM ({available_vram:.0f} MB available < {required_vram:.0f} MB required)."
            else:
                can_train_on_gpu = True
                passed = True
                summary = f"Preflight PASSED: GPU VRAM ({available_vram:.0f} MB) is sufficient for {model_name}."

            can_train_on_cpu = available_ram >= required_ram

        return PreflightReport(
            passed=passed,
            can_train_on_gpu=can_train_on_gpu,
            can_train_on_cpu=can_train_on_cpu,
            target_model=model_name,
            parameter_count_str="7 Billion" if "7b" in model_key else "Lightweight",
            available_vram_mb=available_vram,
            required_vram_mb=required_vram,
            available_ram_gb=available_ram,
            required_ram_gb=required_ram,
            summary=summary,
            rejection_reason=rejection_reason,
            recommended_alternative=recommended_alt,
        )


training_preflight = TrainingPreflight()
