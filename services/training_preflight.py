"""
Hardware Preflight Check for LoRA Training in Local Mesh AI.
Enforces physical hardware safety boundaries using dynamically measured FREE VRAM
plus safety margins, protecting the NVIDIA RTX 3050 Laptop GPU from CUDA OOM crashes.
Honestly rejects oversized training workloads while supporting dataset preparation and export.
"""

from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from core.gpu_broker import gpu_broker
import psutil


class PreflightReport(BaseModel):
    passed: bool
    can_train_on_gpu: bool
    can_train_on_cpu: bool
    target_model: str
    parameter_count_str: str
    available_vram_mb: Optional[float]
    free_vram_mb: Optional[float]
    required_vram_mb: float
    safety_margin_mb: float
    available_ram_gb: float
    required_ram_gb: float
    summary: str
    rejection_reason: str = ""
    recommended_alternative: str = ""
    hardware_feasibility: str = "unsupported"  # "supported", "unsupported", "cpu_only"


class TrainingPreflight:
    # Minimum required dedicated VRAM for LoRA training (MB)
    MODEL_VRAM_REQUIREMENTS = {
        "qwen2.5-coder-7b": 9500.0,     # ~9.5 GB minimum
        "qwen2.5-coder-1.5b": 3800.0,   # ~3.8 GB
        "qwen2.5-coder-0.5b": 2200.0,   # ~2.2 GB
        "triposr-adapter": 2500.0,       # ~2.5 GB
        "local-technical-artist-lora": 1200.0,  # ~1.2 GB lightweight adapter
    }
    VRAM_SAFETY_MARGIN_MB = 1024.0  # 1 GB mandatory safety margin

    @classmethod
    def evaluate(
        cls,
        model_name: str = "qwen2.5-coder-7b",
        force_cpu: bool = False
    ) -> PreflightReport:
        """
        Evaluates system hardware against target model requirements using real measured free VRAM.
        Never substitutes fallback values when physical detection is absent.
        """
        gpu_status = gpu_broker.get_status()
        available_vram = gpu_status.get("total_vram_mb")
        free_vram = gpu_status.get("free_vram_mb")
        cuda_available = gpu_status.get("cuda_available", False)

        vm = psutil.virtual_memory()
        available_ram = round(vm.available / (1024**3), 2)

        model_key = model_name.lower().replace("_", "-")
        required_vram = cls.MODEL_VRAM_REQUIREMENTS.get(model_key, 9500.0)
        required_ram = 2.0  # At least 2 GB RAM recommended for dataset preparation

        can_train_on_gpu = False
        can_train_on_cpu = available_ram >= required_ram
        rejection_reason = ""
        recommended_alt = ""
        feasibility = "unsupported"

        if force_cpu:
            passed = can_train_on_cpu
            feasibility = "cpu_only" if passed else "unsupported"
            summary = "CPU dataset preparation and training mode selected. GPU bypassed."
        else:
            if not cuda_available or free_vram is None:
                passed = False
                rejection_reason = "No CUDA-capable GPU detected or physical VRAM query failed."
                recommended_alt = "Prepare dataset locally and export for cloud training, or use CPU mode."
                summary = "Preflight REJECTED: CUDA acceleration unavailable."
                feasibility = "unsupported"
            elif free_vram < (required_vram + cls.VRAM_SAFETY_MARGIN_MB):
                passed = False
                needed_total = required_vram + cls.VRAM_SAFETY_MARGIN_MB
                rejection_reason = (
                    f"Target model '{model_name}' requires minimum {required_vram:.0f} MB VRAM plus "
                    f"{cls.VRAM_SAFETY_MARGIN_MB:.0f} MB safety margin ({needed_total:.0f} MB total), "
                    f"but detected free VRAM is only {free_vram:.0f} MB (Total physical: {available_vram:.0f} MB)."
                )
                recommended_alt = (
                    "Local 7B training is unsupported on this 6 GB hardware configuration. "
                    "Export the prepared dataset via CPU for remote fine-tuning, "
                    "or train a lightweight adapter (e.g., local-technical-artist-lora)."
                )
                summary = f"Preflight REJECTED: Insufficient free VRAM ({free_vram:.0f} MB free < {needed_total:.0f} MB required)."
                feasibility = "unsupported"
            else:
                passed = True
                can_train_on_gpu = True
                summary = f"Preflight PASSED: Sufficient free VRAM ({free_vram:.0f} MB free >= {required_vram + cls.VRAM_SAFETY_MARGIN_MB:.0f} MB)."
                feasibility = "supported"

        param_str = "7 Billion" if "7b" in model_key else ("1.5 Billion" if "1.5b" in model_key else "Lightweight Adapter")

        return PreflightReport(
            passed=passed,
            can_train_on_gpu=can_train_on_gpu,
            can_train_on_cpu=can_train_on_cpu,
            target_model=model_name,
            parameter_count_str=param_str,
            available_vram_mb=available_vram,
            free_vram_mb=free_vram,
            required_vram_mb=required_vram,
            safety_margin_mb=cls.VRAM_SAFETY_MARGIN_MB,
            available_ram_gb=available_ram,
            required_ram_gb=required_ram,
            summary=summary,
            rejection_reason=rejection_reason,
            recommended_alternative=recommended_alt,
            hardware_feasibility=feasibility,
        )


training_preflight = TrainingPreflight()
