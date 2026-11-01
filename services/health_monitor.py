"""
Real-time System and Hardware Health Monitor for Local Mesh AI.
Collects structured diagnostic telemetry across GPU, VRAM, Blender, host resources,
model weights, job queues, and editing sessions.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil

from core.gpu_broker import gpu_broker
from core.hardware_detector import hardware_detector
from core.path_security import get_data_dir, get_staging_dir, PROJECT_ROOT
from core.recovery_actions import (
    SeverityLevel,
    ActionType,
    DiagnosticIssue,
    SystemHealthSummary,
)


class HealthMonitor:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.start_time = time.time()

    def get_system_uptime(self) -> float:
        return time.time() - self.start_time

    def inspect_host_resources(self) -> Dict[str, Any]:
        """Inspects CPU, RAM, and disk storage."""
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.project_root))
        return {
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_percent_used": vm.percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(logical=True),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_percent_used": disk.percent,
        }

    def inspect_gpu_status(self) -> Dict[str, Any]:
        """Queries GPU Broker and Torch CUDA status."""
        status = gpu_broker.get_status()
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_allocated_mb = round(torch.cuda.memory_allocated() / (1024**2), 2)
                vram_reserved_mb = round(torch.cuda.memory_reserved() / (1024**2), 2)
        except Exception:
            pass

        return {
            "lease_active": status.get("lease_active", False),
            "current_owner": status.get("current_owner"),
            "lease_age_seconds": status.get("lease_age_seconds", 0.0),
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "gpu_name": status.get("device_name", "NVIDIA GeForce RTX 3050 Laptop GPU"),
            "total_vram_mb": status.get("total_vram_mb", 6144),
        }

    def inspect_blender(self) -> Dict[str, Any]:
        """Inspects local Blender installation and executable status."""
        blender_path = hardware_detector.find_blender() or "blender"
        installed = False
        version_str = "Not installed"

        if blender_path and (os.path.exists(blender_path) or shutil.which(blender_path)):
            installed = True
            try:
                res = subprocess.run([blender_path, "--version"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    first_line = res.stdout.splitlines()[0] if res.stdout else "Blender"
                    version_str = first_line.strip()
            except Exception as e:
                version_str = f"Installed (query error: {e})"

        return {
            "installed": installed,
            "path": blender_path if installed else None,
            "version": version_str,
        }

    def inspect_staging_storage(self) -> Dict[str, Any]:
        """Checks staging directory file counts and size."""
        staging = get_staging_dir()
        file_count = 0
        total_size_bytes = 0
        if staging.exists():
            for p in staging.rglob("*"):
                if p.is_file():
                    file_count += 1
                    try:
                        total_size_bytes += p.stat().st_size
                    except OSError:
                        pass

        size_mb = round(total_size_bytes / (1024**2), 2)
        return {
            "staging_dir": str(staging),
            "file_count": file_count,
            "size_mb": size_mb,
        }

    def inspect_model_weights(self) -> Dict[str, Any]:
        """Checks for essential model checkpoints."""
        qwen_path = self.project_root / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        return {
            "qwen_present": qwen_path.exists(),
            "qwen_path": str(qwen_path),
            "qwen_size_mb": round(qwen_path.stat().st_size / (1024**2), 2) if qwen_path.exists() else 0.0,
        }

    def diagnose_system(self) -> List[DiagnosticIssue]:
        """Analyzes all subsystems and generates typed diagnostic issues."""
        issues: List[DiagnosticIssue] = []

        # 1. Blender Check
        blender_info = self.inspect_blender()
        if not blender_info["installed"]:
            issues.append(
                DiagnosticIssue(
                    severity=SeverityLevel.ERROR,
                    subsystem="blender",
                    title="Blender Not Detected",
                    description="Blender 4.x / 5.x was not located in default paths or PATH. 3D operations will be restricted.",
                    recommended_action=ActionType.VERIFY_BLENDER,
                    requires_approval=False,
                )
            )

        # 2. Stale GPU Lease Check
        gpu_info = self.inspect_gpu_status()
        if gpu_info["lease_active"] and gpu_info["lease_age_seconds"] > 300.0:
            issues.append(
                DiagnosticIssue(
                    severity=SeverityLevel.WARNING,
                    subsystem="gpu",
                    title="Stale GPU Lease Detected",
                    description=f"GPU lease held by '{gpu_info['current_owner']}' for {int(gpu_info['lease_age_seconds'])}s exceeds 300s timeout.",
                    recommended_action=ActionType.PURGE_STALE_GPU_LEASE,
                    requires_approval=True,
                    parameters={"owner": gpu_info["current_owner"]},
                )
            )

        # 3. Staging Storage Bloat Check
        staging_info = self.inspect_staging_storage()
        if staging_info["size_mb"] > 1024.0 or staging_info["file_count"] > 100:
            issues.append(
                DiagnosticIssue(
                    severity=SeverityLevel.WARNING,
                    subsystem="storage",
                    title="Staging Storage Accumulation",
                    description=f"Staging directory holds {staging_info['file_count']} files ({staging_info['size_mb']} MB). Cleanup recommended.",
                    recommended_action=ActionType.CLEAN_STAGING_DIR,
                    requires_approval=True,
                    parameters={"file_count": staging_info["file_count"], "size_mb": staging_info["size_mb"]},
                )
            )

        # 4. Host Resources Check
        res_info = self.inspect_host_resources()
        if res_info["ram_percent_used"] > 90.0:
            issues.append(
                DiagnosticIssue(
                    severity=SeverityLevel.CRITICAL,
                    subsystem="memory",
                    title="High RAM Utilization",
                    description=f"System RAM utilization is at {res_info['ram_percent_used']}%. Low memory may cause worker crashes.",
                    recommended_action=ActionType.RESET_FAILED_JOBS,
                    requires_approval=True,
                )
            )

        if res_info["disk_free_gb"] < 2.0:
            issues.append(
                DiagnosticIssue(
                    severity=SeverityLevel.CRITICAL,
                    subsystem="storage",
                    title="Low Free Disk Space",
                    description=f"Available disk space is only {res_info['disk_free_gb']} GB on the project drive.",
                    recommended_action=ActionType.CLEAN_TEMP_MESHES,
                    requires_approval=True,
                )
            )

        return issues

    def get_summary(self) -> SystemHealthSummary:
        """Returns comprehensive aggregated system health summary."""
        issues = self.diagnose_system()
        recommendations = list(dict.fromkeys(i.recommended_action for i in issues))

        has_critical = any(i.severity == SeverityLevel.CRITICAL for i in issues)
        has_error = any(i.severity == SeverityLevel.ERROR for i in issues)
        has_warning = any(i.severity == SeverityLevel.WARNING for i in issues)

        if has_critical or has_error:
            overall = "degraded"
        elif has_warning:
            overall = "degraded"
        else:
            overall = "healthy"

        subsystems = {
            "uptime_seconds": round(self.get_system_uptime(), 1),
            "host": self.inspect_host_resources(),
            "gpu": self.inspect_gpu_status(),
            "blender": self.inspect_blender(),
            "staging": self.inspect_staging_storage(),
            "models": self.inspect_model_weights(),
        }

        return SystemHealthSummary(
            overall_status=overall,
            subsystems=subsystems,
            active_issues=issues,
            recovery_recommendations=recommendations,
        )


health_monitor = HealthMonitor()
