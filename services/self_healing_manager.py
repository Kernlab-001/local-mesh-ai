"""
Autonomous Self-Healing Manager for Local Mesh AI.
Orchestrates diagnosis, impact previews, and authorized safe execution
of remediation actions across GPU, memory, storage, Blender, and sessions.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

from core.gpu_broker import gpu_broker
from core.path_security import get_staging_dir, get_data_dir, PROJECT_ROOT
from core.recovery_actions import (
    ActionType,
    RecoveryActionRequest,
    RecoveryActionResult,
    SystemHealthSummary,
)
from services.health_monitor import health_monitor
from services.recovery_policy_engine import recovery_policy_engine


class SelfHealingManager:
    def __init__(self):
        self.monitor = health_monitor
        self.policy = recovery_policy_engine

    def diagnose(self) -> SystemHealthSummary:
        """Runs complete diagnostics and returns issues and recommendations."""
        return self.monitor.get_summary()

    def preview_action(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        """Evaluates an action in dry-run mode without modifying system state."""
        dry_req = req.model_copy(update={"dry_run": True})
        return self._dispatch_action(dry_req)

    def execute_action(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        """Validates policy and executes the requested remediation action."""
        # 1. Policy check
        allowed, reason = self.policy.validate_action_policy(req)
        if not allowed:
            res = RecoveryActionResult(
                action_type=req.action_type,
                status="requires_approval" if "approval" in reason.lower() else "rejected",
                message=reason,
                changes_made=[],
            )
            self.policy.record_execution(res)
            return res

        # 2. Dispatch
        result = self._dispatch_action(req)
        self.policy.record_execution(result)
        return result

    def _dispatch_action(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        if req.action_type == ActionType.VERIFY_BLENDER:
            return self._action_verify_blender(req)
        elif req.action_type == ActionType.PURGE_STALE_GPU_LEASE:
            return self._action_purge_gpu_lease(req)
        elif req.action_type == ActionType.CLEAN_STAGING_DIR:
            return self._action_clean_staging_dir(req)
        elif req.action_type == ActionType.CLEAN_TEMP_MESHES:
            return self._action_clean_temp_meshes(req)
        elif req.action_type == ActionType.RESET_FAILED_JOBS:
            return self._action_reset_failed_jobs(req)
        elif req.action_type == ActionType.RECOVER_ORPHANED_SESSIONS:
            return self._action_recover_orphaned_sessions(req)
        elif req.action_type == ActionType.VERIFY_MODEL_WEIGHTS:
            return self._action_verify_model_weights(req)
        else:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="failed",
                message=f"Unknown action type '{req.action_type}'.",
            )

    def _action_verify_blender(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        info = self.monitor.inspect_blender()
        if info["installed"]:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run" if req.dry_run else "success",
                message=f"Blender verified at '{info['path']}' ({info['version']}).",
                changes_made=[f"Found Blender: {info['path']}"],
            )
        else:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="failed",
                message="Blender executable could not be verified in standard paths.",
                error="Blender not installed or not in PATH.",
            )

    def _action_purge_gpu_lease(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        status = gpu_broker.get_status()
        if not status.get("lease_active"):
            return RecoveryActionResult(
                action_type=req.action_type,
                status="skipped",
                message="No active GPU lease to purge.",
            )

        owner = status.get("current_owner", "unknown")
        age = int(status.get("lease_age_seconds", 0))

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Would force release GPU lease held by '{owner}' (age {age}s) and purge VRAM.",
                changes_made=[f"Proposed release of lease held by {owner}"],
            )

        gpu_broker._force_release()
        gpu_broker.purge_vram()
        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Purged GPU lease held by '{owner}' and flushed VRAM cache.",
            changes_made=[f"Released lease held by {owner}", "Flushed PyTorch CUDA VRAM cache"],
        )

    def _action_clean_staging_dir(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        staging = get_staging_dir()
        if not staging.exists():
            return RecoveryActionResult(
                action_type=req.action_type,
                status="skipped",
                message="Staging directory does not exist.",
            )

        files = [p for p in staging.iterdir() if p.is_file()]
        total_size = sum(p.stat().st_size for p in files)
        size_mb = round(total_size / (1024**2), 2)

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Would delete {len(files)} files ({size_mb} MB) in staging directory.",
                changes_made=[f"Candidate cleanup: {len(files)} files, {size_mb} MB"],
            )

        deleted_count = 0
        for p in files:
            try:
                p.unlink()
                deleted_count += 1
            except OSError:
                pass

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Cleaned staging directory: removed {deleted_count} files ({size_mb} MB).",
            changes_made=[f"Removed {deleted_count} files ({size_mb} MB) from {staging}"],
        )

    def _action_clean_temp_meshes(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        temp_dir = Path(tempfile.gettempdir())
        patterns = ["staged_*.glb", "staged_*.obj", "hp_bake_*.obj", "lp_bake_*.obj", "temp_*.glb"]
        matched_files = []
        for pat in patterns:
            matched_files.extend(list(temp_dir.glob(pat)))

        total_size = sum(p.stat().st_size for p in matched_files if p.is_file())
        size_mb = round(total_size / (1024**2), 2)

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Would remove {len(matched_files)} temporary mesh files ({size_mb} MB).",
                changes_made=[f"Identified {len(matched_files)} temporary mesh files"],
            )

        deleted = 0
        for p in matched_files:
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Removed {deleted} temporary mesh files ({size_mb} MB).",
            changes_made=[f"Deleted {deleted} temp files from {temp_dir}"],
        )

    def _action_reset_failed_jobs(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        from services.job_manager import job_manager
        failed_count = sum(1 for j in job_manager.jobs.values() if j.get("status") == "failed")

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Would reset/clear {failed_count} failed jobs from memory.",
                changes_made=[f"Found {failed_count} failed jobs"],
            )

        # Clear failed jobs
        keys_to_remove = [k for k, v in job_manager.jobs.items() if v.get("status") == "failed"]
        for k in keys_to_remove:
            job_manager.jobs.pop(k, None)

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Cleared {len(keys_to_remove)} failed jobs from job queue.",
            changes_made=[f"Removed job IDs: {keys_to_remove}"],
        )

    def _action_recover_orphaned_sessions(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        from services.editing_session_manager import editing_session_manager
        orphaned = []
        with editing_session_manager._global_lock:
            for s_id, sess in editing_session_manager._sessions.items():
                if not os.path.exists(sess.working_mesh_path):
                    orphaned.append(s_id)

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Identified {len(orphaned)} orphaned sessions with missing working mesh files.",
                changes_made=[f"Orphaned session IDs: {orphaned}"],
            )

        with editing_session_manager._global_lock:
            for s_id in orphaned:
                editing_session_manager._sessions.pop(s_id, None)

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Removed {len(orphaned)} orphaned sessions with missing working files.",
            changes_made=[f"Cleaned sessions: {orphaned}"],
        )

    def _action_verify_model_weights(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        info = self.monitor.inspect_model_weights()
        if info["qwen_present"]:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="success",
                message=f"Qwen model verified at '{info['qwen_path']}' ({info['qwen_size_mb']} MB).",
                changes_made=[],
            )
        else:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="failed",
                message=f"Qwen GGUF model not found at expected location '{info['qwen_path']}'.",
                error="Model checkpoint missing.",
            )


self_healing_manager = SelfHealingManager()
