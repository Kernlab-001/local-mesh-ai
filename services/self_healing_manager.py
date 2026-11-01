"""
Autonomous Self-Healing Manager for Local Mesh AI.
Orchestrates diagnosis, impact previews, and authorized safe execution
of remediation actions across GPU, memory, storage, Blender, and sessions.
Protects active session assets and verifies state post-remediation.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Set

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
        """Validates policy, executes remediation, and verifies post-remediation state."""
        # 1. Policy check
        allowed, reason = self.policy.validate_action_policy(req)
        if not allowed:
            res = RecoveryActionResult(
                action_type=req.action_type,
                status="requires_approval" if "approval" in reason.lower() else "rejected",
                message=reason,
                changes_made=[],
                affected_files=[],
                verification_status="not_applicable",
            )
            self.policy.record_execution(res)
            return res

        # 2. Dispatch action
        result = self._dispatch_action(req)

        # 3. Post-remediation verification
        if result.status == "success" and not req.dry_run:
            verified, post_summary = self._verify_remediation(req.action_type)
            result.verification_status = "verified" if verified else "unresolved"
            result.post_remediation_summary = post_summary

        self.policy.record_execution(result)
        return result

    def _verify_remediation(self, action_type: ActionType) -> (bool, str):
        """Verifies that the problem was genuinely resolved after execution."""
        summary = self.monitor.get_summary()
        remaining_issues = [i for i in summary.active_issues if i.recommended_action == action_type]
        if not remaining_issues:
            return True, f"Verified: Issue for '{action_type}' resolved successfully."
        else:
            return False, f"Unresolved: {len(remaining_issues)} related issues still detected."

    def _get_protected_file_paths(self) -> Set[str]:
        """Collects all file paths currently held by active sessions and jobs to prevent accidental deletion."""
        protected = set()
        try:
            from services.editing_session_manager import editing_session_manager
            with editing_session_manager._global_lock:
                for sess in editing_session_manager._sessions.values():
                    if sess.working_mesh_path:
                        protected.add(os.path.abspath(sess.working_mesh_path))
                    if sess.original_mesh_path:
                        protected.add(os.path.abspath(sess.original_mesh_path))
                    for snap in getattr(sess, "history", []):
                        if getattr(snap, "mesh_path", None):
                            protected.add(os.path.abspath(snap.mesh_path))
        except Exception:
            pass

        try:
            from services.job_manager import job_manager
            for job in job_manager.jobs.values():
                for k in ["mesh_path", "output_path", "source_path"]:
                    p = job.get(k)
                    if p and isinstance(p, str):
                        protected.add(os.path.abspath(p))
        except Exception:
            pass

        return protected

    def _dispatch_action(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        if req.action_type == ActionType.VERIFY_BLENDER:
            return self._action_verify_blender(req)
        elif req.action_type == ActionType.PURGE_STALE_GPU_LEASE:
            return self._action_purge_gpu_lease(req)
        elif req.action_type == ActionType.CLEAN_STAGING_DIR:
            return self._action_clean_staging_dir(req)
        elif req.action_type == ActionType.CLEAN_TEMP_MESHES:
            return self._action_clean_staging_dir(req)  # Unified safe staging cleanup
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
                verification_status="verified",
            )
        else:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="failed",
                message="Blender executable could not be verified in standard paths.",
                error="Blender not installed or not in PATH.",
                verification_status="unresolved",
            )

    def _action_purge_gpu_lease(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        status = gpu_broker.get_status()
        if not status.get("lease_active"):
            return RecoveryActionResult(
                action_type=req.action_type,
                status="skipped",
                message="No active GPU lease to purge.",
                verification_status="verified",
            )

        owner = status.get("current_owner", "unknown")
        age = int(status.get("lease_age_seconds", 0))

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Would force release GPU lease held by '{owner}' (age {age}s) and purge VRAM.",
                changes_made=[f"Proposed release of lease held by {owner}"],
                affected_files=[gpu_broker.lock_file] if os.path.exists(gpu_broker.lock_file) else [],
            )

        gpu_broker._force_release()
        gpu_broker.purge_vram()
        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Purged GPU lease held by '{owner}' and flushed VRAM cache.",
            changes_made=[f"Released lease held by {owner}", "Flushed PyTorch CUDA VRAM cache"],
            affected_files=[gpu_broker.lock_file],
        )

    def _action_clean_staging_dir(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        staging = get_staging_dir()
        if not staging.exists():
            return RecoveryActionResult(
                action_type=req.action_type,
                status="skipped",
                message="Staging directory does not exist.",
            )

        protected_paths = self._get_protected_file_paths()
        min_age_seconds = req.parameters.get("min_age_seconds", 3600.0)  # 1 hour minimum age
        now = time.time()

        candidate_files: List[Path] = []
        for p in staging.rglob("*"):
            if p.is_file():
                abs_p = str(p.resolve())
                # Protected checks: active session or active job
                if abs_p in protected_paths:
                    continue
                # Age check
                try:
                    mtime = p.stat().st_mtime
                    if now - mtime >= min_age_seconds:
                        candidate_files.append(p)
                except OSError:
                    pass

        total_size = sum(p.stat().st_size for p in candidate_files)
        size_mb = round(total_size / (1024**2), 2)
        affected_list = [str(p) for p in candidate_files]

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Identified {len(candidate_files)} unreferenced staging files ({size_mb} MB) older than {int(min_age_seconds)}s for cleanup.",
                changes_made=[f"Targeted {len(candidate_files)} files ({size_mb} MB) in staging"],
                affected_files=affected_list,
            )

        deleted_count = 0
        for p in candidate_files:
            try:
                p.unlink()
                deleted_count += 1
            except OSError:
                pass

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Cleaned staging directory: removed {deleted_count} unreferenced files ({size_mb} MB).",
            changes_made=[f"Removed {deleted_count} files ({size_mb} MB) from {staging}"],
            affected_files=affected_list,
        )

    def _action_reset_failed_jobs(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        from services.job_manager import job_manager
        failed_keys = [k for k, v in job_manager.jobs.items() if v.get("status") == "failed"]

        if req.dry_run:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Would reset/clear {len(failed_keys)} failed jobs from memory queue.",
                changes_made=[f"Candidate failed jobs: {failed_keys}"],
            )

        for k in failed_keys:
            job_manager.jobs.pop(k, None)

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Cleared {len(failed_keys)} failed jobs from memory queue.",
            changes_made=[f"Removed job IDs: {failed_keys}"],
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
                verification_status="verified",
            )
        else:
            return RecoveryActionResult(
                action_type=req.action_type,
                status="failed",
                message=f"Qwen GGUF model not found at expected location '{info['qwen_path']}'.",
                error="Model checkpoint missing.",
                verification_status="unresolved",
            )


self_healing_manager = SelfHealingManager()
