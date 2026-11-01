"""
Autonomous Self-Healing Manager for Local Mesh AI.
Orchestrates diagnosis, impact previews, and authorized safe execution
of remediation actions across GPU, memory, storage, Blender, and sessions.
Protects active session assets and verifies state post-remediation.
Binds execution cryptographically and temporally to server-side preview IDs.
"""

import os
import json
import time
import uuid
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Set, Optional

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


@dataclass
class CleanupPreviewRecord:
    preview_id: str
    action_type: ActionType
    canonical_paths: List[str]
    file_hashes: Dict[str, str]
    created_at: float
    expires_at: float
    enforced_min_age_seconds: float
    total_size_mb: float


class SelfHealingManager:
    def __init__(self):
        self.monitor = health_monitor
        self.policy = recovery_policy_engine
        self._active_previews: Dict[str, CleanupPreviewRecord] = {}
        self._preview_lock = threading.Lock()
        self.audit_log_file = get_data_dir() / "recovery_audit.json"

    def diagnose(self) -> SystemHealthSummary:
        """Runs complete diagnostics and returns issues and recommendations."""
        return self.monitor.get_summary()

    def preview_action(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        """Evaluates an action in dry-run mode without modifying system state, generating a bound preview_id."""
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

    def _record_audit(self, entry: Dict[str, Any]):
        """Persists a durable audit record of recovery execution."""
        try:
            records = []
            if self.audit_log_file.exists():
                try:
                    with open(self.audit_log_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                except Exception:
                    records = []
            records.append(entry)
            self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

    def _compute_file_sha256(self, file_path: Path) -> str:
        """Computes streaming SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def _dispatch_action(self, req: RecoveryActionRequest) -> RecoveryActionResult:
        if req.action_type == ActionType.VERIFY_BLENDER:
            return self._action_verify_blender(req)
        elif req.action_type == ActionType.PURGE_STALE_GPU_LEASE:
            return self._action_purge_gpu_lease(req)
        elif req.action_type in (ActionType.CLEAN_STAGING_DIR, ActionType.CLEAN_TEMP_MESHES):
            return self._action_clean_staging_dir(req)
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
        # Enforce 1-hour minimum cleanup age on backend (cannot be bypassed by min_age_seconds=0)
        raw_min_age = req.parameters.get("min_age_seconds", 3600.0)
        try:
            min_age_seconds = max(float(raw_min_age), 3600.0)
        except (ValueError, TypeError):
            min_age_seconds = 3600.0
        now = time.time()
        staging_resolved = staging.resolve()

        # --- A. Dry-Run Preview Mode: Generates cryptographically bound, single-use preview_id ---
        if req.dry_run:
            candidate_files: List[Path] = []
            file_hashes: Dict[str, str] = {}
            for p in staging.rglob("*"):
                try:
                    if not p.is_file() or p.is_symlink():
                        continue
                    resolved_p = p.resolve()
                    # Strict canonical path containment check
                    if not resolved_p.is_relative_to(staging_resolved):
                        continue
                    if str(resolved_p) in protected_paths:
                        continue
                    mtime = p.stat().st_mtime
                    if now - mtime >= min_age_seconds:
                        candidate_files.append(p)
                        file_hashes[str(resolved_p)] = self._compute_file_sha256(p)
                except OSError:
                    pass

            total_size = sum(p.stat().st_size for p in candidate_files)
            size_mb = round(total_size / (1024**2), 2)
            affected_list = [str(p.resolve()) for p in candidate_files]

            preview_id = f"prev_{uuid.uuid4().hex[:12]}"
            expires_at = now + 900.0  # 15 minutes validity

            preview_record = CleanupPreviewRecord(
                preview_id=preview_id,
                action_type=req.action_type,
                canonical_paths=affected_list,
                file_hashes=file_hashes,
                created_at=now,
                expires_at=expires_at,
                enforced_min_age_seconds=min_age_seconds,
                total_size_mb=size_mb
            )

            with self._preview_lock:
                self._active_previews[preview_id] = preview_record

            return RecoveryActionResult(
                action_type=req.action_type,
                status="dry_run",
                message=f"[Dry-Run] Identified {len(candidate_files)} unreferenced staging files ({size_mb} MB) older than {int(min_age_seconds)}s for cleanup.",
                changes_made=[f"Targeted {len(candidate_files)} files ({size_mb} MB) in staging with preview_id '{preview_id}'"],
                affected_files=affected_list,
                preview_id=preview_id,
                expires_at=expires_at,
            )

        # --- B. Approved Execution Mode: MANDATORY Two-Stage Execution ---
        # Destructive staging cleanup strictly requires a valid server-generated preview ID.
        # Fallback to unreviewed deletion is completely eliminated.
        preview_id = req.preview_id or req.parameters.get("preview_id")
        if not preview_id:
            raise ValueError(
                "Execution rejected: Destructive cleanup requires a valid, server-generated preview_id from a prior preview call."
            )

        # Atomic single-use token consumption
        with self._preview_lock:
            preview = self._active_previews.pop(preview_id, None)

        if not preview:
            raise ValueError(
                f"Execution rejected: Preview ID '{preview_id}' is invalid, has expired, or was already consumed."
            )

        if now > preview.expires_at:
            raise ValueError(
                f"Execution rejected: Preview ID '{preview_id}' expired at {time.ctime(preview.expires_at)}."
            )

        # Action-type binding verification
        if preview.action_type != req.action_type:
            raise ValueError(
                f"Execution rejected: Preview ID '{preview_id}' was generated for action '{preview.action_type}', cannot execute for '{req.action_type}'."
            )

        # Re-check live protections immediately before deletion
        current_protected = self._get_protected_file_paths()
        target_files_to_delete: List[Path] = []

        # Optional parameter: filter down to a subset of previewed candidate files if user selected specific ones
        approved_filter = req.parameters.get("approved_files")
        approved_set = {str(Path(f).resolve()) for f in approved_filter} if approved_filter is not None else None

        for path_str in preview.canonical_paths:
            p = Path(path_str)
            try:
                resolved_p = p.resolve()
                # 1. Strict canonical path containment (protect against path traversal & symlink hopping)
                if not resolved_p.is_relative_to(staging_resolved):
                    continue
                # 2. Strict symlink rejection
                if p.is_symlink() or resolved_p.is_symlink():
                    continue
                # 3. Re-verify active session and job protections
                if str(resolved_p) in current_protected:
                    continue
                # 4. Re-verify file existence and type
                if not p.exists() or not p.is_file():
                    continue
                # 5. Re-verify enforced minimum age
                if (now - p.stat().st_mtime) < preview.enforced_min_age_seconds:
                    continue
                # 6. Re-verify file SHA-256 hash has not changed
                curr_hash = self._compute_file_sha256(p)
                if curr_hash != preview.file_hashes.get(str(resolved_p)):
                    continue  # File was modified or replaced after preview
                # 7. Check user-approved subset filter if provided
                if approved_set is not None and str(resolved_p) not in approved_set:
                    continue

                target_files_to_delete.append(p)
            except OSError:
                pass

        deleted_count = 0
        deleted_paths = []
        total_deleted_size = 0

        for p in target_files_to_delete:
            try:
                sz = p.stat().st_size
                p.unlink()
                deleted_count += 1
                deleted_paths.append(str(p))
                total_deleted_size += sz
            except OSError:
                pass

        size_mb = round(total_deleted_size / (1024**2), 2)

        # Durable audit logging
        audit_entry = {
            "timestamp": now,
            "action_type": req.action_type.value,
            "preview_id": preview_id,
            "deleted_count": deleted_count,
            "size_mb": size_mb,
            "deleted_files": deleted_paths,
            "approved": True
        }
        self._record_audit(audit_entry)

        return RecoveryActionResult(
            action_type=req.action_type,
            status="success",
            message=f"Cleaned staging directory: removed {deleted_count} unreferenced reviewed files ({size_mb} MB).",
            changes_made=[f"Removed {deleted_count} files ({size_mb} MB) from {staging}"],
            affected_files=deleted_paths,
            preview_id=preview_id,
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
