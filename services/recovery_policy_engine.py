"""
Recovery Policy Engine for Local Mesh AI.
Enforces execution safety, circuit breakers (counting successes and failures),
approval requirements, active GPU protection, persistent audit trails,
and dry-run impact evaluation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from core.gpu_broker import gpu_broker
from core.path_security import get_data_dir
from core.recovery_actions import (
    ActionType,
    RecoveryActionRequest,
    RecoveryActionResult,
)

DESTRUCTIVE_ACTIONS = {
    ActionType.PURGE_STALE_GPU_LEASE,
    ActionType.CLEAN_STAGING_DIR,
    ActionType.CLEAN_TEMP_MESHES,
    ActionType.RESET_FAILED_JOBS,
    ActionType.RECOVER_ORPHANED_SESSIONS,
}

MAX_ATTEMPTS_PER_WINDOW = 3
CIRCUIT_BREAKER_WINDOW_SECONDS = 300.0  # 5 minutes


class RecoveryPolicyEngine:
    def __init__(self):
        self._action_history: Dict[ActionType, List[float]] = defaultdict(list)
        self.audit_file = get_data_dir() / "health" / "recovery_audit.json"
        self._audit_log: List[RecoveryActionResult] = self._load_audit_log()

    def _load_audit_log(self) -> List[RecoveryActionResult]:
        if self.audit_file.exists():
            try:
                with open(self.audit_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [RecoveryActionResult(**item) for item in data]
            except Exception:
                pass
        return []

    def _save_audit_log(self):
        try:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_file, "w", encoding="utf-8") as f:
                json.dump([item.model_dump() for item in self._audit_log[-200:]], f, indent=2)
        except Exception:
            pass

    def validate_action_policy(self, req: RecoveryActionRequest) -> Tuple[bool, str]:
        """
        Validates whether the requested recovery action passes safety policies.
        Returns (is_allowed, reason).
        """
        now = time.time()
        # Clean history outside window
        self._action_history[req.action_type] = [
            t for t in self._action_history[req.action_type]
            if now - t < CIRCUIT_BREAKER_WINDOW_SECONDS
        ]

        # 1. Circuit Breaker Check (counts recent attempts)
        attempts = len(self._action_history[req.action_type])
        if attempts >= MAX_ATTEMPTS_PER_WINDOW:
            return False, (
                f"Circuit breaker tripped for '{req.action_type}': "
                f"exceeded {MAX_ATTEMPTS_PER_WINDOW} attempts in {int(CIRCUIT_BREAKER_WINDOW_SECONDS)}s."
            )

        # 2. Approval Requirement Check for Destructive Actions
        if not req.dry_run and req.action_type in DESTRUCTIVE_ACTIONS:
            if not req.approved:
                return False, (
                    f"Action '{req.action_type}' modifies system resources and requires informed "
                    f"user authorization (approved=True)."
                )

        # 3. Active GPU Protection
        if req.action_type == ActionType.PURGE_STALE_GPU_LEASE and not req.dry_run:
            status = gpu_broker.get_status()
            if status.get("lease_active"):
                holder = status.get("holder") or {}
                holder_pid = holder.get("pid")
                owner = status.get("current_owner", "unknown")
                heartbeat_age = status.get("heartbeat_age_seconds", 0.0)

                # If process is actively running, protect lease from being purged
                if holder_pid and gpu_broker._is_pid_alive(holder_pid):
                    return False, (
                        f"Cannot purge GPU lease: Process PID {holder_pid} ('{owner}') is actively running "
                        f"(heartbeat age: {int(heartbeat_age)}s)."
                    )

        return True, "Policy check passed."

    def record_execution(self, result: RecoveryActionResult):
        """
        Records an execution event for persistent audit and circuit breaker tracking.
        Counts both successful and failed attempts to prevent rapid flapping.
        """
        if result.status in ("success", "failed"):
            self._action_history[result.action_type].append(time.time())

        self._audit_log.append(result)
        if len(self._audit_log) > 200:
            self._audit_log.pop(0)

        self._save_audit_log()

    def get_audit_log(self) -> List[RecoveryActionResult]:
        """Returns the audit log."""
        return list(self._audit_log)


    def reset_circuit_breaker(self, action_type=None):
        if action_type is not None:
            self._action_history[action_type].clear()
        else:
            self._action_history.clear()


recovery_policy_engine = RecoveryPolicyEngine()
