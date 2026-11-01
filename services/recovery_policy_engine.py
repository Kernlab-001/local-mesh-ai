"""
Recovery Policy Engine for Local Mesh AI.
Enforces execution safety, circuit breakers, approval requirements,
active GPU protection, and dry-run impact evaluation.
"""

import time
from typing import Dict, List, Tuple
from collections import defaultdict

from core.gpu_broker import gpu_broker
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
        self._audit_log: List[RecoveryActionResult] = []

    def validate_action_policy(self, req: RecoveryActionRequest) -> Tuple[bool, str]:
        """
        Validates whether the requested recovery action passes safety policies.
        Returns (is_allowed, reason).
        """
        now = time.time()
        # Clean older history
        self._action_history[req.action_type] = [
            t for t in self._action_history[req.action_type]
            if now - t < CIRCUIT_BREAKER_WINDOW_SECONDS
        ]

        # 1. Circuit Breaker Check
        attempts = len(self._action_history[req.action_type])
        if attempts >= MAX_ATTEMPTS_PER_WINDOW:
            return False, f"Circuit breaker tripped for '{req.action_type}': exceeded {MAX_ATTEMPTS_PER_WINDOW} executions in {int(CIRCUIT_BREAKER_WINDOW_SECONDS)}s."

        # 2. Approval Requirement Check
        if not req.dry_run and req.action_type in DESTRUCTIVE_ACTIONS:
            if not req.approved:
                return False, f"Action '{req.action_type}' modifies system resources and requires explicit user approval (approved=True)."

        # 3. Active GPU Protection
        if req.action_type == ActionType.PURGE_STALE_GPU_LEASE and not req.dry_run:
            status = gpu_broker.get_status()
            if status.get("lease_active"):
                age = status.get("lease_age_seconds", 0.0)
                owner = status.get("current_owner", "unknown")
                if age < 60.0 and "training" in owner.lower():
                    return False, f"Cannot purge GPU lease: Active training job owned by '{owner}' started only {int(age)}s ago."

        return True, "Policy check passed."

    def record_execution(self, result: RecoveryActionResult):
        """Records an execution event for audit and circuit breaker tracking."""
        if result.status == "success":
            self._action_history[result.action_type].append(time.time())
        self._audit_log.append(result)
        # Keep last 200 log entries
        if len(self._audit_log) > 200:
            self._audit_log.pop(0)

    def get_audit_log(self) -> List[RecoveryActionResult]:
        """Returns the in-memory audit log."""
        return list(self._audit_log)


recovery_policy_engine = RecoveryPolicyEngine()
