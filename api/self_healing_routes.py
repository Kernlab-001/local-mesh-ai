"""
FastAPI Routes for Autonomous Runtime Health Diagnostics and Self-Healing.
Provides endpoints for system telemetry, issue identification, dry-run previews,
and authorized remediation execution.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from core.recovery_actions import (
    DiagnosticIssue,
    RecoveryActionRequest,
    RecoveryActionResult,
    SystemHealthSummary,
)
from services.self_healing_manager import self_healing_manager
from services.recovery_policy_engine import recovery_policy_engine

router = APIRouter(prefix="/api/v1/health", tags=["Autonomous Self-Healing"])


@router.get("/diagnostics", response_model=SystemHealthSummary)
def get_diagnostics():
    """Returns comprehensive real-time system health summary and issues."""
    try:
        return self_healing_manager.diagnose()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic inspection failed: {str(e)}")


@router.get("/recommendations", response_model=List[DiagnosticIssue])
def get_recommendations():
    """Returns list of currently active diagnostic issues and proposed actions."""
    try:
        summary = self_healing_manager.diagnose()
        return summary.active_issues
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recommendations: {str(e)}")


@router.post("/recover", response_model=RecoveryActionResult)
def execute_recovery(req: RecoveryActionRequest):
    """
    Executes or previews an autonomous remediation action.
    Requires approved=True for destructive/resource-clearing actions unless dry_run=True.
    """
    try:
        if req.dry_run:
            return self_healing_manager.preview_action(req)
        return self_healing_manager.execute_action(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery action failed: {str(e)}")


@router.get("/audit-log", response_model=List[RecoveryActionResult])
def get_recovery_audit_log():
    """Returns recent history of proposed, dry-run, and executed recovery actions."""
    return recovery_policy_engine.get_audit_log()
