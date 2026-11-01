"""
Typed Schemas for Autonomous Runtime Health and Self-Healing Actions.
Provides strict Pydantic definitions for system diagnostics, policy constraints,
and recovery action proposals and results.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time
import uuid

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ActionType(str, Enum):
    VERIFY_BLENDER = "verify_blender"
    PURGE_STALE_GPU_LEASE = "purge_stale_gpu_lease"
    CLEAN_STAGING_DIR = "clean_staging_dir"
    CLEAN_TEMP_MESHES = "clean_temp_meshes"
    RECOVER_ORPHANED_SESSIONS = "recover_orphaned_sessions"
    VERIFY_MODEL_WEIGHTS = "verify_model_weights"
    RESET_FAILED_JOBS = "reset_failed_jobs"

class DiagnosticIssue(BaseModel):
    issue_id: str = Field(default_factory=lambda: f"issue_{uuid.uuid4().hex[:8]}")
    severity: SeverityLevel
    subsystem: str
    title: str
    description: str
    recommended_action: ActionType
    requires_approval: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    detected_at: float = Field(default_factory=time.time)

class RecoveryActionRequest(BaseModel):
    action_type: ActionType
    target_id: Optional[str] = None
    dry_run: bool = False
    approved: bool = False
    parameters: Dict[str, Any] = Field(default_factory=dict)

class RecoveryActionResult(BaseModel):
    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    action_type: ActionType
    timestamp: float = Field(default_factory=time.time)
    status: str  # "success", "dry_run", "failed", "requires_approval", "skipped"
    message: str
    changes_made: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class SystemHealthSummary(BaseModel):
    overall_status: str  # "healthy", "degraded", "critical"
    timestamp: float = Field(default_factory=time.time)
    subsystems: Dict[str, Any]
    active_issues: List[DiagnosticIssue] = Field(default_factory=list)
    recovery_recommendations: List[ActionType] = Field(default_factory=list)
