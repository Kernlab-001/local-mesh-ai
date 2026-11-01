"""
Atomic Model Rollback Manager for Local Mesh AI.
Safely rolls back active neural adapters to previously verified versions with audit trail logging.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.path_security import get_data_dir
from services.model_version_manager import model_version_manager, LifecycleState


class ModelRollbackResult(BaseModel):
    success: bool
    timestamp: float = Field(default_factory=time.time)
    previous_active_id: str
    restored_active_id: str
    reason: str
    audit_id: str

    @property
    def rolled_back_from(self) -> str:
        return self.previous_active_id

    @property
    def rolled_back_to(self) -> str:
        return self.restored_active_id


class ModelRollbackManager:
    def __init__(self):
        self.audit_file = get_data_dir() / "models" / "rollback_audit.json"
        self._audit_log: List[Dict[str, Any]] = self._load_audit_log()

    def _load_audit_log(self) -> List[Dict[str, Any]]:
        if self.audit_file.exists():
            try:
                with open(self.audit_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def rollback_to_previous_active(self, reason: str = "User initiated rollback") -> ModelRollbackResult:
        """Atomically rolls back the active model to the previously active verified version."""
        history = model_version_manager.get_version_history()
        if len(history) < 2:
            raise ValueError("Cannot rollback: No previous verified model version exists in history.")

        current_active = history[-1]
        previous_active = history[-2]

        # Update version state
        model_version_manager._state["active_adapter_id"] = (
            previous_active.model_id if previous_active.model_id != "qwen-7b-base" else None
        )
        current_active.lifecycle_state = LifecycleState.ARCHIVED
        previous_active.lifecycle_state = LifecycleState.ACTIVE

        # Update in version manager state list
        model_version_manager._state["version_history"][-1] = current_active.model_dump()
        model_version_manager._state["version_history"][-2] = previous_active.model_dump()
        model_version_manager._save_state()

        audit_entry = {
            "audit_id": f"rb_{int(time.time())}",
            "timestamp": time.time(),
            "rolled_back_from": current_active.model_id,
            "rolled_back_to": previous_active.model_id,
            "reason": reason,
        }
        self._audit_log.append(audit_entry)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_file, "w", encoding="utf-8") as f:
            json.dump(self._audit_log, f, indent=2)

        return ModelRollbackResult(
            success=True,
            previous_active_id=current_active.model_id,
            restored_active_id=previous_active.model_id,
            reason=reason,
            audit_id=audit_entry["audit_id"],
        )

    def get_rollback_audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)


model_rollback_manager = ModelRollbackManager()
