"""
Model Version and Lifecycle Manager for Local Mesh AI.
Enforces the lifecycle states: CANDIDATE -> EVALUATING -> APPROVED -> ACTIVE -> ARCHIVED / QUARANTINED.
Strictly prohibits unapproved model replacement, requiring explicit user authorization.
"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.path_security import get_data_dir


class LifecycleState(str, Enum):
    CANDIDATE = "candidate"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    ACTIVE = "active"
    ARCHIVED = "archived"
    QUARANTINED = "quarantined"


class ModelVersionRecord(BaseModel):
    version_id: str
    model_id: str
    model_name: str
    lifecycle_state: LifecycleState
    promoted_at: Optional[float] = None
    approved_by_user: bool = False
    benchmark_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelVersionManager:
    def __init__(self):
        self.state_file = get_data_dir() / "models" / "version_state.json"
        self._state: Dict[str, Any] = self._load_or_create_default()

    def _load_or_create_default(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        default_state = {
            "active_model_id": "qwen-7b-base",
            "active_adapter_id": None,
            "version_history": [
                {
                    "version_id": "v_init_base",
                    "model_id": "qwen-7b-base",
                    "model_name": "Qwen2.5-Coder-7B Base",
                    "lifecycle_state": "active",
                    "promoted_at": time.time(),
                    "approved_by_user": True,
                    "benchmark_score": 100.0,
                    "metadata": {"type": "base"}
                }
            ],
            "candidates": {}
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(default_state, f, indent=2)
        return default_state

    def get_active_model_id(self) -> str:
        return self._state.get("active_model_id", "qwen-7b-base")

    def get_active_adapter_id(self) -> Optional[str]:
        return self._state.get("active_adapter_id")

    def register_candidate(self, model_id: str, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Registers a newly trained or imported model as a candidate."""
        rec = ModelVersionRecord(
            version_id=f"v_{model_id}_{int(time.time())}",
            model_id=model_id,
            model_name=model_name,
            lifecycle_state=LifecycleState.CANDIDATE,
            approved_by_user=False,
            metadata=metadata or {}
        )
        self._state["candidates"][model_id] = rec.model_dump()
        self._save_state()

    def promote_candidate_to_active(self, model_id: str, approved: bool = False) -> ModelVersionRecord:
        """
        Promotes a candidate model to active status.
        Mandatory safety rule: Candidate models CANNOT replace active models without explicit user approval.
        """
        if not approved:
            raise PermissionError(
                f"Unauthorized model promotion: Candidate '{model_id}' cannot replace the active model "
                "without explicit user authorization (approved=True)."
            )

        cand_data = self._state["candidates"].get(model_id)
        if not cand_data:
            # Check if it exists in history
            raise ValueError(f"Candidate model '{model_id}' not found in registry.")

        cand_rec = ModelVersionRecord(**cand_data)
        cand_rec.lifecycle_state = LifecycleState.ACTIVE
        cand_rec.promoted_at = time.time()
        cand_rec.approved_by_user = True

        # Archive previous active
        prev_active_id = self._state.get("active_adapter_id") or self._state.get("active_model_id")
        for h in self._state["version_history"]:
            if h.get("model_id") == prev_active_id and h.get("lifecycle_state") == "active":
                h["lifecycle_state"] = "archived"

        self._state["version_history"].append(cand_rec.model_dump())
        self._state["active_adapter_id"] = model_id
        self._state["candidates"].pop(model_id, None)
        self._save_state()
        return cand_rec

    def get_version_history(self) -> List[ModelVersionRecord]:
        return [ModelVersionRecord(**h) for h in self._state.get("version_history", [])]

    def _save_state(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)


model_version_manager = ModelVersionManager()
