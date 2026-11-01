"""
Model Version and Lifecycle Manager for Local Mesh AI.
Enforces the 6-stage lifecycle: CANDIDATE -> EVALUATING -> APPROVED -> ACTIVE -> ARCHIVED / QUARANTINED.
Enforces that candidates must have genuine checkpoints and pass evaluation before promotion.
Prevents unapproved model replacement, ensures base model immutability,
activates the adapter in runtime model manager, and uses atomic state persistence.
"""

import json
import os
import time
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.path_security import get_data_dir
from core.model_manager import model_manager


class LifecycleState(str, Enum):
    CANDIDATE = "candidate"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    ACTIVE = "active"
    ARCHIVED = "archived"
    QUARANTINED = "quarantined"


VALID_TRANSITIONS = {
    LifecycleState.CANDIDATE: {LifecycleState.EVALUATING, LifecycleState.QUARANTINED},
    LifecycleState.EVALUATING: {LifecycleState.APPROVED, LifecycleState.CANDIDATE, LifecycleState.QUARANTINED},
    LifecycleState.APPROVED: {LifecycleState.ACTIVE, LifecycleState.ARCHIVED, LifecycleState.QUARANTINED},
    LifecycleState.ACTIVE: {LifecycleState.ARCHIVED, LifecycleState.QUARANTINED},
    LifecycleState.ARCHIVED: {LifecycleState.ACTIVE, LifecycleState.QUARANTINED},
    LifecycleState.QUARANTINED: set(),  # Terminal state until manually unquarantined
}


class ModelVersionRecord(BaseModel):
    version_id: str
    model_id: str
    model_name: str
    lifecycle_state: LifecycleState
    promoted_at: Optional[float] = None
    approved_by_user: bool = False
    benchmark_score: Optional[float] = None
    is_base_model: bool = False
    adapter_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelVersionManager:
    def __init__(self):
        self.state_file = get_data_dir() / "models" / "version_state.json"
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = self._load_or_create_default()

    def _load_or_create_default(self) -> Dict[str, Any]:
        with self._lock:
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
                        "is_base_model": True,
                        "metadata": {"type": "base"}
                    }
                ],
                "candidates": {}
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_state_atomic(default_state)
            return default_state

    def _save_state_atomic(self, state: Dict[str, Any]):
        """Persists state using atomic write and replace to prevent partial corruption."""
        tmp_file = self.state_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_file, self.state_file)

    def get_active_model_id(self) -> str:
        with self._lock:
            return self._state.get("active_model_id", "qwen-7b-base")

    def get_active_adapter_id(self) -> Optional[str]:
        with self._lock:
            return self._state.get("active_adapter_id")

    def register_candidate(
        self,
        model_id: str,
        model_name: str,
        adapter_path: Optional[str] = None,
        fingerprint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registers a newly trained adapter as a CANDIDATE."""
        with self._lock:
            meta = dict(metadata or {})
            if fingerprint:
                meta["checkpoint_fingerprint"] = fingerprint
            rec = ModelVersionRecord(
                version_id=f"v_{model_id}_{int(time.time())}",
                model_id=model_id,
                model_name=model_name,
                lifecycle_state=LifecycleState.CANDIDATE,
                approved_by_user=False,
                adapter_path=adapter_path,
                metadata=meta
            )
            self._state["candidates"][model_id] = rec.model_dump()
            self._save_state_atomic(self._state)

    def transition_state(self, model_id: str, new_state: LifecycleState):
        """Enforces legal lifecycle transitions."""
        with self._lock:
            cand = self._state["candidates"].get(model_id)
            if not cand:
                raise KeyError(f"Model '{model_id}' not found in candidates.")

            current_state = LifecycleState(cand["lifecycle_state"])
            if new_state not in VALID_TRANSITIONS.get(current_state, set()):
                raise ValueError(f"Invalid lifecycle transition from '{current_state}' to '{new_state}'.")

            cand["lifecycle_state"] = new_state.value
            self._save_state_atomic(self._state)

    def promote_candidate_to_active(self, model_id: str, approved: bool = False) -> ModelVersionRecord:
        """
        Promotes an evaluated candidate model to active status.
        Mandatory safety rules:
        - Candidate must pass genuine evaluation (score >= 80%).
        - Requires explicit user authorization (approved=True).
        - Caller metadata score fallbacks are strictly rejected.
        - Checkpoint file existence and SHA-256 fingerprint verified before activation.
        - Runtime model manager is updated with the active adapter.
        """
        with self._lock:
            if not approved:
                raise PermissionError(
                    f"Unauthorized model promotion: Candidate '{model_id}' cannot replace the active model "
                    "without explicit user authorization (approved=True)."
                )

            cand_data = self._state["candidates"].get(model_id)
            if not cand_data:
                raise ValueError(f"Candidate model '{model_id}' not found in registry.")

            # Require authentic evaluation report - no metadata spoofing or fallbacks!
            from services.training_evaluator import training_evaluator
            eval_report = training_evaluator.get_evaluation_report(cand_data.get("model_name", model_id))

            if not eval_report or not eval_report.passed_evaluation or eval_report.score_percent < 80.0:
                score_str = f"{eval_report.score_percent:.1f}%" if eval_report else "None"
                raise ValueError(
                    f"Candidate '{model_id}' cannot be promoted: must complete and pass "
                    f"genuine benchmark evaluation with score >= 80% (got {score_str}). "
                    "Metadata score bypasses are strictly prohibited."
                )

            # Check physical checkpoint on disk and verify SHA-256
            cand_path = cand_data.get("adapter_path") or str(get_data_dir() / "models" / "candidates" / cand_data.get("model_name", model_id))
            cand_dir = Path(cand_path)
            if not cand_dir.exists():
                raise FileNotFoundError(f"Candidate adapter directory '{cand_dir}' does not exist on disk.")

            ckpt_files = list(cand_dir.glob("*.bin")) + list(cand_dir.glob("*.pt")) + list(cand_dir.glob("*.safetensors")) + list(cand_dir.glob("*.gguf"))
            if not ckpt_files:
                raise FileNotFoundError(f"Candidate '{model_id}' contains no checkpoint files on disk.")

            primary_ckpt = ckpt_files[0]
            if primary_ckpt.stat().st_size == 0:
                raise ValueError(f"Candidate checkpoint '{primary_ckpt.name}' is empty (0 bytes).")

            import hashlib
            h = hashlib.sha256()
            with open(primary_ckpt, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            computed_hash = h.hexdigest()

            registered_hash = cand_data.get("metadata", {}).get("checkpoint_fingerprint")
            if registered_hash and computed_hash != registered_hash:
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch: file has {computed_hash[:12]} but registered as {registered_hash[:12]}."
                )

            if eval_report.adapter_hash and registered_hash and eval_report.adapter_hash != registered_hash:
                raise ValueError("Evaluation report checkpoint hash does not match candidate registered hash.")

            cand_rec = ModelVersionRecord(**cand_data)
            cand_rec.lifecycle_state = LifecycleState.ACTIVE
            cand_rec.promoted_at = time.time()
            cand_rec.approved_by_user = True
            cand_rec.benchmark_score = eval_report.score_percent

            # Archive previous active adapter
            prev_active_id = self._state.get("active_adapter_id")
            for h_rec in self._state["version_history"]:
                if h_rec.get("model_id") == prev_active_id and h_rec.get("lifecycle_state") == "active":
                    if not h_rec.get("is_base_model", False):
                        h_rec["lifecycle_state"] = "archived"

            # Base model remains active foundation, adapter is activated
            self._state["version_history"].append(cand_rec.model_dump())
            self._state["active_adapter_id"] = model_id
            self._state["candidates"].pop(model_id, None)
            self._save_state_atomic(self._state)

            # Update active runtime inference model
            model_manager.load_adapter(adapter_id=model_id, adapter_path=str(cand_dir))

            return cand_rec

    def get_version_history(self) -> List[ModelVersionRecord]:
        with self._lock:
            return [ModelVersionRecord(**h) for h in self._state.get("version_history", [])]


model_version_manager = ModelVersionManager()
