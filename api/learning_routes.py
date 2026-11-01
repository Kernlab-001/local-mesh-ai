"""
FastAPI Routes for Controlled Local Learning, Dataset Management, and Model Lifecycle.
Enforces strictly local consent boundaries, preflight safety gating, and non-destructive version promotion.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.learning_schemas import (
    ConsentState,
    LearningSample,
    SampleReviewStatus,
    DatasetManifest,
)
from services.data_consent_manager import data_consent_manager
from services.learning_data_manager import learning_data_manager
from services.training_preflight import training_preflight, PreflightReport
from services.lora_training_manager import lora_training_manager, TrainingJobStatus
from services.training_evaluator import training_evaluator, EvaluationReport
from services.model_registry import model_registry, ModelArtifact
from services.model_version_manager import model_version_manager, ModelVersionRecord
from services.model_rollback_manager import model_rollback_manager, ModelRollbackResult

router = APIRouter(prefix="/api/v1/learning", tags=["Controlled Learning & Lifecycle"])


class ConsentUpdateRequest(BaseModel):
    opted_in: bool
    anonymize_mesh_names: bool = True
    retention_days: int = 90
    export_allowed: bool = False


class SampleReviewRequest(BaseModel):
    sample_id: str
    approved: bool
    user_rating: Optional[int] = None


class BuildDatasetRequest(BaseModel):
    dataset_name: str = "custom_editing_dataset"
    train_ratio: float = 0.8


class PreflightRequest(BaseModel):
    model_name: str = "qwen2.5-coder-7b"
    force_cpu: bool = False


class StartTrainingRequest(BaseModel):
    target_model: str = "qwen2.5-coder-7b"
    adapter_name: Optional[str] = None
    dataset_path: Optional[str] = None
    epochs: int = 3
    force_cpu: bool = False
    dry_run: bool = False


class PromoteModelRequest(BaseModel):
    model_id: str
    approved: bool = False


class RollbackModelRequest(BaseModel):
    reason: str = "User initiated rollback via UI"


# --- 1. Consent ---
@router.get("/consent", response_model=ConsentState)
def get_consent():
    """Returns the current strictly local user consent state."""
    return data_consent_manager.get_consent_state()


@router.post("/consent", response_model=ConsentState)
def update_consent(req: ConsentUpdateRequest):
    """Explicitly updates user consent preferences for local learning collection."""
    return data_consent_manager.set_consent(
        opted_in=req.opted_in,
        anonymize=req.anonymize_mesh_names,
        retention_days=req.retention_days,
        export_allowed=req.export_allowed,
    )


# --- 2. Samples & Review ---
@router.get("/samples", response_model=List[LearningSample])
def list_samples(status: Optional[SampleReviewStatus] = None):
    """Lists recorded interaction samples filtered by status."""
    return learning_data_manager.get_samples(status=status)


@router.post("/review", response_model=LearningSample)
def review_sample_endpoint(req: SampleReviewRequest):
    """Approves or rejects a candidate learning sample."""
    sample = learning_data_manager.review_sample(
        sample_id=req.sample_id,
        approved=req.approved,
        user_rating=req.user_rating,
    )
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample '{req.sample_id}' not found.")
    return sample


@router.post("/dataset/build", response_model=DatasetManifest)
def build_dataset_endpoint(req: BuildDatasetRequest):
    """Partitions approved samples by asset family and exports instruction dataset."""
    try:
        return learning_data_manager.build_dataset(
            dataset_name=req.dataset_name,
            train_ratio=req.train_ratio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dataset: {str(e)}")


# --- 3. Preflight & Training ---
@router.post("/preflight", response_model=PreflightReport)
def check_preflight_endpoint(req: PreflightRequest):
    """Evaluates target model requirements against 6 GB VRAM physical constraints."""
    return training_preflight.evaluate(model_name=req.model_name, force_cpu=req.force_cpu)


@router.post("/train", response_model=TrainingJobStatus)
def start_training_endpoint(req: StartTrainingRequest):
    """Launches controlled LoRA fine-tuning worker after verifying preflight approval."""
    return lora_training_manager.start_training(
        target_model=req.target_model,
        adapter_name=req.adapter_name,
        dataset_path=req.dataset_path,
        epochs=req.epochs,
        force_cpu=req.force_cpu,
        dry_run=req.dry_run,
    )


@router.get("/train/{job_id}", response_model=TrainingJobStatus)
def get_training_status_endpoint(job_id: str):
    job = lora_training_manager.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return job


@router.post("/train/{job_id}/cancel")
def cancel_training_endpoint(job_id: str):
    cancelled = lora_training_manager.cancel_training(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail=f"Could not cancel training job '{job_id}'.")
    return {"status": "cancelled", "job_id": job_id}


# --- 4. Model Registry & Version Management ---
@router.get("/models", response_model=List[ModelArtifact])
def list_models():
    """Lists foundational models and trained candidate/active adapters."""
    return model_registry.discover_models()


@router.get("/models/history", response_model=List[ModelVersionRecord])
def get_model_history():
    """Returns lifecycle audit history of model versions."""
    return model_version_manager.get_version_history()


@router.post("/models/promote", response_model=ModelVersionRecord)
def promote_model_endpoint(req: PromoteModelRequest):
    """Promotes candidate model to active status. Gated behind explicit approval."""
    try:
        return model_version_manager.promote_candidate_to_active(
            model_id=req.model_id,
            approved=req.approved,
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Promotion failed: {str(e)}")


@router.post("/models/rollback", response_model=ModelRollbackResult)
def rollback_model_endpoint(req: RollbackModelRequest):
    """Atomically rolls back active adapter to previous verified version."""
    try:
        return model_rollback_manager.rollback_to_previous_active(reason=req.reason)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")
