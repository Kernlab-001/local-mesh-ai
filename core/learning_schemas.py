"""
Typed Schemas for Controlled Local Learning and Dataset Lifecycle.
Provides models for user consent, interaction samples, dataset splits, and review states.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time
import uuid


class SampleReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ConsentState(BaseModel):
    opted_in: bool = False
    consent_timestamp: Optional[float] = None
    retention_days: int = 90
    anonymize_mesh_names: bool = True
    export_allowed: bool = False
    storage_mode: str = "strictly_local"  # Zero telemetry / external connections


class LearningSample(BaseModel):
    sample_id: str = Field(default_factory=lambda: f"smp_{uuid.uuid4().hex[:10]}")
    timestamp: float = Field(default_factory=time.time)
    prompt: str
    source_mesh_hash: str
    result_mesh_hash: str
    component_ids: List[str] = Field(default_factory=list)
    operations_applied: List[Dict[str, Any]] = Field(default_factory=list)
    metrics_before: Dict[str, Any] = Field(default_factory=dict)
    metrics_after: Dict[str, Any] = Field(default_factory=dict)
    review_status: SampleReviewStatus = Field(default=SampleReviewStatus.PENDING)
    user_rating: Optional[int] = None  # 1 to 5
    tags: List[str] = Field(default_factory=list)
    asset_family: str = Field(default="generic")
    anonymized: bool = True


class DatasetSplit(BaseModel):
    train_ids: List[str] = Field(default_factory=list)
    test_ids: List[str] = Field(default_factory=list)
    asset_families_train: List[str] = Field(default_factory=list)
    asset_families_test: List[str] = Field(default_factory=list)
    insufficient_families: bool = False
    message: str = "Split completed successfully without data leakage."


class DatasetManifest(BaseModel):
    manifest_id: str = Field(default_factory=lambda: f"mnf_{uuid.uuid4().hex[:8]}")
    created_at: float = Field(default_factory=time.time)
    dataset_name: str
    version: str = "1.0.0"
    total_samples: int = 0
    approved_count: int = 0
    splits: DatasetSplit = Field(default_factory=DatasetSplit)
    export_format: str = "instruction_jsonl"
    manifest_path: Optional[str] = None
