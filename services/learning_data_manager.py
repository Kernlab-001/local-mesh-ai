"""
Learning Data Manager for Local Mesh AI.
Controls the storage, categorization, deduplication, review, and dataset export
of user-approved interaction samples with strict local data privacy.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from core.path_security import get_data_dir
from core.learning_schemas import (
    LearningSample,
    SampleReviewStatus,
    DatasetManifest,
)
from services.data_consent_manager import data_consent_manager
from services.dataset_validator import dataset_validator


class LearningDataManager:
    def __init__(self):
        self.base_dir = get_data_dir() / "learning"
        self.inbox_dir = self.base_dir / "inbox"
        self.approved_dir = self.base_dir / "approved"
        self.rejected_dir = self.base_dir / "rejected"
        self.datasets_dir = self.base_dir / "datasets"
        self.manifests_dir = self.base_dir / "manifests"
        self.evaluation_dir = self.base_dir / "evaluation"

        for d in [
            self.inbox_dir,
            self.approved_dir,
            self.rejected_dir,
            self.datasets_dir,
            self.manifests_dir,
            self.evaluation_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def record_interaction(self, sample: LearningSample) -> Optional[str]:
        """
        Records an interaction sample into the local inbox.
        Strictly a NO-OP if user has not explicitly consented to learning collection.
        """
        if not data_consent_manager.is_opted_in():
            return None

        valid, reason = dataset_validator.validate_sample(sample)
        if not valid:
            return None

        # Deduplication check across inbox and approved
        fp = dataset_validator.compute_sample_fingerprint(sample)
        for existing in self.get_all_samples():
            if dataset_validator.compute_sample_fingerprint(existing) == fp:
                return existing.sample_id

        sample_file = self.inbox_dir / f"{sample.sample_id}.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample.model_dump(), f, indent=2)

        return sample.sample_id

    def review_sample(
        self,
        sample_id: str,
        approved: bool,
        user_rating: Optional[int] = None
    ) -> Optional[LearningSample]:
        """Reviews a sample, moving it from inbox to approved or rejected."""
        sample_file = self.inbox_dir / f"{sample_id}.json"
        if not sample_file.exists():
            # Check if already in approved or rejected
            sample_file = self.approved_dir / f"{sample_id}.json"
            if not sample_file.exists():
                sample_file = self.rejected_dir / f"{sample_id}.json"
                if not sample_file.exists():
                    return None

        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample = LearningSample(**data)

        sample.review_status = SampleReviewStatus.APPROVED if approved else SampleReviewStatus.REJECTED
        if user_rating is not None:
            sample.user_rating = max(1, min(5, user_rating))

        target_dir = self.approved_dir if approved else self.rejected_dir
        target_file = target_dir / f"{sample_id}.json"

        # Remove from previous location if different
        if sample_file != target_file and sample_file.exists():
            sample_file.unlink()

        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(sample.model_dump(), f, indent=2)

        return sample

    def get_sample_by_id(self, sample_id: str) -> Optional[LearningSample]:
        for d in [self.inbox_dir, self.approved_dir, self.rejected_dir]:
            f = d / f"{sample_id}.json"
            if f.exists():
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        return LearningSample(**json.load(fp))
                except Exception:
                    pass
        return None

    def get_samples(self, status: Optional[SampleReviewStatus] = None) -> List[LearningSample]:
        dirs = []
        if status == SampleReviewStatus.PENDING:
            dirs = [self.inbox_dir]
        elif status == SampleReviewStatus.APPROVED:
            dirs = [self.approved_dir]
        elif status == SampleReviewStatus.REJECTED:
            dirs = [self.rejected_dir]
        else:
            dirs = [self.inbox_dir, self.approved_dir, self.rejected_dir]

        samples = []
        for d in dirs:
            for p in d.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        samples.append(LearningSample(**json.load(f)))
                except Exception:
                    pass
        return sorted(samples, key=lambda s: s.timestamp, reverse=True)

    def get_all_samples(self) -> List[LearningSample]:
        return self.get_samples(status=None)

    def build_dataset(self, dataset_name: str, train_ratio: float = 0.8) -> DatasetManifest:
        """
        Builds an instruction-tuning dataset from approved samples,
        partitioned by asset family to guarantee zero test leakage.
        """
        approved_samples = self.get_samples(status=SampleReviewStatus.APPROVED)
        splits = dataset_validator.partition_train_test_by_family(approved_samples, train_ratio=train_ratio)

        train_set = [s for s in approved_samples if s.sample_id in splits.train_ids]
        test_set = [s for s in approved_samples if s.sample_id in splits.test_ids]

        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ("_", "-")) or "dataset"
        train_file = self.datasets_dir / f"{safe_name}_train.jsonl"
        test_file = self.datasets_dir / f"{safe_name}_test.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for s in train_set:
                item = {
                    "sample_id": s.sample_id,
                    "prompt": s.prompt,
                    "operations": s.operations_applied,
                    "asset_family": s.asset_family,
                }
                f.write(json.dumps(item) + "\n")

        with open(test_file, "w", encoding="utf-8") as f:
            for s in test_set:
                item = {
                    "sample_id": s.sample_id,
                    "prompt": s.prompt,
                    "operations": s.operations_applied,
                    "asset_family": s.asset_family,
                }
                f.write(json.dumps(item) + "\n")

        manifest = DatasetManifest(
            dataset_name=safe_name,
            total_samples=len(approved_samples),
            approved_count=len(approved_samples),
            splits=splits,
            manifest_path=str(self.manifests_dir / f"{safe_name}_manifest.json"),
        )

        with open(manifest.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        return manifest

    def clear_all_learning_data(self):
        """Purges all collected learning data in compliance with user privacy."""
        for d in [self.inbox_dir, self.approved_dir, self.rejected_dir, self.datasets_dir, self.manifests_dir]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)


learning_data_manager = LearningDataManager()
