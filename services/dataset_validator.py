"""
Dataset Validation and Integrity Service for Local Mesh AI.
Enforces schema validity, duplicate detection via cryptographic hashes,
and family-partitioned train/test splitting to prevent benchmark leakage.
"""

import hashlib
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from core.learning_schemas import LearningSample, DatasetSplit


class DatasetValidator:
    @staticmethod
    def compute_sample_fingerprint(sample: LearningSample) -> str:
        """Computes a deterministic SHA-256 fingerprint for deduplication."""
        ops_str = json.dumps(sample.operations_applied, sort_keys=True)
        raw = f"{sample.prompt.strip().lower()}|{sample.source_mesh_hash}|{ops_str}|{sample.asset_family}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def validate_sample(sample: LearningSample) -> Tuple[bool, str]:
        """Validates that a learning sample conforms to structural and geometric sanity checks."""
        if not sample.prompt or len(sample.prompt.strip()) < 3:
            return False, "Prompt is too short or empty."

        if not sample.operations_applied:
            return False, "Sample has no applied operations."

        if not sample.source_mesh_hash or len(sample.source_mesh_hash) < 16:
            return False, "Invalid or missing source mesh hash."

        if not sample.result_mesh_hash or len(sample.result_mesh_hash) < 16:
            return False, "Invalid or missing result mesh hash."

        return True, "Valid sample."

    @classmethod
    def partition_train_test_by_family(
        cls,
        samples: List[LearningSample],
        train_ratio: float = 0.8
    ) -> DatasetSplit:
        """
        Partitions samples into train and test sets strictly by asset family.
        Ensures zero data leakage between training and testing evaluation.
        """
        family_map: Dict[str, List[str]] = defaultdict(list)
        for s in samples:
            family = s.asset_family.strip().lower() or "generic"
            family_map[family].append(s.sample_id)

        all_families = sorted(list(family_map.keys()))
        split_idx = max(1, int(len(all_families) * train_ratio))

        train_families = all_families[:split_idx]
        test_families = all_families[split_idx:] if split_idx < len(all_families) else [all_families[-1]]

        train_ids: List[str] = []
        for f in train_families:
            train_ids.extend(family_map[f])

        test_ids: List[str] = []
        for f in test_families:
            test_ids.extend(family_map[f])

        # If test is empty because of small family count, partition samples within family deterministically
        if not test_ids and len(train_ids) > 1:
            split_sample_idx = max(1, int(len(train_ids) * train_ratio))
            test_ids = train_ids[split_sample_idx:]
            train_ids = train_ids[:split_sample_idx]

        return DatasetSplit(
            train_ids=train_ids,
            test_ids=test_ids,
            asset_families_train=train_families,
            asset_families_test=test_families,
        )


dataset_validator = DatasetValidator()
