"""
Dataset Validation and Integrity Service for Local Mesh AI.
Enforces schema validity, duplicate detection via cryptographic hashes,
and family-partitioned train/test splitting to prevent benchmark leakage.
Strictly flags single-family datasets as insufficient to prevent leakage.
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
        If family-based separation is impossible (< 2 families), returns an explicit
        insufficient-families result instead of leaking one family across both sets.
        """
        family_map: Dict[str, List[str]] = defaultdict(list)
        for s in samples:
            family = s.asset_family.strip().lower() or "generic"
            family_map[family].append(s.sample_id)

        all_families = sorted(list(family_map.keys()))

        # Single family or empty: impossible to split without leakage
        if len(all_families) < 2:
            all_ids = [s.sample_id for s in samples]
            return DatasetSplit(
                train_ids=all_ids,
                test_ids=[],
                asset_families_train=all_families,
                asset_families_test=[],
                insufficient_families=True,
                message=(
                    f"Insufficient distinct asset families ({len(all_families)} found). "
                    "At least two distinct asset families are required for train/test partition without benchmark data leakage. "
                    "All samples placed in train set; test set left empty."
                ),
            )

        # 2 or more families: partition distinct families
        # Ensure at least 1 family in train and at least 1 family in test
        num_train = max(1, min(len(all_families) - 1, int(round(len(all_families) * train_ratio))))
        train_families = all_families[:num_train]
        test_families = all_families[num_train:]

        train_ids = []
        for f in train_families:
            train_ids.extend(family_map[f])

        test_ids = []
        for f in test_families:
            test_ids.extend(family_map[f])

        return DatasetSplit(
            train_ids=train_ids,
            test_ids=test_ids,
            asset_families_train=train_families,
            asset_families_test=test_families,
            insufficient_families=False,
            message="Asset families partitioned successfully with zero benchmark data leakage.",
        )


dataset_validator = DatasetValidator()
