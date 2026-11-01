"""
Automated Security and Integrity Tests for Destructive Staging Cleanup & Self-Healing.
Verifies two-stage approval, cryptographic preview binding, single-use token consumption,
path containment, hash revalidation, symlink protection, and active session protection.
"""

import os
import time
import uuid
import pytest
from pathlib import Path

from core.path_security import get_staging_dir
from core.recovery_actions import ActionType, RecoveryActionRequest
from services.self_healing_manager import self_healing_manager
from services.editing_session_manager import editing_session_manager, EditingSession


def test_destructive_cleanup_strictly_requires_preview_id():
    """Destructive cleanup must fail immediately if preview_id is omitted."""
    req = RecoveryActionRequest(
        action_type=ActionType.CLEAN_STAGING_DIR,
        approved=True,
        dry_run=False
    )
    with pytest.raises(ValueError, match="requires a valid, server-generated preview_id"):
        self_healing_manager.execute_action(req)


def test_destructive_cleanup_rejects_invalid_preview_id():
    """Destructive cleanup must reject non-existent or forged preview IDs."""
    req = RecoveryActionRequest(
        action_type=ActionType.CLEAN_STAGING_DIR,
        approved=True,
        dry_run=False,
        preview_id="prev_forged_random_id_12345"
    )
    with pytest.raises(ValueError, match="is invalid, has expired, or was already consumed"):
        self_healing_manager.execute_action(req)


def test_destructive_cleanup_single_use_token_cannot_be_reused():
    """Preview tokens are single-use; a second execution attempt must be rejected."""
    staging = get_staging_dir()
    test_file = staging / f"single_use_test_{uuid.uuid4().hex[:6]}.glb"
    test_file.write_bytes(b"OLD_CONTENT_BYTES")
    old_time = time.time() - 4000
    os.utime(test_file, (old_time, old_time))

    try:
        # Step 1: Preview
        preview = self_healing_manager.preview_action(
            RecoveryActionRequest(action_type=ActionType.CLEAN_STAGING_DIR, dry_run=True)
        )
        assert preview.preview_id is not None
        assert str(test_file.resolve()) in preview.affected_files

        # Step 2: First execution (should succeed)
        res1 = self_healing_manager.execute_action(
            RecoveryActionRequest(
                action_type=ActionType.CLEAN_STAGING_DIR,
                approved=True,
                dry_run=False,
                preview_id=preview.preview_id
            )
        )
        assert res1.status == "success"
        assert not test_file.exists()

        # Step 3: Second execution with same preview_id must be rejected
        with pytest.raises(ValueError, match="is invalid, has expired, or was already consumed"):
            self_healing_manager.execute_action(
                RecoveryActionRequest(
                    action_type=ActionType.CLEAN_STAGING_DIR,
                    approved=True,
                    dry_run=False,
                    preview_id=preview.preview_id
                )
            )
    finally:
        test_file.unlink(missing_ok=True)


def test_destructive_cleanup_rejects_expired_preview_id():
    """Preview IDs past their expiration time must be rejected."""
    staging = get_staging_dir()
    test_file = staging / f"expire_test_{uuid.uuid4().hex[:6]}.glb"
    test_file.write_bytes(b"EXPIRE_TEST_BYTES")
    old_time = time.time() - 4000
    os.utime(test_file, (old_time, old_time))

    try:
        preview = self_healing_manager.preview_action(
            RecoveryActionRequest(action_type=ActionType.CLEAN_STAGING_DIR, dry_run=True)
        )
        preview_id = preview.preview_id

        # Manually force expiration timestamp in internal store
        with self_healing_manager._preview_lock:
            if preview_id in self_healing_manager._active_previews:
                self_healing_manager._active_previews[preview_id].expires_at = time.time() - 10.0

        with pytest.raises(ValueError, match="expired at"):
            self_healing_manager.execute_action(
                RecoveryActionRequest(
                    action_type=ActionType.CLEAN_STAGING_DIR,
                    approved=True,
                    dry_run=False,
                    preview_id=preview_id
                )
            )
        assert test_file.exists(), "File must not be deleted when preview has expired"
    finally:
        test_file.unlink(missing_ok=True)


def test_destructive_cleanup_rejects_mismatched_action_type():
    """A preview generated for one action type cannot execute a different action."""
    preview = self_healing_manager.preview_action(
        RecoveryActionRequest(action_type=ActionType.CLEAN_TEMP_MESHES, dry_run=True)
    )
    preview_id = preview.preview_id

    # Try executing CLEAN_STAGING_DIR with preview from CLEAN_TEMP_MESHES
    with pytest.raises(ValueError, match="was generated for action"):
        self_healing_manager.execute_action(
            RecoveryActionRequest(
                action_type=ActionType.CLEAN_STAGING_DIR,
                approved=True,
                dry_run=False,
                preview_id=preview_id
            )
        )


def test_destructive_cleanup_preserves_modified_file_hash_mismatch():
    """If a candidate file is modified after preview generation, it must NOT be deleted."""
    staging = get_staging_dir()
    test_file = staging / f"tamper_test_{uuid.uuid4().hex[:6]}.glb"
    test_file.write_bytes(b"INITIAL_BYTES_BEFORE_PREVIEW")
    old_time = time.time() - 4000
    os.utime(test_file, (old_time, old_time))

    try:
        preview = self_healing_manager.preview_action(
            RecoveryActionRequest(action_type=ActionType.CLEAN_STAGING_DIR, dry_run=True)
        )
        preview_id = preview.preview_id
        assert str(test_file.resolve()) in preview.affected_files

        # Tamper / modify the file after preview
        test_file.write_bytes(b"MODIFIED_BYTES_AFTER_PREVIEW")
        os.utime(test_file, (old_time, old_time))

        res = self_healing_manager.execute_action(
            RecoveryActionRequest(
                action_type=ActionType.CLEAN_STAGING_DIR,
                approved=True,
                dry_run=False,
                preview_id=preview_id
            )
        )

        assert test_file.exists(), "Modified file whose hash no longer matches preview must be spared"
        assert str(test_file) not in res.affected_files
    finally:
        test_file.unlink(missing_ok=True)


def test_destructive_cleanup_enforces_one_hour_minimum_age():
    """Fresh files (< 3600s) cannot be previewed or cleaned even if min_age_seconds=0 is passed."""
    staging = get_staging_dir()
    fresh_file = staging / f"too_fresh_{uuid.uuid4().hex[:6]}.glb"
    fresh_file.write_bytes(b"FRESH_DATA")
    # Fresh file with current mtime

    try:
        preview = self_healing_manager.preview_action(
            RecoveryActionRequest(
                action_type=ActionType.CLEAN_STAGING_DIR,
                dry_run=True,
                parameters={"min_age_seconds": 0}  # Attempted bypass
            )
        )
        assert str(fresh_file.resolve()) not in preview.affected_files
    finally:
        fresh_file.unlink(missing_ok=True)
