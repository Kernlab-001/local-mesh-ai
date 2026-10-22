"""
Tests for Async Job Manager & State Machine.
"""

import time
import pytest
from services.job_manager import JobManager, JobStatus, Job

def test_job_lifecycle():
    jm = JobManager()
    job = jm.create_job({"image_path": "dummy.png"})
    
    assert job.job_id is not None
    assert job.status in (JobStatus.QUEUED, JobStatus.VALIDATING, JobStatus.FAILED)

    retrieved = jm.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.job_id == job.job_id

def test_job_cancellation():
    job = Job("test_cancel_id", {"image_path": "test.png"})
    assert job.status == JobStatus.QUEUED
    job.cancel()
    assert job.status == JobStatus.CANCELLED
    assert job._cancel_requested is True

def test_job_state_dict():
    job = Job("dict_test", {"image_path": "test.png"})
    d = job.to_dict()
    assert d["job_id"] == "dict_test"
    assert d["status"] == JobStatus.QUEUED
    assert "elapsed_seconds" in d
