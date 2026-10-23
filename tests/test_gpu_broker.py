"""
Tests for Cross-Process GPU Lease Broker.
"""

import os
import pytest
from core.gpu_broker import GPUBroker

def test_gpu_broker_lifecycle(tmp_path):
    lock_file = str(tmp_path / "test_gpu.lock")
    broker = GPUBroker(lock_file=lock_file)

    assert not broker.is_locked()
    
    # Acquire lease
    acquired = broker.acquire("TestWorker", timeout=2.0)
    assert acquired
    assert broker.is_locked()

    status = broker.get_status()
    assert status["locked"]
    assert status["holder"]["client"] == "TestWorker"

    # Second acquisition should fail with short timeout
    second_acq = broker.acquire("WorkerTwo", timeout=0.5)
    assert not second_acq

    # Release lease
    broker.release("TestWorker")
    assert not broker.is_locked()

def test_gpu_broker_context_manager(tmp_path):
    lock_file = str(tmp_path / "test_gpu_ctx.lock")
    broker = GPUBroker(lock_file=lock_file)

    with broker.lease("ContextWorker", timeout=2.0):
        assert broker.is_locked()

    assert not broker.is_locked()

def test_gpu_broker_prevent_foreign_release(tmp_path):
    lock_file = str(tmp_path / "test_gpu_foreign.lock")
    broker = GPUBroker(lock_file=lock_file)

    acquired = broker.acquire("AuthorizedClient", timeout=2.0)
    assert acquired
    assert broker.is_locked()

    # Foreign client attempts release -> must be rejected!
    broker.release("UnauthorizedClient")
    assert broker.is_locked()

    # Authorized release succeeds
    broker.release("AuthorizedClient")
    assert not broker.is_locked()

def test_gpu_broker_cleanup_on_exception(tmp_path):
    lock_file = str(tmp_path / "test_gpu_err.lock")
    broker = GPUBroker(lock_file=lock_file)

    with pytest.raises(RuntimeError):
        with broker.lease("CrashWorker", timeout=2.0):
            assert broker.is_locked()
            raise RuntimeError("Simulated worker failure during GPU execution")

    # Lock must be safely released despite exception
    assert not broker.is_locked()
