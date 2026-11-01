import os
import sys

# Ensure project root E:\GPUPLAT is in sys.path during test runs
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest

@pytest.fixture(autouse=True)
def reset_circuit_breaker_isolation():
    try:
        from services.recovery_policy_engine import recovery_policy_engine
        recovery_policy_engine.reset_circuit_breaker()
    except Exception:
        pass
    yield
