"""
Tests for Command Schemas, Operation Allowlist & Malicious Input Rejection.
"""

import pytest
from pydantic import ValidationError
from core.command_schemas import (
    CommandPlan,
    SetMaterialCommand,
    BevelCommand,
    SetHeightCommand,
    DecimateCommand
)

def test_valid_command_plan():
    raw = {
        "version": "1.0",
        "description": "Make gold chair",
        "commands": [
            {"op": "set_material", "target_object": "Chair", "preset": "gold", "metallic": 1.0, "roughness": 0.2},
            {"op": "bevel", "target_object": "Chair", "width": 0.02, "segments": 3},
            {"op": "set_height", "target_object": "Chair", "height_meters": 1.2}
        ]
    }
    plan = CommandPlan.model_validate(raw)
    assert len(plan.commands) == 3
    assert plan.commands[0].op == "set_material"
    assert plan.commands[0].preset == "gold"

def test_reject_unknown_operation():
    raw = {
        "version": "1.0",
        "commands": [
            {"op": "delete_all_files", "target_object": "System"}
        ]
    }
    with pytest.raises(ValidationError):
        CommandPlan.model_validate(raw)

def test_reject_malicious_code_payload():
    raw = {
        "version": "1.0",
        "commands": [
            {"op": "exec", "code": "import os; os.system('rmdir /s /q C:\\')"}
        ]
    }
    with pytest.raises(ValidationError):
        CommandPlan.model_validate(raw)

def test_reject_out_of_bounds_numbers():
    # Negative bevel width
    with pytest.raises(ValidationError):
        BevelCommand(target_object="Box", width=-0.5)

    # Decimate ratio > 1.0
    with pytest.raises(ValidationError):
        DecimateCommand(target_object="Box", ratio=2.5)

    # Target height <= 0
    with pytest.raises(ValidationError):
        SetHeightCommand(target_object="Box", height_meters=-1.0)
