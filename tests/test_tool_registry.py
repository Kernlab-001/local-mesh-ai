"""
Unit Tests for Controlled Tool Registry & Safety Boundary (Phase 2).
Verifies:
- Only allowlisted tools can be executed.
- Arbitrary tool names or commands are rejected.
- Pydantic schema validation rejects malformed parameters.
- Bounded feedback loop converges and handles rollbacks on error.
"""

import pytest
import trimesh
from pathlib import Path
from pydantic import ValidationError
from core.tool_registry import tool_registry, ToolDefinition
from core.bounded_correction_loop import bounded_correction_loop


def test_tool_registry_allowlist():
    """Verifies that expected tools are present and allowlisted."""
    tools = tool_registry.list_tools()
    tool_names = [t.name for t in tools]

    assert "analyze_geometry" in tool_names
    assert "decompose_components" in tool_names
    assert "align_and_ground" in tool_names
    assert "calibrate_scale" in tool_names
    assert "enforce_symmetry" in tool_names
    assert "decimate_mesh" in tool_names
    assert "export_submeshes" in tool_names


def test_reject_unauthorized_tool():
    """Verifies that non-allowlisted tool calls raise ValueError."""
    with pytest.raises(ValueError, match="Unknown or unauthorized tool"):
        tool_registry.execute_tool("execute_arbitrary_shell_command", {"cmd": "rmdir /s /q"})

    with pytest.raises(ValueError, match="Unknown or unauthorized tool"):
        tool_registry.execute_tool("eval_python_code", {"code": "import os; os.system('calc')"})


def test_reject_invalid_tool_parameters(tmp_path):
    """Verifies that invalid parameter types or out-of-range bounds raise ValidationError."""
    box = trimesh.creation.box(extents=(1, 1, 1))
    p = str(tmp_path / "test_box.glb")
    box.export(p)

    # Negative target height
    with pytest.raises(ValidationError):
        tool_registry.execute_tool("calibrate_scale", {"mesh_path": p, "target_height_m": -2.0})

    # Out-of-bounds decimation ratio (> 1.0)
    with pytest.raises(ValidationError):
        tool_registry.execute_tool("decimate_mesh", {"mesh_path": p, "ratio": 1.5})


def test_safe_tool_execution(tmp_path):
    """Verifies successful execution of allowlisted tool."""
    box = trimesh.creation.box(extents=(1, 2, 3))
    p = str(tmp_path / "valid_box.glb")
    box.export(p)

    res = tool_registry.execute_tool("analyze_geometry", {"mesh_path": p})
    assert res["status"] == "success"
    assert res["tool"] == "analyze_geometry"
    assert "bounds" in res["result"]
    assert "counts" in res["result"]


def test_bounded_correction_loop_convergence(tmp_path):
    """Verifies bounded correction loop converges within 3 attempts."""
    # Create an uncalibrated box
    box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    box.apply_translation([5.0, -3.0, 2.0])  # Floating, unaligned
    p = str(tmp_path / "uncalibrated.glb")
    box.export(p)

    report = bounded_correction_loop.run_loop(
        mesh_path=p,
        target_height_m=1.2,
        target_upright=True,
        target_grounded=True,
        max_attempts=3
    )

    assert report.is_converged is True
    assert report.status == "SUCCESS"
    assert report.attempts_made <= 3
    # Check grounded and scaled
    final_mesh = trimesh.load(p, force="mesh")
    assert pytest.approx(final_mesh.vertices[:, 2].min(), abs=1e-3) == 0.0
    assert pytest.approx(final_mesh.extents[2], abs=0.05) == 1.2
