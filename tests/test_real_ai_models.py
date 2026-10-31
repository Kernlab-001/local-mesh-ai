"""
Comprehensive Integration & Verification Tests for Real AI Models (Phase 3.1).
Verifies:
1. Model Path Resolver and physical checkpoint existence
2. NVIDIA PartField neural worker and continuous feature-field decomposition
3. Qwen3-VL multimodal image perception and strict anti-hallucination guard
4. Qwen Coder dynamic technical-artist planning (arbitrary %, axes, invariants)
5. Bounded correction loop ambiguous target gating (NEEDS_COMPONENT_CONFIRMATION)
6. Bounded correction loop actual physical geometry measurement and rollback
"""

import os
import sys
import pytest
import trimesh
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows finds llama-cpp / torch DLLs
if sys.platform == "win32":
    for p in [PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib",
              PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"]:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass

from core.model_paths import model_path_resolver
from core.engines.partfield_adapter import partfield_adapter
from core.engines.vision_language_adapter import vision_language_adapter
from core.qwen_planner import qwen_planner
from core.bounded_correction_loop import bounded_correction_loop
from core.component_decomposer import ComponentHierarchy, ComponentNode, BoundingBox3D
from core.command_schemas import ComponentEditPlan, ComponentInvariant


@pytest.fixture
def test_chair_mesh_path(tmp_path):
    """Creates a standardized test chair mesh for testing."""
    seat = trimesh.creation.box(extents=[0.5, 0.5, 0.05])
    seat.apply_translation([0, 0, 0.45])

    back = trimesh.creation.box(extents=[0.5, 0.05, 0.5])
    back.apply_translation([0, -0.225, 0.725])

    leg1 = trimesh.creation.cylinder(radius=0.02, height=0.45)
    leg1.apply_translation([0.22, 0.22, 0.225])

    leg2 = trimesh.creation.cylinder(radius=0.02, height=0.45)
    leg2.apply_translation([-0.22, 0.22, 0.225])

    leg3 = trimesh.creation.cylinder(radius=0.02, height=0.45)
    leg3.apply_translation([0.22, -0.22, 0.225])

    leg4 = trimesh.creation.cylinder(radius=0.02, height=0.45)
    leg4.apply_translation([-0.22, -0.22, 0.225])

    combined = trimesh.util.concatenate([seat, back, leg1, leg2, leg3, leg4])
    obj_path = tmp_path / "test_chair.obj"
    combined.export(str(obj_path))
    return str(obj_path)


def test_model_path_resolver_physical_presence():
    """Verify that model path resolver detects physical checkpoints."""
    partfield_ckpt = model_path_resolver.resolve_partfield_checkpoint()
    assert partfield_ckpt is not None, "PartField checkpoint was not resolved."
    assert partfield_ckpt.exists(), f"PartField checkpoint not found at {partfield_ckpt}"
    assert partfield_ckpt.stat().st_size > 100 * (1024 ** 2), "PartField checkpoint is suspiciously small."

    qwen3_vl_path = model_path_resolver.resolve_qwen3_vl_path()
    assert qwen3_vl_path is not None, "Qwen3-VL path was not resolved."
    assert qwen3_vl_path.exists(), f"Qwen3-VL not found at {qwen3_vl_path}"

    qwen_coder_path = model_path_resolver.resolve_qwen_coder_path()
    assert qwen_coder_path is not None, "Qwen Coder path was not resolved."
    assert qwen_coder_path.exists(), f"Qwen Coder not found at {qwen_coder_path}"


def test_partfield_adapter_neural_execution(test_chair_mesh_path):
    """Verify NVIDIA PartField worker executes without errors and generates components."""
    status = partfield_adapter.get_status()
    assert status.value == "ready", f"PartField adapter status is {status.value}, expected ready"

    hierarchy = partfield_adapter.decompose_mesh(test_chair_mesh_path, num_clusters=4)
    assert hierarchy is not None, "PartField decomposition returned None"
    assert len(hierarchy.components) > 1, f"Expected >1 components, got {len(hierarchy.components)}"
    assert "partfield_neural" in hierarchy.methods_used, f"Expected 'partfield_neural' in {hierarchy.methods_used}"


def test_qwen3_vl_hallucination_guard():
    """Verify Qwen3-VL rejects hallucinated component IDs not in candidate list."""
    dummy_hierarchy = ComponentHierarchy(
        root_asset_id="test_asset",
        provenance_mesh_path="dummy.obj",
        total_components=2,
        root_component_ids=["part_01", "part_02"],
        components={
            "part_01": ComponentNode(
                component_id="part_01",
                bounds=BoundingBox3D(min_point=(0, 0, 0), max_point=(1, 1, 1), dimensions=(1, 1, 1), center=(0.5, 0.5, 0.5), volume=1.0),
                centroid=(0.5, 0.5, 0.5),
                dimensions=(1.0, 1.0, 1.0)
            ),
            "part_02": ComponentNode(
                component_id="part_02",
                bounds=BoundingBox3D(min_point=(0, 0, 0), max_point=(1, 1, 1), dimensions=(1, 1, 1), center=(0.5, 0.5, 0.5), volume=1.0),
                centroid=(0.5, 0.5, 0.5),
                dimensions=(1.0, 1.0, 1.0)
            )
        }
    )

    fake_raw_text = '{"assignments": [{"component_id": "part_99_ghost", "assigned_semantic_label": "wing", "confidence": 0.9}]}'
    proposal = vision_language_adapter._parse_and_validate_vlm_output(fake_raw_text, ["part_01", "part_02"], dummy_hierarchy)

    # Must NOT contain part_99_ghost
    assigned_ids = [a.component_id for a in proposal.assignments]
    assert "part_99_ghost" not in assigned_ids, "Hallucinated component ID was not filtered out!"


def test_dynamic_qwen_coder_plan_arbitrary_percentage():
    """Verify Qwen planner parses non-standard percentages and sets invariants."""
    dummy_hierarchy = ComponentHierarchy(
        root_asset_id="chair_asset",
        provenance_mesh_path="dummy.obj",
        total_components=3,
        root_component_ids=["comp_backrest", "comp_seat", "comp_legs"],
        components={
            "comp_backrest": ComponentNode(
                component_id="comp_backrest",
                semantic_label="backrest",
                bounds=BoundingBox3D(min_point=(0, 0, 0.5), max_point=(1, 0.1, 1.5), dimensions=(1, 0.1, 1.0), center=(0.5, 0.05, 1.0), volume=0.1),
                centroid=(0.5, 0.05, 1.0),
                dimensions=(1.0, 0.1, 1.0)
            ),
            "comp_seat": ComponentNode(
                component_id="comp_seat",
                semantic_label="seat",
                bounds=BoundingBox3D(min_point=(0, 0, 0.4), max_point=(1, 1, 0.5), dimensions=(1, 1, 0.1), center=(0.5, 0.5, 0.45), volume=0.1),
                centroid=(0.5, 0.5, 0.45),
                dimensions=(1.0, 1.0, 0.1)
            ),
            "comp_legs": ComponentNode(
                component_id="comp_legs",
                semantic_label="leg",
                bounds=BoundingBox3D(min_point=(0, 0, 0), max_point=(1, 1, 0.4), dimensions=(1, 1, 0.4), center=(0.5, 0.5, 0.2), volume=0.4),
                centroid=(0.5, 0.5, 0.2),
                dimensions=(1.0, 1.0, 0.4)
            )
        }
    )

    plan = qwen_planner.generate_component_edit_plan(
        user_instruction="Make the backrest 23% taller without changing seat dimensions",
        hierarchy=dummy_hierarchy
    )

    assert plan.status == "READY", f"Expected READY, got {plan.status}"
    assert abs(plan.scale_factor - 1.23) < 0.02, f"Expected factor ~1.23 for 23%, got {plan.scale_factor}"
    assert plan.axis in ("z", "all"), f"Expected z-axis for taller, got {plan.axis}"
    assert "comp_backrest" in plan.target_component_ids

    # Verify invariants: seat must be preserved as requested
    invariant_ids = [inv.component_id for inv in plan.invariants]
    assert "comp_seat" in invariant_ids, "comp_seat was not included in invariants"


def test_bounded_loop_ambiguous_target_rejection(test_chair_mesh_path, tmp_path):
    """Verify that ambiguous or non-existent target components trigger NEEDS_COMPONENT_CONFIRMATION."""
    # Instruct edit on a non-existent component (e.g. 'armrest' on an armless chair)
    report = bounded_correction_loop.run_component_edit_loop(
        mesh_path=test_chair_mesh_path,
        user_instruction="Make the armrest 30% wider",
        working_dir=str(tmp_path / "workspace")
    )

    assert report.status == "NEEDS_COMPONENT_CONFIRMATION", f"Expected NEEDS_COMPONENT_CONFIRMATION, got {report.status}"
    assert report.operation_executed == "none"
    assert len(report.candidate_components) > 0, "Expected candidate components to be listed for confirmation."


def test_bounded_loop_dynamic_edit_and_invariants(test_chair_mesh_path, tmp_path):
    """Verify valid component edit achieves target scaling and preserves non-target invariants."""
    report = bounded_correction_loop.run_component_edit_loop(
        mesh_path=test_chair_mesh_path,
        user_instruction="Make the backrest 20% taller without altering seat dimensions",
        working_dir=str(tmp_path / "workspace")
    )

    assert report.status == "SUCCESS", f"Expected SUCCESS, got {report.status}: {report.summary}"
    assert abs(report.target_height_change_percent - 20.0) < 2.0, f"Expected ~20% change, got {report.target_height_change_percent}%"
    assert report.non_target_invariants_preserved is True, "Expected non-target invariants to be preserved."
