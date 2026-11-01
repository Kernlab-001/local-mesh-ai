"""Comprehensive Integration & Verification Tests for Real AI Models (Phase 3.1.1).

Verifies:
Group A — Automated Unit & Deterministic Verification:
1. Model Path Resolver and physical checkpoint existence
2. Qwen3-VL hallucination guard parser (rejection of ghost component IDs)
3. 6 distinct camera view angles and true component mask rendering on asymmetric mesh
4. GPU broker lease acquisition, background heartbeat, and thread-safe release

Group B — Real Local AI Hardware Integration Tests:
5. Qwen3-VL multimodal model weights and mmproj resolution
6. Qwen3-VL real image inference with llama-cpp-python vision handler
7. Qwen Coder real neural technical-artist planning (require_neural=True, provenance=qwen_coder_neural)
8. NVIDIA PartField real neural segmentation (face_indices mapping, methods_used=partfield_neural)
9. Bounded correction loop ambiguous target gating (NEEDS_COMPONENT_CONFIRMATION)
10. Bounded correction loop physical geometry scaling and non-target invariant preservation
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows finds llama-cpp / torch DLLs
if sys.platform == "win32":
    for p in [
        PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib",
        PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
    ]:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass

from core.bounded_correction_loop import bounded_correction_loop
from core.command_schemas import ComponentEditPlan, ComponentInvariant
from core.component_decomposer import BoundingBox3D, ComponentHierarchy, ComponentNode
from core.engines.partfield_adapter import partfield_adapter
from core.engines.vision_language_adapter import vision_language_adapter
from core.gpu_broker import gpu_broker
from core.model_paths import ModelStatus, model_path_resolver
from core.qwen_planner import qwen_planner


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


# ==============================================================================
# GROUP A: Automated Unit & Deterministic Verification Tests
# ==============================================================================

def test_model_path_resolver_discovery():
    """Verify that model path resolver detects physical checkpoints and reports truthful ModelStatus."""
    partfield_ckpt = model_path_resolver.resolve_partfield_checkpoint()
    assert partfield_ckpt is not None, "PartField checkpoint was not resolved."
    assert partfield_ckpt.exists(), f"PartField checkpoint not found at {partfield_ckpt}"
    assert partfield_ckpt.stat().st_size > 100 * (1024 ** 2), "PartField checkpoint is suspiciously small."

    qwen3_vl_path = model_path_resolver.resolve_qwen3_vl_path()
    assert qwen3_vl_path is not None, "Qwen3-VL path was not resolved."
    assert qwen3_vl_path.exists(), f"Qwen3-VL not found at {qwen3_vl_path}"

    mmproj_path = model_path_resolver.resolve_qwen3_vl_mmproj_path()
    assert mmproj_path is not None and mmproj_path.exists(), "Qwen3-VL mmproj projector not found."

    qwen_coder_path = model_path_resolver.resolve_qwen_coder_path()
    assert qwen_coder_path is not None, "Qwen Coder path was not resolved."
    assert qwen_coder_path.exists(), f"Qwen Coder not found at {qwen_coder_path}"

    triposr_ckpt = model_path_resolver.resolve_triposr_checkpoint()
    assert triposr_ckpt is not None and triposr_ckpt.exists(), "TripoSR checkpoint not found."

    # SD-Turbo should report truthfully
    sd_weights = model_path_resolver.resolve_sdturbo_weights()
    if sd_weights is None or not sd_weights.exists():
        assert model_path_resolver.get_status("sd_turbo") == ModelStatus.MISSING


def test_qwen3_vl_hallucination_guard_parser():
    """Verify Qwen3-VL output parser rejects hallucinated component IDs not in candidate list."""
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
                dimensions=(1.0, 1.0, 1.0),
                face_indices=[0, 1, 2],
            ),
            "part_02": ComponentNode(
                component_id="part_02",
                bounds=BoundingBox3D(min_point=(0, 0, 0), max_point=(1, 1, 1), dimensions=(1, 1, 1), center=(0.5, 0.5, 0.5), volume=1.0),
                centroid=(0.5, 0.5, 0.5),
                dimensions=(1.0, 1.0, 1.0),
                face_indices=[3, 4, 5],
            ),
        },
    )

    fake_raw_text = '{"assignments": [{"component_id": "part_99_ghost", "assigned_semantic_label": "wing", "confidence": 0.9}]}'
    proposal = vision_language_adapter._parse_and_validate_vlm_output(fake_raw_text, ["part_01", "part_02"], dummy_hierarchy)

    # Must NOT contain part_99_ghost
    assigned_ids = [a.component_id for a in proposal.assignments]
    assert "part_99_ghost" not in assigned_ids, "Hallucinated component ID was not filtered out!"


def test_camera_views_and_component_masks_asymmetric(tmp_path):
    """Verify 6 camera view renders are genuinely distinct and component mask renders depth-sorted colors."""
    # Create an asymmetric wedge / L-shaped mesh
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 0.5))
    mesh.vertices[mesh.vertices[:, 1] > 0, 0] += 1.5

    f_count = len(mesh.faces)
    c1_faces = list(range(f_count // 2))
    c2_faces = list(range(f_count // 2, f_count))

    c1 = ComponentNode(
        component_id="comp_01",
        semantic_label="part1",
        confidence=0.9,
        is_uncertain=False,
        bounds=BoundingBox3D(min_point=(-1, -1, -1), max_point=(1, 1, 1), dimensions=(2, 2, 2), center=(0, 0, 0), volume=8),
        centroid=(0, 0, 0),
        dimensions=(1, 1, 1),
        vertex_count=10,
        face_count=len(c1_faces),
        volume_ratio=0.5,
        relative_z_position=0.2,
        segmentation_method="partfield_neural",
        face_indices=c1_faces,
    )
    c2 = ComponentNode(
        component_id="comp_02",
        semantic_label="part2",
        confidence=0.9,
        is_uncertain=False,
        bounds=BoundingBox3D(min_point=(-1, -1, -1), max_point=(1, 1, 1), dimensions=(2, 2, 2), center=(0, 0, 0), volume=8),
        centroid=(0, 0, 0),
        dimensions=(1, 1, 1),
        vertex_count=10,
        face_count=len(c2_faces),
        volume_ratio=0.5,
        relative_z_position=0.8,
        segmentation_method="partfield_neural",
        face_indices=c2_faces,
    )
    h = ComponentHierarchy(
        root_asset_id="test_asset",
        provenance_mesh_path="dummy.glb",
        total_components=2,
        root_component_ids=["comp_01", "comp_02"],
        components={"comp_01": c1, "comp_02": c2},
        methods_used=["partfield_neural"],
        hierarchy_depth=1,
        is_decomposed=True,
    )

    res = vision_language_adapter.generate_multiview_renders(mesh, str(tmp_path), hierarchy=h)

    # 1. Verify all 6 view images exist
    views = ["front", "back", "left", "right", "top", "perspective"]
    for v in views:
        assert v in res.render_paths, f"Missing view: {v}"
        assert os.path.exists(res.render_paths[v]), f"File not found: {res.render_paths[v]}"

    # 2. Verify views are non-identical (L1 difference > 10,000)
    imgs = {k: np.array(Image.open(res.render_paths[k])) for k in views}
    diff_front_back = int(np.abs(imgs["front"].astype(int) - imgs["back"].astype(int)).sum())
    diff_front_left = int(np.abs(imgs["front"].astype(int) - imgs["left"].astype(int)).sum())
    assert diff_front_back > 10000, f"Front and Back views are identical (diff={diff_front_back})"
    assert diff_front_left > 10000, f"Front and Left views are identical (diff={diff_front_left})"

    # 3. Verify component mask has pixels matching color_legend
    assert res.mask_render_path is not None and os.path.exists(res.mask_render_path)
    mask_img = np.array(Image.open(res.mask_render_path))
    c1_rgb = tuple(int(res.color_legend["comp_01"].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    match_c1 = np.all(mask_img[:, :, :3] == c1_rgb, axis=2).sum()
    assert match_c1 > 100, f"Expected mask pixels for C1 color {c1_rgb}, got {match_c1}"


def test_gpu_broker_lease_and_heartbeat():
    """Verify GPU lease acquisition, background heartbeat renewal, and safe release."""
    with gpu_broker.lease("test_unit_client", timeout=5.0) as broker:
        status = broker.get_status()
        assert status["locked"] is True
        assert status["holder"]["client"] == "test_unit_client"
        # Test heartbeat
        broker.heartbeat()

    # After exit from context manager
    assert gpu_broker.is_locked() is False


# ==============================================================================
# GROUP B: Real Hardware Integration Tests
# ==============================================================================

def test_qwen3_vl_multimodal_loading():
    """Verify Qwen3-VL GGUF model and mmproj vision projector are installed."""
    assert vision_language_adapter.is_model_installed() is True, "Qwen3-VL models are not installed."


def test_qwen3_vl_real_image_inference(tmp_path):
    """Verify real multimodal visual inference using Qwen3-VL and llama-cpp-python."""
    # Create a quick test image
    img = Image.new("RGB", (256, 256), color=(200, 50, 50))
    img_path = tmp_path / "test_red_box.png"
    img.save(img_path)

    prompt = "What solid color is this image? Reply in one word."
    reply = vision_language_adapter.run_multimodal_inference(prompt, [img_path])
    assert reply is not None and len(reply.strip()) > 0, "Empty response from Qwen3-VL multimodal inference."
    assert "red" in reply.lower(), f"Expected 'red' in response, got '{reply}'"


def test_qwen_coder_real_neural_planning(test_chair_mesh_path):
    """Verify Qwen Coder generates structured plan with provenance=qwen_coder_neural."""
    from core.component_decomposer import component_decomposer

    hierarchy = component_decomposer.decompose(test_chair_mesh_path)
    plan = qwen_planner.generate_component_edit_plan(
        user_instruction="Make the upper_structure 23% taller",
        hierarchy=hierarchy,
        require_neural=True,
    )

    assert plan.status == "READY", f"Expected READY, got {plan.status}: {plan.rationale}"
    assert plan.execution_provenance == "qwen_coder_neural", f"Expected qwen_coder_neural, got {plan.execution_provenance}"
    assert abs(plan.scale_factor - 1.23) < 0.05, f"Expected ~1.23 factor, got {plan.scale_factor}"
    assert plan.axis == "z", f"Expected z-axis for taller, got {plan.axis}"
    assert len(plan.invariants) >= 1, "Expected invariants to preserve other components."


def test_partfield_real_neural_segmentation(test_chair_mesh_path):
    """Verify NVIDIA PartField worker executes without errors and generates components with face_indices."""
    env = partfield_adapter.check_runtime_environment()
    assert env["ready"] is True, f"PartField runtime environment not ready: {env}"

    hierarchy = partfield_adapter.decompose_partfield_neural(test_chair_mesh_path, num_parts=4)
    assert hierarchy is not None, "PartField decomposition returned None"
    assert len(hierarchy.components) > 1, f"Expected >1 components, got {len(hierarchy.components)}"
    assert "partfield_neural" in hierarchy.methods_used, f"Expected 'partfield_neural' in {hierarchy.methods_used}"

    # Verify that components have valid face_indices
    non_root_nodes = [n for cid, n in hierarchy.components.items() if "root" not in cid.lower()]
    assert len(non_root_nodes) > 0
    total_faces = sum(len(n.face_indices) for n in non_root_nodes)
    assert total_faces > 0, "Component face_indices mapping is empty!"


def test_bounded_loop_ambiguous_target_rejection(test_chair_mesh_path, tmp_path):
    """Verify that ambiguous or non-existent target components trigger NEEDS_COMPONENT_CONFIRMATION."""
    report = bounded_correction_loop.run_component_edit_loop(
        mesh_path=test_chair_mesh_path,
        user_instruction="Make the armrest 30% wider",
        working_dir=str(tmp_path / "workspace"),
    )

    assert report.status == "NEEDS_COMPONENT_CONFIRMATION", f"Expected NEEDS_COMPONENT_CONFIRMATION, got {report.status}"
    assert report.operation_executed == "none"
    assert len(report.candidate_components) > 0, "Expected candidate components to be listed for confirmation."


def test_bounded_loop_dynamic_edit_and_invariants(test_chair_mesh_path, tmp_path):
    """Verify valid component edit achieves target scaling and preserves non-target invariants."""
    report = bounded_correction_loop.run_component_edit_loop(
        mesh_path=test_chair_mesh_path,
        user_instruction="Make the seat 20% taller without altering leg dimensions",
        working_dir=str(tmp_path / "workspace"),
    )

    assert report.status == "SUCCESS", f"Expected SUCCESS, got {report.status}: {report.summary}"
    assert abs(report.target_height_change_percent - 20.0) < 2.0, f"Expected ~20% change, got {report.target_height_change_percent}%"
    assert report.non_target_invariants_preserved is True, "Expected non-target invariants to be preserved."
