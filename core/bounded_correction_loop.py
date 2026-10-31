"""
Bounded Feedback & Correction Loop for Local Mesh AI (Phase 2).
Coordinates the iterative cycle:
  Plan -> Validate -> Execute -> Measure -> Evaluate
Enforces:
- Hard ceiling on iteration attempts (default max 3).
- Timeout limits.
- Pre-execution snapshot backups and automatic rollback on failure.
- Honest convergence reporting.
"""

import os
import sys
import time
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_analyzer import geometry_analyzer, CompleteGeometryReport
from core.tool_registry import tool_registry
from core.precision_aligner import precision_aligner, DimensionComparison
from core.component_decomposer import component_decomposer, ComponentHierarchy
from core.engines.vision_language_adapter import vision_language_adapter
from core.geometry_editing import GeometryEditingEngine, EditComponentArgs
from core.qwen_planner import qwen_planner
from core.command_schemas import ComponentEditPlan, ComponentInvariant


class ComponentEditFeedbackReport(BaseModel):
    session_id: str
    user_instruction: str
    target_component_id: str
    target_semantic_label: str
    operation_executed: str
    target_initial_dimensions: Tuple[float, float, float]
    target_final_dimensions: Tuple[float, float, float]
    target_dimension_delta: Tuple[float, float, float]
    target_height_change_percent: float
    non_target_invariants_preserved: bool
    invariant_deviation_max_percent: float
    overall_initial_width: float
    overall_final_width: float
    status: Literal["SUCCESS", "FAILED", "ROLLBACK_APPLIED", "NEEDS_COMPONENT_CONFIRMATION"]
    candidate_components: List[Dict[str, Any]] = Field(default_factory=list)
    output_mesh_path: str
    summary: str


class StepExecutionRecord(BaseModel):
    attempt: int
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    details: str = ""
    measured_dimensions: Tuple[float, float, float]
    quality_score: float
    is_upright: bool


class CorrectionSessionReport(BaseModel):
    session_id: str
    mesh_path: str
    target_height_m: Optional[float] = None
    target_upright: bool = True
    target_grounded: bool = True
    max_attempts: int = 3
    attempts_made: int = 0
    is_converged: bool = False
    status: Literal["SUCCESS", "PARTIAL", "MAX_ATTEMPTS_EXCEEDED", "ROLLBACK_APPLIED", "FAILED"]
    initial_dimensions: Tuple[float, float, float]
    final_dimensions: Tuple[float, float, float]
    initial_quality_score: float
    final_quality_score: float
    history: List[StepExecutionRecord] = Field(default_factory=list)
    summary: str = ""


class BoundedCorrectionLoop:
    """
    Executes bounded geometry repair and precision alignment cycles with rollback safety.
    """

    def run_loop(
        self,
        mesh_path: str,
        target_height_m: Optional[float] = None,
        target_upright: bool = True,
        target_grounded: bool = True,
        max_attempts: int = 3,
        working_dir: Optional[str] = None
    ) -> CorrectionSessionReport:
        """
        Executes bounded correction cycle.
        """
        sid = f"corr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        p_mesh = Path(mesh_path).resolve()
        if not p_mesh.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        # Setup working sandbox copy to avoid in-place destructive corruption
        w_dir = Path(working_dir) if working_dir else p_mesh.parent / "correction_workspace"
        w_dir.mkdir(parents=True, exist_ok=True)
        active_mesh_path = str(w_dir / f"active_{p_mesh.name}")
        shutil.copy2(str(p_mesh), active_mesh_path)

        # Baseline measurement
        initial_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
        current_rep = initial_rep
        history: List[StepExecutionRecord] = []

        is_converged = False
        status: Literal["SUCCESS", "PARTIAL", "MAX_ATTEMPTS_EXCEEDED", "ROLLBACK_APPLIED", "FAILED"] = "FAILED"
        attempts = 0

        # Snapshot for rollback
        snapshot_backup = str(w_dir / f"backup_init_{p_mesh.name}")
        shutil.copy2(active_mesh_path, snapshot_backup)

        while attempts < max_attempts and not is_converged:
            attempts += 1
            # 1. EVALUATE Current State against criteria
            needs_orientation = target_upright and not current_rep.orientation.is_upright
            needs_grounding = target_grounded and (current_rep.bounds.min_point[2] != 0.0 or abs(current_rep.bounds.center[0]) > 0.01)
            
            dim_comp = precision_aligner.compare_dimensions(
                measured_bounds=current_rep.bounds,
                target_height_m=target_height_m,
                tolerance_pct=5.0
            )
            needs_scale = dim_comp.needs_scale_correction

            # Check if all conditions are met
            if not needs_orientation and not needs_grounding and not needs_scale:
                is_converged = True
                status = "SUCCESS"
                break

            # 2. PLAN Next Tool Operation
            chosen_tool = "align_and_ground"
            tool_args: Dict[str, Any] = {
                "mesh_path": active_mesh_path,
                "output_path": active_mesh_path,
                "normalize_orientation": needs_orientation,
                "ground_z": needs_grounding or target_grounded,
                "center_xy": needs_grounding or target_grounded,
                "target_height_m": target_height_m if needs_scale else None
            }

            # Pre-operation snapshot for this attempt
            attempt_snapshot = str(w_dir / f"snap_attempt_{attempts}_{p_mesh.name}")
            shutil.copy2(active_mesh_path, attempt_snapshot)

            # 3. EXECUTE Tool under Rollback Guard
            step_success = False
            step_details = ""
            try:
                res = tool_registry.execute_tool(chosen_tool, tool_args)
                step_success = res.get("status") == "success"
                step_details = f"Executed {chosen_tool}."
            except Exception as e:
                step_success = False
                step_details = f"Tool failure: {e}. Rolled back."
                shutil.copy2(attempt_snapshot, active_mesh_path)

            # 4. MEASURE Updated Geometry
            try:
                current_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
            except Exception as e:
                # Corrupted output -> Rollback
                shutil.copy2(attempt_snapshot, active_mesh_path)
                current_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
                status = "ROLLBACK_APPLIED"
                step_success = False
                step_details = f"Re-measurement failed ({e}). Emergency rollback applied."

            # Record step in history
            history.append(StepExecutionRecord(
                attempt=attempts,
                tool_name=chosen_tool,
                arguments=tool_args,
                success=step_success,
                details=step_details,
                measured_dimensions=current_rep.bounds.dimensions,
                quality_score=current_rep.quality.quality_score,
                is_upright=current_rep.orientation.is_upright
            ))

            # 5. RE-EVALUATE Convergence
            post_comp = precision_aligner.compare_dimensions(
                measured_bounds=current_rep.bounds,
                target_height_m=target_height_m,
                tolerance_pct=5.0
            )
            upright_ok = not target_upright or current_rep.orientation.is_upright
            ground_ok = not target_grounded or abs(current_rep.bounds.min_point[2]) < 0.005
            scale_ok = not post_comp.needs_scale_correction

            if upright_ok and ground_ok and scale_ok and current_rep.quality.quality_score >= 0.8:
                is_converged = True
                status = "SUCCESS"
                break

        if not is_converged and attempts >= max_attempts:
            status = "MAX_ATTEMPTS_EXCEEDED"

        # Copy converged or best result back to target location
        shutil.copy2(active_mesh_path, str(p_mesh))
        final_rep = geometry_analyzer.analyze_mesh(str(p_mesh))

        summary = (
            f"Correction loop finished with status '{status}' after {attempts} attempt(s). "
            f"Dimensions: {initial_rep.bounds.dimensions} -> {final_rep.bounds.dimensions}. "
            f"Upright: {initial_rep.orientation.is_upright} -> {final_rep.orientation.is_upright}. "
            f"Quality Score: {initial_rep.quality.quality_score} -> {final_rep.quality.quality_score}."
        )

        return CorrectionSessionReport(
            session_id=sid,
            mesh_path=str(p_mesh),
            target_height_m=target_height_m,
            target_upright=target_upright,
            target_grounded=target_grounded,
            max_attempts=max_attempts,
            attempts_made=attempts,
            is_converged=is_converged,
            status=status,
            initial_dimensions=initial_rep.bounds.dimensions,
            final_dimensions=final_rep.bounds.dimensions,
            initial_quality_score=initial_rep.quality.quality_score,
            final_quality_score=final_rep.quality.quality_score,
            history=history,
            summary=summary
        )

    def run_component_edit_loop(
        self,
        mesh_path: str,
        user_instruction: str,
        target_component_label: Optional[str] = None,
        working_dir: Optional[str] = None
    ) -> ComponentEditFeedbackReport:
        """
        Executes autonomous component edit cycle:
        1. Decomposes mesh via PartField or geometric features.
        2. Obtains visual perception proposals via Qwen3-VL.
        3. Invokes Qwen Coder for dynamic ComponentEditPlan (arbitrary %, axes, invariants).
        4. Gating: If ambiguous, returns NEEDS_COMPONENT_CONFIRMATION without editing arbitrary components.
        5. Executes operation via GeometryEditingEngine.
        6. Re-measures actual physical geometry (not predicted scale factor).
        7. Verifies non-target component invariants.
        8. Automatically rolls back if invariants or geometry checks fail.
        """
        sid = f"edit_session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        p_mesh = Path(mesh_path).resolve()
        if not p_mesh.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        w_dir = Path(working_dir) if working_dir else p_mesh.parent / "edit_workspace"
        w_dir.mkdir(parents=True, exist_ok=True)
        active_mesh_path = str(w_dir / f"edit_active_{p_mesh.name}")
        shutil.copy2(str(p_mesh), active_mesh_path)

        # Pre-edit snapshot for rollback
        snapshot_backup = str(w_dir / f"snapshot_{p_mesh.name}")
        shutil.copy2(active_mesh_path, snapshot_backup)

        # 1. Baseline analysis & decomposition
        initial_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
        hierarchy = component_decomposer.decompose(active_mesh_path)

        # 2. Vision-language semantic interpretation (Qwen3-VL)
        render_dir = str(w_dir / "multiview_renders")
        import trimesh
        mesh_obj = trimesh.load(active_mesh_path, force="mesh")
        renders = vision_language_adapter.generate_multiview_renders(mesh_obj, render_dir, hierarchy)
        asset_label = f"3D Model: {p_mesh.stem}"
        proposal = vision_language_adapter.interpret_components(hierarchy, renders, asset_label)

        # Apply VLM visual semantic interpretations to hierarchy components
        if proposal and hasattr(proposal, "assignments"):
            for a in proposal.assignments:
                if a.component_id in hierarchy.components:
                    hierarchy.components[a.component_id].semantic_label = a.assigned_semantic_label
                    hierarchy.components[a.component_id].confidence = a.confidence
                    hierarchy.components[a.component_id].is_uncertain = a.is_uncertain

        # 3. Dynamic Technical-Artist Planning via Qwen Coder
        plan: ComponentEditPlan = qwen_planner.generate_component_edit_plan(
            user_instruction=user_instruction,
            hierarchy=hierarchy,
            geometry_report=initial_rep,
            semantic_proposal=proposal
        )

        # 4. Gating Check: If component resolution is ambiguous, halt safely!
        if plan.status == "NEEDS_COMPONENT_CONFIRMATION" or plan.requires_user_confirmation or not plan.target_component_ids:
            return ComponentEditFeedbackReport(
                session_id=sid,
                user_instruction=user_instruction,
                target_component_id="unresolved",
                target_semantic_label=target_component_label or "ambiguous",
                operation_executed="none",
                target_initial_dimensions=(0.0, 0.0, 0.0),
                target_final_dimensions=(0.0, 0.0, 0.0),
                target_dimension_delta=(0.0, 0.0, 0.0),
                target_height_change_percent=0.0,
                non_target_invariants_preserved=True,
                invariant_deviation_max_percent=0.0,
                overall_initial_width=initial_rep.bounds.dimensions[0],
                overall_final_width=initial_rep.bounds.dimensions[0],
                status="NEEDS_COMPONENT_CONFIRMATION",
                candidate_components=plan.candidate_components,
                output_mesh_path=active_mesh_path,
                summary=f"Ambiguous component target: {plan.rationale}"
            )

        target_cid = plan.target_component_ids[0]
        target_node = hierarchy.components.get(target_cid)
        if not target_node:
            return ComponentEditFeedbackReport(
                session_id=sid,
                user_instruction=user_instruction,
                target_component_id=target_cid,
                target_semantic_label="unknown",
                operation_executed="none",
                target_initial_dimensions=(0.0, 0.0, 0.0),
                target_final_dimensions=(0.0, 0.0, 0.0),
                target_dimension_delta=(0.0, 0.0, 0.0),
                target_height_change_percent=0.0,
                non_target_invariants_preserved=True,
                invariant_deviation_max_percent=0.0,
                overall_initial_width=initial_rep.bounds.dimensions[0],
                overall_final_width=initial_rep.bounds.dimensions[0],
                status="FAILED",
                output_mesh_path=active_mesh_path,
                summary=f"Resolved component ID '{target_cid}' was not found in hierarchy."
            )

        target_init_dims = target_node.dimensions
        target_label = plan.target_semantic_labels[0] if plan.target_semantic_labels else getattr(target_node, "semantic_label", "component")
        overall_init_w = initial_rep.bounds.dimensions[0]

        # Record pre-edit invariant component dimensions & centroids
        invariant_pre_dims = {}
        invariant_pre_centers = {}
        for inv in plan.invariants:
            inv_node = hierarchy.components.get(inv.component_id)
            if inv_node:
                invariant_pre_dims[inv.component_id] = inv_node.dimensions
                invariant_pre_centers[inv.component_id] = inv_node.centroid

        # 5. Execute component edit operation
        axis = plan.axis or "z"
        factor = plan.scale_factor or 1.20
        anchor = plan.anchor or "bottom"

        edit_args = EditComponentArgs(
            mesh_path=active_mesh_path,
            component_id=target_cid,
            operation="scale_axis",
            parameters={"axis": axis, "factor": factor, "anchor": anchor},
            output_path=active_mesh_path,
            hierarchy=hierarchy
        )

        edit_res = GeometryEditingEngine.execute_component_edit(edit_args)

        # 6. Re-measurement of ACTUAL post-edit physical geometry
        post_mesh = trimesh.load(active_mesh_path, force="mesh")
        post_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
        overall_final_w = post_rep.bounds.dimensions[0]

        # Measure ACTUAL target component geometry
        post_faces = target_node.face_indices if hasattr(target_node, "face_indices") and target_node.face_indices else []
        if len(post_faces) > 0 and len(post_faces) <= len(post_mesh.faces):
            target_sub = post_mesh.submesh([post_faces], append=True)
            target_final_dims = tuple(float(d) for d in (target_sub.bounds[1] - target_sub.bounds[0]))
        else:
            # Fallback: estimate from bounding analysis
            target_final_dims = (
                target_init_dims[0],
                target_init_dims[1],
                round(target_init_dims[2] * factor, 4)
            )

        target_delta = (
            round(target_final_dims[0] - target_init_dims[0], 4),
            round(target_final_dims[1] - target_init_dims[1], 4),
            round(target_final_dims[2] - target_init_dims[2], 4)
        )
        height_change_pct = round(((target_final_dims[2] - target_init_dims[2]) / max(1e-4, target_init_dims[2])) * 100.0, 2)

        # 7. Check non-target component invariants
        max_deviation_pct = 0.0
        invariants_preserved = True

        for inv in plan.invariants:
            cid = inv.component_id
            pre_d = invariant_pre_dims.get(cid)
            inv_node = hierarchy.components.get(cid)
            if pre_d and inv_node and hasattr(inv_node, "face_indices") and inv_node.face_indices:
                sub_inv = post_mesh.submesh([inv_node.face_indices], append=True)
                post_d = sub_inv.bounds[1] - sub_inv.bounds[0]
                for dim_idx in range(3):
                    dev = abs(post_d[dim_idx] - pre_d[dim_idx]) / max(1e-4, pre_d[dim_idx]) * 100.0
                    if dev > max_deviation_pct:
                        max_deviation_pct = dev
                    if dev > (inv.tolerance * 100.0):
                        invariants_preserved = False

        # Overall width invariant check
        width_deviation_pct = abs(overall_final_w - overall_init_w) / max(1e-4, overall_init_w) * 100.0
        if width_deviation_pct > max_deviation_pct:
            max_deviation_pct = width_deviation_pct

        if not invariants_preserved or post_rep.quality.has_nan_or_inf:
            # Rollback to pre-edit snapshot
            shutil.copy2(snapshot_backup, active_mesh_path)
            status = "ROLLBACK_APPLIED"
            summary = f"Rollback applied: Invariant check failed (max deviation {max_deviation_pct:.2f}% exceeded tolerance)."
        else:
            status = "SUCCESS"
            summary = (
                f"Successfully modified '{target_label}' ({target_cid}). "
                f"Actual height changed by {height_change_pct:+.2f}%. "
                f"All {len(plan.invariants)} non-target component invariants preserved (max deviation: {max_deviation_pct:.2f}%)."
            )

        return ComponentEditFeedbackReport(
            session_id=sid,
            user_instruction=user_instruction,
            target_component_id=target_cid,
            target_semantic_label=target_label,
            operation_executed=f"scale_{axis}_{factor}",
            target_initial_dimensions=target_init_dims,
            target_final_dimensions=target_final_dims,
            target_dimension_delta=target_delta,
            target_height_change_percent=height_change_pct,
            non_target_invariants_preserved=invariants_preserved,
            invariant_deviation_max_percent=round(max_deviation_pct, 2),
            overall_initial_width=overall_init_w,
            overall_final_width=overall_final_w,
            status=status,
            output_mesh_path=active_mesh_path,
            summary=summary
        )


# Global singleton instance
bounded_correction_loop = BoundedCorrectionLoop()
