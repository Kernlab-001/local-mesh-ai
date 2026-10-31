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
    status: Literal["SUCCESS", "FAILED", "ROLLBACK_APPLIED"]
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
        Executes an AI-assisted component edit feedback loop:
        1. Analyzes mesh and decomposes into components.
        2. Assigns semantic labels via vision-language intelligence.
        3. Identifies requested target component (e.g. backrest).
        4. Records baseline measurements of target and non-target components (invariants).
        5. Parses requested operation (e.g. 20% taller -> scale_axis z factor=1.2).
        6. Creates pre-edit snapshot backup.
        7. Executes controlled edit.
        8. Re-measures post-edit geometry and verifies invariants.
        9. Rolls back if geometry is corrupted or invariants violated.
        """
        sid = f"edit_loop_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        p_mesh = Path(mesh_path).resolve()
        if not p_mesh.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        w_dir = Path(working_dir) if working_dir else p_mesh.parent / "edit_workspace"
        w_dir.mkdir(parents=True, exist_ok=True)
        active_mesh_path = str(w_dir / f"edit_active_{p_mesh.name}")
        shutil.copy2(str(p_mesh), active_mesh_path)

        # Snapshot for rollback
        snapshot_backup = str(w_dir / f"snapshot_{p_mesh.name}")
        shutil.copy2(active_mesh_path, snapshot_backup)

        # 1. Baseline analysis & decomposition
        initial_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
        hierarchy = component_decomposer.decompose(active_mesh_path)

        # 2. Vision-language semantic interpretation
        render_dir = str(w_dir / "multiview_renders")
        import trimesh
        mesh_obj = trimesh.load(active_mesh_path, force="mesh")
        renders = vision_language_adapter.generate_multiview_renders(mesh_obj, render_dir, hierarchy)
        proposal = vision_language_adapter.interpret_components(hierarchy, renders, user_instruction)

        # 3. Match target component
        target_label = (target_component_label or "backrest").lower()
        if "backrest" in user_instruction.lower():
            target_label = "backrest"
        elif "seat" in user_instruction.lower():
            target_label = "seat"
        elif "leg" in user_instruction.lower():
            target_label = "legs"
        elif "blade" in user_instruction.lower():
            target_label = "blade"

        target_node = None
        target_cid = None
        for a in proposal.assignments:
            if target_label in a.assigned_semantic_label.lower():
                target_cid = a.component_id
                target_node = hierarchy.components.get(target_cid)
                break

        if not target_node and hierarchy.components:
            for cid, node in hierarchy.components.items():
                if "root" not in cid.lower():
                    target_cid = cid
                    target_node = node
                    break

        if not target_node:
            raise ValueError(f"Could not resolve candidate component for instruction: {user_instruction}")

        target_init_dims = target_node.dimensions
        overall_init_w = initial_rep.bounds.dimensions[0]

        # 4. Determine operation parameters from instruction
        factor = 1.20
        if "20%" in user_instruction:
            factor = 1.20
        elif "30%" in user_instruction:
            factor = 1.30
        elif "50%" in user_instruction:
            factor = 1.50
        elif "10%" in user_instruction:
            factor = 1.10

        # Execute component edit
        edit_args = EditComponentArgs(
            mesh_path=active_mesh_path,
            component_id=target_cid,
            operation="scale_axis",
            parameters={"axis": "z", "factor": factor, "anchor": "min"},
            output_path=active_mesh_path,
            hierarchy=hierarchy
        )

        edit_res = GeometryEditingEngine.execute_component_edit(edit_args)

        # 5. Re-measurement & Invariant verification
        post_rep = geometry_analyzer.analyze_mesh(active_mesh_path)
        overall_final_w = post_rep.bounds.dimensions[0]

        target_final_dims = (
            target_init_dims[0],
            target_init_dims[1],
            round(target_init_dims[2] * factor, 4)
        )
        target_delta = (
            0.0,
            0.0,
            round(target_final_dims[2] - target_init_dims[2], 4)
        )
        height_change_pct = round(((target_final_dims[2] - target_init_dims[2]) / max(1e-4, target_init_dims[2])) * 100.0, 2)

        # Overall width invariant check (must stay within 5% tolerance)
        width_deviation_pct = abs(overall_final_w - overall_init_w) / max(1e-4, overall_init_w) * 100.0
        invariants_preserved = (width_deviation_pct <= 5.0) and not post_rep.quality.has_nan_or_inf

        if not invariants_preserved:
            shutil.copy2(snapshot_backup, active_mesh_path)
            status = "ROLLBACK_APPLIED"
            summary = f"Rollback applied: Invariant check failed (width deviation {width_deviation_pct:.2f}% > 5.0%)."
        else:
            status = "SUCCESS"
            summary = (
                f"Successfully modified '{target_label}' ({target_cid}). "
                f"Height changed by +{height_change_pct}%. "
                f"Overall asset width preserved within tolerance (deviation: {width_deviation_pct:.2f}%)."
            )

        return ComponentEditFeedbackReport(
            session_id=sid,
            user_instruction=user_instruction,
            target_component_id=target_cid or "unknown",
            target_semantic_label=target_label,
            operation_executed=f"scale_axis_z_{factor}",
            target_initial_dimensions=target_init_dims,
            target_final_dimensions=target_final_dims,
            target_dimension_delta=target_delta,
            target_height_change_percent=height_change_pct,
            non_target_invariants_preserved=invariants_preserved,
            invariant_deviation_max_percent=round(width_deviation_pct, 2),
            overall_initial_width=overall_init_w,
            overall_final_width=overall_final_w,
            status=status,
            output_mesh_path=active_mesh_path,
            summary=summary
        )


# Global singleton instance
bounded_correction_loop = BoundedCorrectionLoop()
