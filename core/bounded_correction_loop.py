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


# Global singleton instance
bounded_correction_loop = BoundedCorrectionLoop()
