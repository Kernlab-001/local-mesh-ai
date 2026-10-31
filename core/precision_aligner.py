"""
Precision Alignment, Orientation Normalization & Dimension Calibration Engine (Phase 2).
Provides deterministic 3D alignment, grounding, and metric scaling:
- Grounding: Snaps bottom-most vertices to ground plane (Z=0) and centers XY pivot.
- Orientation Normalization: Aligns dominant principal elongation axis to canonical coordinates (+Z up, +Y forward).
- Dimension Calibration: Compares measured dimensions against explicit constraints (with clear provenance:
  user_provided, reference_calibrated, uncalibrated_estimate).
- Symmetry Alignment: Aligns bilateral symmetry planes to coordinate axes (e.g. X=0).
- Non-destructive: Maintains immutable snapshot provenance and records before-and-after reports.
"""

import os
import sys
import copy
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_analyzer import CompleteGeometryReport, geometry_analyzer, BoundingBox3D


# ==============================================================================
# Precision Alignment Pydantic Schemas
# ==============================================================================

class DimensionComparison(BaseModel):
    requested_height_m: Optional[float] = None
    requested_width_m: Optional[float] = None
    requested_depth_m: Optional[float] = None
    measured_height_m: float
    measured_width_m: float
    measured_depth_m: float
    height_difference_m: float
    height_difference_pct: float
    needs_scale_correction: bool
    scale_factor_recommended: float
    scale_provenance: Literal["user_provided", "reference_calibrated", "uncalibrated_estimate"] = Field(
        default="uncalibrated_estimate",
        description="Explicit provenance: whether scale is user-verified or estimated"
    )
    notes: List[str] = Field(default_factory=list)


class AlignmentOperation(BaseModel):
    op: Literal["ground_plane", "center_xy", "align_orientation", "calibrate_scale", "align_symmetry"]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    applied: bool = True
    details: str = ""


class AlignmentResult(BaseModel):
    success: bool
    provenance_snapshot_path: Optional[str] = None
    result_mesh_path: Optional[str] = None
    operations_applied: List[AlignmentOperation] = Field(default_factory=list)
    before_dimensions: Tuple[float, float, float]
    after_dimensions: Tuple[float, float, float]
    before_center: Tuple[float, float, float]
    after_center: Tuple[float, float, float]
    before_is_upright: bool
    after_is_upright: bool
    before_symmetry_score: float
    after_symmetry_score: float
    scale_provenance: str = "uncalibrated_estimate"


# ==============================================================================
# Precision Alignment Engine Implementation
# ==============================================================================

class PrecisionAligner:
    """
    Deterministic geometry alignment, orientation normalization, and metric calibration.
    """

    def compare_dimensions(
        self,
        measured_bounds: BoundingBox3D,
        target_height_m: Optional[float] = None,
        target_width_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
        tolerance_pct: float = 5.0,
        is_user_provided: bool = True
    ) -> DimensionComparison:
        """
        Compares measured dimensions against target constraints and produces
        an actionable comparison report with explicit scale provenance.
        """
        w, d, h = measured_bounds.dimensions
        h_diff = 0.0
        h_diff_pct = 0.0
        needs_corr = False
        scale_rec = 1.0
        notes = []

        if target_height_m is not None and target_height_m > 0:
            h_diff = round(target_height_m - h, 4)
            h_diff_pct = round(abs(h_diff) / max(h, 1e-6) * 100.0, 2)
            if h_diff_pct > tolerance_pct:
                needs_corr = True
                scale_rec = round(target_height_m / max(h, 1e-6), 4)
                notes.append(f"Height deviates by {h_diff_pct:.1f}% ({h:.3f}m vs target {target_height_m:.3f}m).")
        else:
            notes.append("No target height specified; relative proportions used.")

        provenance = "user_provided" if is_user_provided and target_height_m else "uncalibrated_estimate"

        return DimensionComparison(
            requested_height_m=target_height_m,
            requested_width_m=target_width_m,
            requested_depth_m=target_depth_m,
            measured_height_m=h,
            measured_width_m=w,
            measured_depth_m=d,
            height_difference_m=h_diff,
            height_difference_pct=h_diff_pct,
            needs_scale_correction=needs_corr,
            scale_factor_recommended=scale_rec,
            scale_provenance=provenance,
            notes=notes
        )

    def execute_alignment(
        self,
        mesh_or_path: Any,
        output_path: Optional[str] = None,
        ground_z: bool = True,
        center_xy: bool = True,
        normalize_orientation: bool = True,
        target_height_m: Optional[float] = None,
        enforce_symmetry_plane: Optional[Literal["yz_plane", "xz_plane"]] = None,
        is_user_provided_scale: bool = True
    ) -> Tuple[trimesh.Trimesh, AlignmentResult]:
        """
        Applies a sequence of deterministic precision alignment and calibration operations:
        1. Non-destructive backup of initial state.
        2. Orientation normalization to +Z (if requested and needed).
        3. Bilateral symmetry centering (if requested and approximately symmetric).
        4. Centering on (0, 0) and snapping min Z to 0.0 (ground plane).
        5. Metric scale calibration against target constraints.
        6. Re-measurement and audit trail generation.
        """
        mesh, source_path = self._resolve_mesh(mesh_or_path)

        # Baseline measurement before transformations
        before_rep = geometry_analyzer.analyze_mesh(mesh)
        ops_applied: List[AlignmentOperation] = []

        # Save provenance backup if modifying an existing file
        snapshot_file = None
        if source_path and os.path.exists(source_path):
            snap_dir = Path(source_path).parent / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snapshot_file = str(snap_dir / f"snapshot_{int(Path(source_path).stat().st_mtime)}_{uuid.uuid4().hex[:6]}.glb")
            try:
                mesh.export(snapshot_file)
            except Exception:
                snapshot_file = None

        # 1. Orientation Normalization (Align to vertical +Z)
        if normalize_orientation and not before_rep.orientation.is_upright:
            orient_op = self._align_to_vertical(mesh, before_rep)
            if orient_op.applied:
                ops_applied.append(orient_op)

        # 2. Symmetry Centering across bilateral plane
        if enforce_symmetry_plane or before_rep.symmetry.is_approximately_symmetric:
            plane = enforce_symmetry_plane or before_rep.symmetry.primary_plane
            if plane in ("yz_plane", "xz_plane"):
                sym_op = self._center_symmetry_axis(mesh, plane)
                if sym_op.applied:
                    ops_applied.append(sym_op)

        # 3. Grounding & Centering
        if center_xy or ground_z:
            ground_op = self._apply_ground_and_center(mesh, ground_z=ground_z, center_xy=center_xy)
            if ground_op.applied:
                ops_applied.append(ground_op)

        # 4. Metric Scale Calibration
        scale_provenance = "uncalibrated_estimate"
        if target_height_m is not None and target_height_m > 0:
            scale_op = self._apply_metric_scale(mesh, target_height_m)
            if scale_op.applied:
                ops_applied.append(scale_op)
                scale_provenance = "user_provided" if is_user_provided_scale else "reference_calibrated"

        # Re-measurement after transformations
        after_rep = geometry_analyzer.analyze_mesh(mesh)

        # Save to output file if specified
        out_file = output_path or source_path
        if out_file and out_file != "in_memory_mesh":
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            mesh.export(out_file)

        result = AlignmentResult(
            success=True,
            provenance_snapshot_path=snapshot_file,
            result_mesh_path=out_file,
            operations_applied=ops_applied,
            before_dimensions=before_rep.bounds.dimensions,
            after_dimensions=after_rep.bounds.dimensions,
            before_center=before_rep.bounds.center,
            after_center=after_rep.bounds.center,
            before_is_upright=before_rep.orientation.is_upright,
            after_is_upright=after_rep.orientation.is_upright,
            before_symmetry_score=before_rep.symmetry.deviation_score,
            after_symmetry_score=after_rep.symmetry.deviation_score,
            scale_provenance=scale_provenance
        )

        return mesh, result

    def _resolve_mesh(self, mesh_or_path: Any) -> Tuple[trimesh.Trimesh, Optional[str]]:
        if isinstance(mesh_or_path, (str, Path)):
            p = Path(mesh_or_path).resolve()
            loaded = trimesh.load(str(p), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
            else:
                mesh = loaded
            return mesh, str(p)
        elif isinstance(mesh_or_path, trimesh.Trimesh):
            return copy.deepcopy(mesh_or_path), None
        raise TypeError(f"Invalid mesh source: {type(mesh_or_path)}")

    def _apply_ground_and_center(
        self,
        mesh: trimesh.Trimesh,
        ground_z: bool = True,
        center_xy: bool = True
    ) -> AlignmentOperation:
        """Translates mesh so (min_z = 0.0) and (center_x = 0.0, center_y = 0.0)."""
        verts = np.asarray(mesh.vertices)
        min_pt = np.min(verts, axis=0)
        max_pt = np.max(verts, axis=0)

        dx = 0.0
        dy = 0.0
        dz = 0.0

        if center_xy:
            center_x = (min_pt[0] + max_pt[0]) / 2.0
            center_y = (min_pt[1] + max_pt[1]) / 2.0
            dx = -center_x
            dy = -center_y

        if ground_z:
            dz = -min_pt[2]

        translation = np.array([dx, dy, dz], dtype=np.float64)
        mesh.apply_translation(translation)

        return AlignmentOperation(
            op="ground_plane" if ground_z else "center_xy",
            parameters={"dx": round(dx, 5), "dy": round(dy, 5), "dz": round(dz, 5), "ground_z": ground_z, "center_xy": center_xy},
            applied=True,
            details=f"Grounded mesh to Z=0 and centered XY pivot (translation offset: [{dx:.4f}, {dy:.4f}, {dz:.4f}]m)."
        )

    def _align_to_vertical(
        self,
        mesh: trimesh.Trimesh,
        report: CompleteGeometryReport
    ) -> AlignmentOperation:
        """
        Rotates mesh so that its primary principal elongation axis or detected up vector
        aligns with vertical +Z.
        """
        estimated_axis = report.orientation.estimated_up_axis
        rot_matrix = np.eye(4)

        if estimated_axis in ("+X", "-X"):
            # 90 degree rotation around Y axis
            angle = -np.pi / 2.0 if estimated_axis == "+X" else np.pi / 2.0
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        elif estimated_axis in ("+Y", "-Y"):
            # 90 degree rotation around X axis
            angle = np.pi / 2.0 if estimated_axis == "+Y" else -np.pi / 2.0
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        elif estimated_axis == "-Z":
            # Inverted upright, rotate 180 degrees around X
            rot_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            return AlignmentOperation(op="align_orientation", applied=False, details="Already upright.")

        mesh.apply_transform(rot_matrix)
        return AlignmentOperation(
            op="align_orientation",
            parameters={"from_axis": estimated_axis, "to_axis": "+Z"},
            applied=True,
            details=f"Normalized orientation: Rotated mesh from {estimated_axis} to canonical +Z vertical axis."
        )

    def _center_symmetry_axis(
        self,
        mesh: trimesh.Trimesh,
        symmetry_plane: str
    ) -> AlignmentOperation:
        """
        Centers the bilateral symmetry plane of the mesh exactly on the coordinate axis.
        For yz_plane, centers X coordinate so left and right are symmetric around X=0.
        For xz_plane, centers Y coordinate so front and back are symmetric around Y=0.
        """
        verts = np.asarray(mesh.vertices)
        min_pt = np.min(verts, axis=0)
        max_pt = np.max(verts, axis=0)
        offset = 0.0

        if symmetry_plane == "yz_plane":
            cx = (min_pt[0] + max_pt[0]) / 2.0
            if abs(cx) > 1e-5:
                mesh.apply_translation([-cx, 0.0, 0.0])
                offset = -cx
        elif symmetry_plane == "xz_plane":
            cy = (min_pt[1] + max_pt[1]) / 2.0
            if abs(cy) > 1e-5:
                mesh.apply_translation([0.0, -cy, 0.0])
                offset = -cy

        return AlignmentOperation(
            op="align_symmetry",
            parameters={"plane": symmetry_plane, "offset": round(offset, 5)},
            applied=abs(offset) > 1e-5,
            details=f"Aligned {symmetry_plane} symmetry axis to 0.0 (offset: {offset:.5f}m)."
        )

    def _apply_metric_scale(
        self,
        mesh: trimesh.Trimesh,
        target_height_m: float
    ) -> AlignmentOperation:
        """
        Scales the mesh uniformly to match target metric height.
        Preserves authentic proportions and vertex colors.
        """
        current_h = float(mesh.extents[2])
        if current_h <= 1e-6:
            return AlignmentOperation(op="calibrate_scale", applied=False, details="Mesh has near-zero height.")

        scale_factor = target_height_m / current_h
        mesh.apply_scale(scale_factor)

        return AlignmentOperation(
            op="calibrate_scale",
            parameters={"target_height_m": target_height_m, "original_height_m": round(current_h, 4), "scale_factor": round(scale_factor, 4)},
            applied=True,
            details=f"Calibrated physical scale: scaled Z-height from {current_h:.3f}m to {target_height_m:.3f}m ({scale_factor:.3f}x)."
        )


# Global singleton instance
precision_aligner = PrecisionAligner()
