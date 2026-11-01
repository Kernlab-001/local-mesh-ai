"""
Multi-Session Editing Manager & Asset State Registry for Local Mesh AI (Phase 6).
Maintains durable, multi-session tracking of staged Blender assets,
component hierarchies, multi-view visual renders, AI-generated edit plans,
deterministic modification executions, physical re-measurements, and rollback snapshots.
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_analyzer import geometry_analyzer
from core.component_decomposer import component_decomposer, ComponentHierarchy, ComponentNode
from core.engines.partfield_adapter import PartFieldAdapter
from core.engines.vision_language_adapter import vision_language_adapter, MultiViewRenderResult
from core.geometry_editing import GeometryEditingEngine, EditComponentArgs
from core.qwen_planner import qwen_planner
from core.command_schemas import ComponentEditPlan, ComponentInvariant
from core.bounded_correction_loop import ComponentEditFeedbackReport

logger = logging.getLogger("EditingSessionManager")

SESSIONS_ROOT = PROJECT_ROOT / "jobs" / "editing_sessions"
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)


class EditingSession:
    """Represents a live or persisted technical artist editing session for a 3D asset."""

    def __init__(
        self,
        session_id: str,
        asset_id: str,
        asset_name: str,
        working_dir: str,
        original_mesh_path: str,
        working_mesh_path: str,
        fingerprint: Optional[str] = None,
        object_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.asset_id = asset_id
        self.asset_name = asset_name
        self.working_dir = working_dir
        self.original_mesh_path = original_mesh_path
        self.working_mesh_path = working_mesh_path
        self.fingerprint = fingerprint or f"fp_{uuid.uuid4().hex[:8]}"
        self.object_metadata = object_metadata or {}

        self.version = 1
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at
        self.status = "INITIALIZED"  # INITIALIZED, ANALYZED, PLANNED, EXECUTED, ROLLED_BACK, FAILED

        self.pre_edit_snapshot_path: Optional[str] = None
        self.hierarchy: Optional[ComponentHierarchy] = None
        self.render_paths: Dict[str, str] = {}
        self.mask_render_path: Optional[str] = None
        self.color_legend: Dict[str, str] = {}
        self.current_plan: Optional[ComponentEditPlan] = None
        self.history: List[Dict[str, Any]] = []

        self.decomposition_provenance: str = "pending"
        self.vlm_provenance: str = "pending"
        self.summary: str = f"Asset '{asset_name}' staged for AI Technical Artist analysis."

    def to_dict(self) -> Dict[str, Any]:
        """Serializes editing session to JSON-serializable dictionary."""
        return {
            "session_id": self.session_id,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "fingerprint": self.fingerprint,
            "original_mesh_path": self.original_mesh_path,
            "working_mesh_path": self.working_mesh_path,
            "pre_edit_snapshot_path": self.pre_edit_snapshot_path,
            "has_pre_edit_snapshot": bool(self.pre_edit_snapshot_path and os.path.exists(self.pre_edit_snapshot_path)),
            "decomposition_provenance": self.decomposition_provenance,
            "vlm_provenance": self.vlm_provenance,
            "summary": self.summary,
            "object_metadata": self.object_metadata,
            "render_paths": self.render_paths,
            "mask_render_path": self.mask_render_path,
            "color_legend": self.color_legend,
            "current_plan": self.current_plan.model_dump() if self.current_plan else None,
            "history": self.history,
            "hierarchy": self.hierarchy.model_dump() if self.hierarchy else None,
        }

    def save(self):
        """Persists session state to disk atomically."""
        session_file = os.path.join(self.working_dir, "session.json")
        tmp_file = f"{session_file}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_file, session_file)
        except Exception as e:
            logger.warning(f"Failed to persist editing session {self.session_id}: {e}")


class EditingSessionManager:
    """Manages active and historical AI technical-artist editing sessions."""

    def __init__(self):
        self._sessions: Dict[str, EditingSession] = {}
        self._session_locks: Dict[str, threading.RLock] = {}
        self._lock = threading.Lock()
        self._partfield = PartFieldAdapter()

    def get_session_lock(self, session_id: str) -> threading.RLock:
        with self._lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.RLock()
            return self._session_locks[session_id]

    def _validate_session_path(self, session_id: str) -> Path:
        """Validates session ID format and ensures strict containment in SESSIONS_ROOT."""
        if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
            raise ValueError(f"Invalid session ID format: '{session_id}'")
        session_dir = (SESSIONS_ROOT / session_id).resolve()
        try:
            session_dir.relative_to(SESSIONS_ROOT.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected for session ID: '{session_id}'")
        return session_dir

    def create_session(
        self,
        asset_name: str,
        source_mesh_path: str,
        object_metadata: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[str] = None
    ) -> EditingSession:
        """Creates a new editing session and initializes its working sandboxed directory."""
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        asset_id = f"asset_{uuid.uuid4().hex[:8]}"
        session_dir = SESSIONS_ROOT / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Copy original mesh to immutable copy and working active copy
        ext = Path(source_mesh_path).suffix or ".glb"
        orig_copy = session_dir / f"original{ext}"
        working_copy = session_dir / f"working{ext}"
        shutil.copy2(source_mesh_path, str(orig_copy))
        shutil.copy2(source_mesh_path, str(working_copy))

        session = EditingSession(
            session_id=session_id,
            asset_id=asset_id,
            asset_name=asset_name,
            working_dir=str(session_dir),
            original_mesh_path=str(orig_copy),
            working_mesh_path=str(working_copy),
            fingerprint=fingerprint,
            object_metadata=object_metadata or {}
        )

        with self._lock:
            self._sessions[session_id] = session
        session.save()
        logger.info(f"Created EditingSession '{session_id}' for asset '{asset_name}'")
        return session

    def get_session(self, session_id: str) -> Optional[EditingSession]:
        """Retrieves session from memory cache or loads from disk."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

        try:
            session_dir = self._validate_session_path(session_id)
        except ValueError:
            return None

        session_file = session_dir / "session.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            session = EditingSession(
                session_id=data["session_id"],
                asset_id=data["asset_id"],
                asset_name=data["asset_name"],
                working_dir=str(session_dir),
                original_mesh_path=data.get("original_mesh_path", str(session_dir / "original.glb")),
                working_mesh_path=data.get("working_mesh_path", str(session_dir / "working.glb")),
                fingerprint=data.get("fingerprint"),
                object_metadata=data.get("object_metadata", {})
            )
            session.version = data.get("version", 1)
            session.status = data.get("status", "INITIALIZED")
            session.created_at = data.get("created_at", "")
            session.updated_at = data.get("updated_at", "")
            session.pre_edit_snapshot_path = data.get("pre_edit_snapshot_path")
            session.decomposition_provenance = data.get("decomposition_provenance", "pending")
            session.vlm_provenance = data.get("vlm_provenance", "pending")
            session.summary = data.get("summary", "")
            session.render_paths = data.get("render_paths", {})
            session.mask_render_path = data.get("mask_render_path")
            session.color_legend = data.get("color_legend", {})
            session.history = data.get("history", [])

            if data.get("current_plan"):
                session.current_plan = ComponentEditPlan(**data["current_plan"])
            if data.get("hierarchy"):
                session.hierarchy = ComponentHierarchy(**data["hierarchy"])

            with self._lock:
                self._sessions[session_id] = session
            return session
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def analyze_session(
        self,
        session_id: str,
        decomposition_method: str = "auto",
        num_clusters: int = 4,
        include_vlm: bool = True,
        user_description: Optional[str] = None
    ) -> EditingSession:
        """Executes full component decomposition, 6-view + mask rendering, and visual semantic grounding."""
        with self.get_session_lock(session_id):
            session = self.get_session(session_id)
            if not session:
                raise KeyError(f"Editing session '{session_id}' not found.")

            working_path = session.working_mesh_path
            if not os.path.exists(working_path):
                raise FileNotFoundError(f"Working mesh does not exist: {working_path}")

            mesh = trimesh.load(working_path, force="mesh")
            session_dir = Path(session.working_dir)

            # 1. Component Decomposition (PartField with fallback)
            submeshes_dir = str(session_dir / "submeshes")
            if decomposition_method == "connected":
                hierarchy = component_decomposer.decompose(mesh, method="connected", output_dir=submeshes_dir)
                session.decomposition_provenance = "connected_components"
            elif decomposition_method == "partfield":
                hierarchy = self._partfield.decompose_mesh(
                    session.working_mesh_path, num_clusters=num_clusters, output_dir=submeshes_dir, allow_fallback=True
                )
                session.decomposition_provenance = (
                    "partfield_neural" if "partfield_neural" in hierarchy.methods_used else "geometric_feature_field"
                )
            else:
                # Auto mode: If mesh has discrete multi-body connected assemblies, separate topologically
                conn = []
                try:
                    conn = mesh.split(only_watertight=False)
                except Exception:
                    pass
                if len(conn) > 1:
                    hierarchy = component_decomposer.decompose(mesh, method="connected", output_dir=submeshes_dir)
                    session.decomposition_provenance = "connected_components"
                else:
                    hierarchy = self._partfield.decompose_mesh(
                        session.working_mesh_path, num_clusters=num_clusters, output_dir=submeshes_dir, allow_fallback=True
                    )
                    session.decomposition_provenance = (
                        "partfield_neural" if "partfield_neural" in hierarchy.methods_used else "geometric_feature_field"
                    )

            # 2. Multi-view and Component Mask Rendering
            render_dir = str(session_dir / "renders")
            renders = vision_language_adapter.generate_multiview_renders(mesh, render_dir, hierarchy)
            session.render_paths = renders.render_paths
            session.mask_render_path = renders.mask_render_path
            session.color_legend = renders.color_legend

            # 3. Visual Understanding: First apply deterministic spatial-visual baseline
            asset_label = user_description or f"3D Model: {session.asset_name}"
            candidate_ids = [
                cid for cid in hierarchy.components.keys()
                if not ("root" in cid.lower() or "assembly" in cid.lower())
            ]
            heuristic_proposal = vision_language_adapter._heuristic_visual_interpret(candidate_ids, hierarchy, asset_label)
            if heuristic_proposal and hasattr(heuristic_proposal, "assignments"):
                for a in heuristic_proposal.assignments:
                    if a.component_id in hierarchy.components:
                        hierarchy.components[a.component_id].semantic_label = a.assigned_semantic_label
                        hierarchy.components[a.component_id].confidence = a.confidence
                        hierarchy.components[a.component_id].is_uncertain = a.is_uncertain
            session.vlm_provenance = "heuristic_baseline"

            # If VLM is installed and requested, refine with multimodal neural vision
            if include_vlm and vision_language_adapter.is_model_installed():
                try:
                    proposal = vision_language_adapter.interpret_components(hierarchy, renders, asset_label)
                    if proposal and hasattr(proposal, "assignments"):
                        for a in proposal.assignments:
                            if a.component_id in hierarchy.components:
                                hierarchy.components[a.component_id].semantic_label = a.assigned_semantic_label
                                hierarchy.components[a.component_id].confidence = a.confidence
                                hierarchy.components[a.component_id].is_uncertain = a.is_uncertain
                        session.vlm_provenance = proposal.inference_source
                except Exception as ve:
                    logger.warning(f"VLM interpretation fallback: {ve}")
                    session.vlm_provenance = f"fallback ({str(ve)})"

            session.semantic_proposal = heuristic_proposal
            session.hierarchy = hierarchy
            session.status = "ANALYZED"
            session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

            # Create human-readable summary
            comp_count = len([c for c in hierarchy.components.values() if "root" not in c.component_id.lower() and "assembly" not in c.component_id.lower()])
            comp_names = [c.semantic_label for c in hierarchy.components.values() if "root" not in c.component_id.lower() and "assembly" not in c.component_id.lower()]
            session.summary = (
                f"Asset '{session.asset_name}' decomposed into {comp_count} components: "
                f"{', '.join(comp_names[:8])}. Decomposition provenance: {session.decomposition_provenance}. "
                f"Visual grounding: {session.vlm_provenance}."
            )

            session.save()
            return session

    def generate_plan(
        self,
        session_id: str,
        user_instruction: str,
        target_component_id: Optional[str] = None
    ) -> ComponentEditPlan:
        """Interprets user instruction via Qwen Coder and produces a structured ComponentEditPlan."""
        with self.get_session_lock(session_id):
            session = self.get_session(session_id)
            if not session:
                raise KeyError(f"Editing session '{session_id}' not found.")
            if not session.hierarchy:
                self.analyze_session(session_id)

            mesh = trimesh.load(session.working_mesh_path, force="mesh")
            geo_rep = geometry_analyzer.analyze_mesh(mesh)

            proposal = getattr(session, "semantic_proposal", None)
            plan = qwen_planner.generate_component_edit_plan(
                user_instruction=user_instruction,
                hierarchy=session.hierarchy,
                geometry_report=geo_rep,
                semantic_proposal=proposal
            )

            # Apply explicit target override if requested by user
            if target_component_id and target_component_id in session.hierarchy.components:
                target_node = session.hierarchy.components[target_component_id]
                plan.target_component_ids = [target_component_id]
                plan.target_semantic_labels = [getattr(target_node, "semantic_label", target_component_id)]
                plan.status = "READY"
                plan.requires_user_confirmation = False
                plan.rationale = f"Explicit user selection of target '{target_component_id}'. " + plan.rationale

                # Update invariants to protect all other non-target components
                plan.invariants = [
                    ComponentInvariant(component_id=cid, preserve_dimensions=True, preserve_position=True, tolerance=0.05)
                    for cid in session.hierarchy.components.keys()
                    if cid != target_component_id and "root" not in cid.lower() and "assembly" not in cid.lower()
                ]

            session.current_plan = plan
            session.status = "PLANNED"
            session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            session.save()
            return plan

    def execute_plan(
        self,
        session_id: str,
        plan_id: Optional[str] = None,
        confirmed_target_id: Optional[str] = None,
        plan_override: Optional[Dict[str, Any]] = None
    ) -> ComponentEditFeedbackReport:
        """Executes the approved plan deterministically, checks invariants, and rolls back if invalid."""
        with self.get_session_lock(session_id):
            session = self.get_session(session_id)
            if not session:
                raise KeyError(f"Editing session '{session_id}' not found.")

            plan = session.current_plan
            if not plan:
                raise RuntimeError("No active editing plan generated for this session.")

            # Plan integrity checks: validate supplied plan ID is provided and matches active approved plan
            if not plan_id:
                raise ValueError("plan_id is required for plan execution. Missing plan ID.")

            if plan.plan_id != plan_id:
                raise ValueError(
                    f"Supplied plan ID '{plan_id}' does not match active approved plan '{plan.plan_id}'. "
                    f"Plan may be expired, stale, or superseded."
                )

            # Validate target component ID exists in active hierarchy
            if plan.target_component_ids and session.hierarchy:
                for tid in plan.target_component_ids:
                    if tid not in session.hierarchy.components:
                        raise ValueError(
                            f"Target component ID '{tid}' does not exist in current session hierarchy. "
                            f"Available components: {list(session.hierarchy.components.keys())}"
                        )

            # Restrict plan overrides to explicitly allowed fields
            ALLOWED_PLAN_OVERRIDES = {"invariants", "scale_factor", "axis", "anchor", "translation_delta", "parameters"}
            if plan_override:
                for k, v in plan_override.items():
                    if k not in ALLOWED_PLAN_OVERRIDES:
                        logger.warning(f"Ignoring unauthorized plan override field: '{k}'")
                        continue
                    if k == "invariants" and isinstance(v, list):
                        plan.invariants = [ComponentInvariant(**item) if isinstance(item, dict) else item for item in v]
                    elif hasattr(plan, k):
                        setattr(plan, k, v)

            # If user explicitly confirmed an ambiguous target
            if confirmed_target_id and session.hierarchy and confirmed_target_id in session.hierarchy.components:
                plan.target_component_ids = [confirmed_target_id]
                node = session.hierarchy.components[confirmed_target_id]
                plan.target_semantic_labels = [getattr(node, "semantic_label", confirmed_target_id)]
                plan.status = "READY"
                plan.requires_user_confirmation = False
                plan.invariants = [
                    ComponentInvariant(component_id=cid, preserve_dimensions=True, preserve_position=True, tolerance=0.05)
                    for cid in session.hierarchy.components.keys()
                    if cid != confirmed_target_id and "root" not in cid.lower() and "assembly" not in cid.lower()
                ]

            if plan.status == "NEEDS_COMPONENT_CONFIRMATION" or plan.requires_user_confirmation or not plan.target_component_ids:
                return ComponentEditFeedbackReport(
                    session_id=session_id,
                    user_instruction=plan.user_request,
                    target_component_id="unresolved",
                    target_semantic_label="ambiguous",
                    operation_executed="none",
                    target_initial_dimensions=(0.0, 0.0, 0.0),
                    target_final_dimensions=(0.0, 0.0, 0.0),
                    target_dimension_delta=(0.0, 0.0, 0.0),
                    target_height_change_percent=0.0,
                    non_target_invariants_preserved=True,
                    invariant_deviation_max_percent=0.0,
                    overall_initial_width=0.0,
                    overall_final_width=0.0,
                    status="NEEDS_COMPONENT_CONFIRMATION",
                    candidate_components=plan.candidate_components,
                    output_mesh_path=session.working_mesh_path,
                    summary=f"Plan requires explicit component target confirmation: {plan.rationale}"
                )

            # 1. Create Pre-Edit Snapshot for Rollback
            snapshot_path = os.path.join(session.working_dir, f"snapshot_v{session.version}.glb")
            shutil.copy2(session.working_mesh_path, snapshot_path)
            session.pre_edit_snapshot_path = snapshot_path

            # 2. Record Pre-Edit Measurements
            mesh_pre = trimesh.load(session.working_mesh_path, force="mesh")
            geo_pre = geometry_analyzer.analyze_mesh(mesh_pre)
            overall_init_w = geo_pre.bounds.dimensions[0]

            target_cid = plan.target_component_ids[0]
            target_node = session.hierarchy.components.get(target_cid)
            target_init_dims = target_node.dimensions if target_node else (1.0, 1.0, 1.0)
            target_label = plan.target_semantic_labels[0] if plan.target_semantic_labels else "component"

            invariant_pre_dims = {}
            invariant_pre_centroids = {}
            for inv in plan.invariants:
                inv_node = session.hierarchy.components.get(inv.component_id)
                if inv_node:
                    invariant_pre_dims[inv.component_id] = inv_node.dimensions
                    if hasattr(inv_node, "face_indices") and inv_node.face_indices:
                        if max(inv_node.face_indices) < len(mesh_pre.faces):
                            try:
                                sub_pre = mesh_pre.submesh([inv_node.face_indices], append=True)
                                invariant_pre_centroids[inv.component_id] = sub_pre.centroid
                            except Exception:
                                pass

            # 3. Execute Deterministic Component Edit
            op_name = plan.operation_type if plan.operation_type != "scale" else "scale_axis"
            exec_params = dict(plan.parameters) if plan.parameters else {}
            if "axis" not in exec_params:
                exec_params["axis"] = plan.axis or "z"
            if "factor" not in exec_params:
                exec_params["factor"] = plan.scale_factor or 1.20
            if "anchor" not in exec_params:
                exec_params["anchor"] = plan.anchor or "bottom"

            edit_args = EditComponentArgs(
                mesh_path=session.working_mesh_path,
                component_id=target_cid,
                operation=op_name,
                parameters=exec_params,
                output_path=session.working_mesh_path,
                hierarchy=session.hierarchy
            )

            try:
                edit_res = GeometryEditingEngine.execute_component_edit(edit_args)
            except Exception as e:
                # Immediate rollback on error
                shutil.copy2(snapshot_path, session.working_mesh_path)
                session.status = "FAILED"
                session.save()
                return ComponentEditFeedbackReport(
                    session_id=session_id,
                    user_instruction=plan.user_request,
                    target_component_id=target_cid,
                    target_semantic_label=target_label,
                    operation_executed=op_name,
                    target_initial_dimensions=target_init_dims,
                    target_final_dimensions=target_init_dims,
                    target_dimension_delta=(0.0, 0.0, 0.0),
                    target_height_change_percent=0.0,
                    non_target_invariants_preserved=True,
                    invariant_deviation_max_percent=0.0,
                    overall_initial_width=overall_init_w,
                    overall_final_width=overall_init_w,
                    status="FAILED",
                    output_mesh_path=session.working_mesh_path,
                    summary=f"Execution error: {str(e)}"
                )

            # 4. Re-measurement of ACTUAL Physical Geometry
            mesh_post = trimesh.load(session.working_mesh_path, force="mesh")
            geo_post = geometry_analyzer.analyze_mesh(mesh_post)
            overall_final_w = geo_post.bounds.dimensions[0]

            post_faces = target_node.face_indices if (target_node and hasattr(target_node, "face_indices") and target_node.face_indices) else []
            if len(post_faces) > 0 and len(post_faces) <= len(mesh_post.faces):
                target_sub = mesh_post.submesh([post_faces], append=True)
                target_final_dims = tuple(float(d) for d in (target_sub.bounds[1] - target_sub.bounds[0]))
            else:
                target_final_dims = tuple(float(d) for d in (edit_res.modified_dimensions or target_init_dims))

            target_delta = (
                round(target_final_dims[0] - target_init_dims[0], 4),
                round(target_final_dims[1] - target_init_dims[1], 4),
                round(target_final_dims[2] - target_init_dims[2], 4)
            )
            height_change_pct = round(((target_final_dims[2] - target_init_dims[2]) / max(1e-4, target_init_dims[2])) * 100.0, 2)

            # 5. Check Non-Target Invariants
            max_deviation_pct = 0.0
            invariants_preserved = True

            for inv in plan.invariants:
                cid = inv.component_id if hasattr(inv, "component_id") else inv.get("component_id")
                tol = inv.tolerance if hasattr(inv, "tolerance") else inv.get("tolerance", 0.05)
                if cid == target_cid:
                    continue
                pre_d = invariant_pre_dims.get(cid)
                inv_node = session.hierarchy.components.get(cid)
                if pre_d and inv_node and hasattr(inv_node, "face_indices") and inv_node.face_indices:
                    if max(inv_node.face_indices) < len(mesh_post.faces):
                        sub_inv = mesh_post.submesh([inv_node.face_indices], append=True)
                        post_d = sub_inv.bounds[1] - sub_inv.bounds[0]
                        for dim_idx in range(3):
                            dev = abs(post_d[dim_idx] - pre_d[dim_idx]) / max(1e-4, pre_d[dim_idx]) * 100.0
                            if dev > max_deviation_pct:
                                max_deviation_pct = dev
                            if dev > (tol * 100.0):
                                invariants_preserved = False

                        # Centroid invariant validation
                        pre_c = invariant_pre_centroids.get(cid)
                        if pre_c is not None:
                            post_c = sub_inv.centroid
                            c_dist = float(np.linalg.norm(post_c - pre_c))
                            ref_rad = max(0.01, float(np.linalg.norm(pre_d)))
                            c_pct = (c_dist / ref_rad) * 100.0
                            if c_pct > max_deviation_pct:
                                max_deviation_pct = c_pct
                            if c_pct > (tol * 100.0):
                                invariants_preserved = False

            if not invariants_preserved or geo_post.quality.has_nan_or_inf:
                # Automatic Rollback
                shutil.copy2(snapshot_path, session.working_mesh_path)
                session.status = "ROLLED_BACK"
                first_tol = 0.05
                if plan.invariants:
                    inv0 = plan.invariants[0]
                    first_tol = inv0.tolerance if hasattr(inv0, "tolerance") else inv0.get("tolerance", 0.05)
                summary = f"Rollback applied: Invariant check failed (max deviation {max_deviation_pct:.2f}% exceeded {first_tol * 100.0}% tolerance)."
                status_val = "ROLLBACK_APPLIED"
            else:
                session.version += 1
                session.status = "EXECUTED"
                status_val = "SUCCESS"
                summary = (
                    f"Successfully modified '{target_label}' ({target_cid}). "
                    f"Height change: {height_change_pct:+.2f}%. "
                    f"All {len(plan.invariants)} non-target component invariants preserved (max deviation: {max_deviation_pct:.2f}%)."
                )

            report = ComponentEditFeedbackReport(
                session_id=session_id,
                user_instruction=plan.user_request,
                target_component_id=target_cid,
                target_semantic_label=target_label,
                operation_executed=f"{op_name}_{exec_params.get('axis', 'z')}_{exec_params.get('factor', 1.0)}",
                target_initial_dimensions=target_init_dims,
                target_final_dimensions=target_final_dims,
                target_dimension_delta=target_delta,
                target_height_change_percent=height_change_pct,
                non_target_invariants_preserved=invariants_preserved,
                invariant_deviation_max_percent=round(max_deviation_pct, 2),
                overall_initial_width=overall_init_w,
                overall_final_width=overall_final_w,
                status=status_val,
                output_mesh_path=session.working_mesh_path,
                summary=summary
            )

            session.history.append({
                "version": session.version,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_instruction": plan.user_request,
                "target": target_cid,
                "target_label": target_label,
                "operation": op_name,
                "params": exec_params,
                "status": status_val,
                "summary": summary,
                "height_change_percent": height_change_pct,
                "invariant_deviation_max_percent": round(max_deviation_pct, 2),
                "invariants_preserved": invariants_preserved
            })

            session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            session.save()
            return report

    def rollback_session(self, session_id: str) -> Tuple[bool, str]:
        """Rolls back the active working mesh to the pre-edit snapshot."""
        with self.get_session_lock(session_id):
            session = self.get_session(session_id)
            if not session:
                return False, f"Session '{session_id}' not found."

            if not session.pre_edit_snapshot_path or not os.path.exists(session.pre_edit_snapshot_path):
                return False, "No pre-edit snapshot available for rollback."

            shutil.copy2(session.pre_edit_snapshot_path, session.working_mesh_path)
            session.status = "ROLLED_BACK"
            session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

            session.history.append({
                "version": session.version,
                "timestamp": session.updated_at,
                "user_instruction": "Manual User Rollback",
                "operation": "rollback",
                "status": "ROLLED_BACK",
                "summary": "Restored pre-edit mesh snapshot."
            })

            session.save()
            return True, "Successfully rolled back mesh to pre-edit state."

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists metadata of all editing sessions."""
        sessions = []
        if not SESSIONS_ROOT.exists():
            return []
        for s_dir in SESSIONS_ROOT.iterdir():
            if s_dir.is_dir() and (s_dir / "session.json").exists():
                sess = self.get_session(s_dir.name)
                if sess:
                    sessions.append({
                        "session_id": sess.session_id,
                        "asset_id": sess.asset_id,
                        "asset_name": sess.asset_name,
                        "version": sess.version,
                        "status": sess.status,
                        "created_at": sess.created_at,
                        "updated_at": sess.updated_at,
                        "summary": sess.summary
                    })
        return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)


editing_session_manager = EditingSessionManager()
