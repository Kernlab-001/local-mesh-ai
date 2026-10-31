"""
Controlled Tool Registry & Execution Safety Boundary for Local Mesh AI (Phase 2).
Enforces zero arbitrary code execution by requiring all geometric operations to be
registered, strongly typed, and strictly allowlisted.
Qwen Technical-Artist emits structured tool call requests which are validated against
Pydantic schemas before deterministic execution.
"""

import os
import sys
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Type
from pydantic import BaseModel, Field, field_validator
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_analyzer import geometry_analyzer, CompleteGeometryReport
from core.component_decomposer import component_decomposer, ComponentHierarchy
from core.precision_aligner import precision_aligner, AlignmentResult
from core.geometry_construction import GeometryConstructionEngine, ConstructPrimitiveArgs
from core.geometry_editing import GeometryEditingEngine, EditComponentArgs, ComponentEditResult
from core.engines.partfield_adapter import partfield_adapter
from core.engines.vision_language_adapter import vision_language_adapter


# ==============================================================================
# Tool Input Schemas
# ==============================================================================

class AnalyzeGeometryArgs(BaseModel):
    mesh_path: str = Field(..., description="Absolute or relative path to 3D mesh")

    @field_validator("mesh_path")
    def validate_path(cls, v):
        if not v or ".." in v and not os.path.exists(v):
            raise ValueError("Invalid mesh path")
        return v


class DecomposeComponentsArgs(BaseModel):
    mesh_path: str = Field(..., description="Path to mesh to decompose")
    asset_id: Optional[str] = Field(default=None, max_length=64)
    method: str = Field(default="auto", description="Segmentation method: auto, connected, spatial, color")
    min_faces: int = Field(default=100, ge=10, le=10000)
    output_dir: Optional[str] = Field(default=None)


class AlignAndGroundArgs(BaseModel):
    mesh_path: str = Field(..., description="Path to mesh to align and ground")
    output_path: Optional[str] = Field(default=None)
    ground_z: bool = Field(default=True, description="Snap bottom-most vertices to Z=0")
    center_xy: bool = Field(default=True, description="Center mesh around X=0, Y=0")
    normalize_orientation: bool = Field(default=True, description="Rotate primary elongation to +Z")
    target_height_m: Optional[float] = Field(default=None, ge=0.01, le=100.0)
    enforce_symmetry_plane: Optional[str] = Field(default=None)
    is_user_provided_scale: bool = Field(default=True)


class CalibrateScaleArgs(BaseModel):
    mesh_path: str = Field(...)
    target_height_m: float = Field(..., ge=0.01, le=100.0)
    output_path: Optional[str] = Field(default=None)
    is_user_provided: bool = Field(default=True)


class EnforceSymmetryArgs(BaseModel):
    mesh_path: str = Field(...)
    symmetry_plane: str = Field(default="yz_plane", description="Target symmetry plane: yz_plane or xz_plane")
    output_path: Optional[str] = Field(default=None)


class DecimateMeshArgs(BaseModel):
    mesh_path: str = Field(...)
    ratio: Optional[float] = Field(default=None, ge=0.01, le=0.99)
    target_faces: Optional[int] = Field(default=None, ge=100, le=500000)
    output_path: Optional[str] = Field(default=None)


class ExportSubmeshesArgs(BaseModel):
    mesh_path: str = Field(...)
    output_dir: str = Field(...)
    asset_id: Optional[str] = Field(default=None)


class SegmentSpecialist3DArgs(BaseModel):
    mesh_path: str = Field(...)
    num_clusters: int = Field(default=4, ge=2, le=16)
    output_dir: Optional[str] = Field(default=None)


class InterpretVisionLanguageArgs(BaseModel):
    mesh_path: str = Field(...)
    user_asset_description: Optional[str] = Field(default=None)
    render_dir: Optional[str] = Field(default=None)


# ==============================================================================
# Tool Registry & Safe Invoker
# ==============================================================================

class ToolDefinition(BaseModel):
    name: str
    description: str
    args_schema: str  # Class name
    is_destructive: bool = False


class ToolRegistry:
    """
    Central allowlisted tool registry. Disallows arbitrary shell/python execution.
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(
            name="analyze_geometry",
            description="Performs complete metric, topological, and quality analysis of a 3D mesh.",
            args_model=AnalyzeGeometryArgs,
            func=lambda args: geometry_analyzer.analyze_mesh(args.mesh_path).model_dump(),
            is_destructive=False
        )

        self.register_tool(
            name="decompose_components",
            description="Decomposes a mesh into candidate components and constructs a ComponentHierarchy.",
            args_model=DecomposeComponentsArgs,
            func=lambda args: component_decomposer.decompose(
                mesh_or_path=args.mesh_path,
                asset_id=args.asset_id,
                method=args.method,  # type: ignore
                min_face_threshold=args.min_faces,
                output_dir=args.output_dir
            ).model_dump(),
            is_destructive=False
        )

        self.register_tool(
            name="align_and_ground",
            description="Normalizes orientation to +Z, centers XY coordinates, grounds Z to 0, and scales to metric target.",
            args_model=AlignAndGroundArgs,
            func=lambda args: precision_aligner.execute_alignment(
                mesh_or_path=args.mesh_path,
                output_path=args.output_path,
                ground_z=args.ground_z,
                center_xy=args.center_xy,
                normalize_orientation=args.normalize_orientation,
                target_height_m=args.target_height_m,
                enforce_symmetry_plane=args.enforce_symmetry_plane,  # type: ignore
                is_user_provided_scale=args.is_user_provided_scale
            )[1].model_dump(),
            is_destructive=True
        )

        self.register_tool(
            name="calibrate_scale",
            description="Scales mesh uniformly to target metric height in meters.",
            args_model=CalibrateScaleArgs,
            func=lambda args: precision_aligner.execute_alignment(
                mesh_or_path=args.mesh_path,
                output_path=args.output_path,
                ground_z=False,
                center_xy=False,
                normalize_orientation=False,
                target_height_m=args.target_height_m,
                is_user_provided_scale=args.is_user_provided
            )[1].model_dump(),
            is_destructive=True
        )

        self.register_tool(
            name="enforce_symmetry",
            description="Centers bilateral symmetry plane (yz or xz) to coordinate axis 0.0.",
            args_model=EnforceSymmetryArgs,
            func=lambda args: precision_aligner.execute_alignment(
                mesh_or_path=args.mesh_path,
                output_path=args.output_path,
                ground_z=False,
                center_xy=False,
                normalize_orientation=False,
                enforce_symmetry_plane=args.symmetry_plane  # type: ignore
            )[1].model_dump(),
            is_destructive=True
        )

        self.register_tool(
            name="decimate_mesh",
            description="Reduces mesh polygon count using quadric edge collapse while preserving vertex colors.",
            args_model=DecimateMeshArgs,
            func=self._execute_decimation,
            is_destructive=True
        )

        self.register_tool(
            name="export_submeshes",
            description="Exports each candidate component of a hierarchy into isolated GLB submesh files.",
            args_model=ExportSubmeshesArgs,
            func=lambda args: component_decomposer.decompose(
                mesh_or_path=args.mesh_path,
                asset_id=args.asset_id,
                output_dir=args.output_dir
            ).model_dump(),
            is_destructive=False
        )

        self.register_tool(
            name="construct_primitive",
            description="Constructs a parametric 3D primitive (cube, uv_sphere, icosphere, cylinder, cone, torus, plane, circle, custom_profile).",
            args_model=ConstructPrimitiveArgs,
            func=self._execute_primitive_construction,
            is_destructive=False
        )

        self.register_tool(
            name="edit_component",
            description="Executes a parameterized transformation or geometric edit targeting a specific component in a mesh.",
            args_model=EditComponentArgs,
            func=lambda args: GeometryEditingEngine.execute_component_edit(args).model_dump(),
            is_destructive=True
        )

        self.register_tool(
            name="segment_specialist_3d",
            description="Decomposes a mesh using specialist PartField feature-field clustering.",
            args_model=SegmentSpecialist3DArgs,
            func=lambda args: partfield_adapter.decompose_mesh(
                mesh_or_path=args.mesh_path,
                num_clusters=args.num_clusters,
                output_dir=args.output_dir
            ).model_dump(),
            is_destructive=False
        )

        self.register_tool(
            name="interpret_vision_language",
            description="Interprets multi-view visual evidence and assigns semantic labels to candidate components.",
            args_model=InterpretVisionLanguageArgs,
            func=self._execute_vision_interpretation,
            is_destructive=False
        )

    def register_tool(
        self,
        name: str,
        description: str,
        args_model: Type[BaseModel],
        func: Callable[[Any], Any],
        is_destructive: bool = False
    ):
        self._tools[name] = {
            "name": name,
            "description": description,
            "args_model": args_model,
            "func": func,
            "is_destructive": is_destructive
        }

    def list_tools(self) -> List[ToolDefinition]:
        """Returns metadata for all allowlisted tools."""
        return [
            ToolDefinition(
                name=t["name"],
                description=t["description"],
                args_schema=t["args_model"].__name__,
                is_destructive=t["is_destructive"]
            )
            for t in self._tools.values()
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates arguments with Pydantic model and executes allowlisted tool safely.
        Raises ValueError or ValidationError if invalid.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown or unauthorized tool '{tool_name}'. Allowlisted tools: {list(self._tools.keys())}")

        tool_entry = self._tools[tool_name]
        args_model = tool_entry["args_model"]

        # Validate arguments against strict schema
        validated_args = args_model.model_validate(arguments)

        # Execute safe function
        result = tool_entry["func"](validated_args)
        return {
            "tool": tool_name,
            "status": "success",
            "result": result
        }

    def _execute_decimation(self, args: DecimateMeshArgs) -> Dict[str, Any]:
        import trimesh
        mesh = trimesh.load(args.mesh_path, force="mesh", process=False)
        curr_faces = len(mesh.faces)

        target_faces = args.target_faces
        if target_faces is None and args.ratio:
            target_faces = max(100, int(curr_faces * args.ratio))
        elif target_faces is None:
            target_faces = max(100, curr_faces // 2)

        if target_faces < curr_faces:
            orig_verts = np.asarray(mesh.vertices).copy()
            orig_colors = None
            if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                orig_colors = np.asarray(mesh.visual.vertex_colors).copy()

            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

            # Preserve authentic vertex colors via KDTree
            if orig_colors is not None and len(orig_colors) == len(orig_verts):
                from scipy.spatial import cKDTree
                tree = cKDTree(orig_verts)
                _, indices = tree.query(mesh.vertices)
                mesh.visual.vertex_colors = orig_colors[indices]

        out_path = args.output_path or args.mesh_path
        mesh.export(out_path)
        report = geometry_analyzer.analyze_mesh(mesh)
        return report.model_dump()

    def _execute_primitive_construction(self, args: ConstructPrimitiveArgs) -> Dict[str, Any]:
        mesh = GeometryConstructionEngine.construct_from_args(args)
        report = geometry_analyzer.analyze_mesh(mesh)
        return {
            "primitive_type": args.primitive_type,
            "output_path": args.output_path,
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "dimensions": report.bounds.dimensions,
            "is_watertight": report.topology.is_watertight
        }

    def _execute_vision_interpretation(self, args: InterpretVisionLanguageArgs) -> Dict[str, Any]:
        import trimesh
        mesh = trimesh.load(args.mesh_path, force="mesh")
        h = component_decomposer.decompose(mesh)
        rdir = args.render_dir or os.path.join(os.path.dirname(os.path.abspath(args.mesh_path)), "renders")
        renders = vision_language_adapter.generate_multiview_renders(mesh, rdir, h)
        proposal = vision_language_adapter.interpret_components(h, renders, args.user_asset_description)
        return proposal.model_dump()


# Global singleton instance
tool_registry = ToolRegistry()
