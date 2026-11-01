"""
Strict Pydantic Command Schemas & Allowlist for Qwen 2.5 Coder and Technical Artist Planner.
Protects Blender and downstream reconstruction pipelines from arbitrary code execution
by requiring versioned, validated, strongly-typed JSON command plans.
Provides comprehensive schemas for structural analysis, component hierarchy,
proportions, PBR materials, topology budgets, and Blender operations.
"""

from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
import math
import uuid


class BaseCommand(BaseModel):
    op: str
    target_object: str = Field(default="AI_Generated_Model", description="Name of the object in Blender")

    @field_validator("*", mode="before")
    def check_finite_numbers(cls, v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Numeric values must be finite (not NaN or Inf)")
        return v


class SetMaterialCommand(BaseCommand):
    op: Literal["set_material"] = "set_material"
    preset: Optional[Literal[
        "custom", "metal", "gold", "wood", "plastic", "stone", "glass",
        "leather", "fabric", "steel", "brass", "bronze", "iron", "organic"
    ]] = "custom"
    base_color: Optional[List[float]] = Field(default=[0.8, 0.8, 0.8, 1.0], description="RGBA values [0.0 - 1.0]")
    metallic: float = Field(default=0.0, ge=0.0, le=1.0)
    roughness: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("base_color")
    def validate_rgba(cls, v):
        if v is not None:
            if len(v) != 4 or not all(0.0 <= c <= 1.0 for c in v):
                raise ValueError("base_color must be an RGBA list of 4 floats between 0.0 and 1.0")
        return v


class BevelCommand(BaseCommand):
    op: Literal["bevel"] = "bevel"
    width: float = Field(default=0.01, ge=0.0001, le=1.0, description="Bevel width in meters")
    segments: int = Field(default=3, ge=1, le=16, description="Number of bevel segments")


class SetHeightCommand(BaseCommand):
    op: Literal["set_height"] = "set_height"
    height_meters: float = Field(default=1.8, ge=0.01, le=100.0, description="Target height in meters")


class SetLocationCommand(BaseCommand):
    op: Literal["set_location"] = "set_location"
    x: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    y: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    z: float = Field(default=0.0, ge=-1000.0, le=1000.0)


class SetRotationCommand(BaseCommand):
    op: Literal["set_rotation"] = "set_rotation"
    rx_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    ry_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    rz_deg: float = Field(default=0.0, ge=-360.0, le=360.0)


class SmoothShadingCommand(BaseCommand):
    op: Literal["smooth_shading"] = "smooth_shading"
    enable: bool = True
    angle_degrees: float = Field(default=30.0, ge=0.0, le=180.0)


class AddSubdivisionCommand(BaseCommand):
    op: Literal["add_subdivision"] = "add_subdivision"
    levels: int = Field(default=1, ge=1, le=4, description="Subdivision surface levels (1-4)")


class DecimateCommand(BaseCommand):
    op: Literal["decimate"] = "decimate"
    ratio: float = Field(default=0.5, ge=0.01, le=0.99, description="Target triangle reduction ratio (0.01-0.99)")


class DuplicateCommand(BaseCommand):
    op: Literal["duplicate"] = "duplicate"
    new_name: str = Field(..., min_length=1, max_length=64)
    offset_x: float = Field(default=1.0, ge=-100.0, le=100.0)
    offset_y: float = Field(default=0.0, ge=-100.0, le=100.0)
    offset_z: float = Field(default=0.0, ge=-100.0, le=100.0)


CommandType = Union[
    SetMaterialCommand,
    BevelCommand,
    SetHeightCommand,
    SetLocationCommand,
    SetRotationCommand,
    SmoothShadingCommand,
    AddSubdivisionCommand,
    DecimateCommand,
    DuplicateCommand
]


class CommandPlan(BaseModel):
    version: str = Field(default="1.0", description="Schema version")
    description: str = Field(default="", description="Summary of the changes planned")
    commands: List[CommandType] = Field(default_factory=list, description="Ordered list of safe commands")


# ==============================================================================
# Technical Artist Extended Planning Schemas
# ==============================================================================

class ComponentDefinition(BaseModel):
    name: str = Field(..., description="Unique component name (e.g. blade, crossguard, grip, pommel)")
    role: Literal["primary_structural", "detail", "functional", "accessory", "decorative"] = Field(
        default="primary_structural", description="Functional structural role of this component"
    )
    parent_component: Optional[str] = Field(default=None, description="Parent in component hierarchy")
    relative_scale: Optional[List[float]] = Field(default=None, description="Relative dimensions [x, y, z]")
    relative_position: Optional[List[float]] = Field(default=None, description="Relative offset [x, y, z]")
    material_preset: Optional[str] = Field(default=None, description="Associated material name/preset")


class DimensionConstraints(BaseModel):
    target_height_m: float = Field(default=1.0, ge=0.01, le=500.0, description="Target bounding height in meters")
    target_width_m: Optional[float] = Field(default=None, ge=0.01, le=500.0, description="Target bounding width in meters")
    target_depth_m: Optional[float] = Field(default=None, ge=0.01, le=500.0, description="Target bounding depth in meters")
    proportions: Optional[List[float]] = Field(default=None, description="Normalized proportions [width, depth, height]")
    symmetry_axis: Literal["none", "x", "y", "z", "radial"] = Field(
        default="none", description="Symmetry constraint axis for geometric alignment"
    )


class MaterialSpecification(BaseModel):
    material_id: str = Field(..., description="Unique material identifier")
    target_component: str = Field(default="primary", description="Target component or mesh region")
    preset: Literal[
        "custom", "metal", "gold", "wood", "plastic", "stone", "glass",
        "leather", "fabric", "steel", "brass", "bronze", "iron", "organic"
    ] = Field(default="custom", description="Base PBR preset")
    base_color_rgba: List[float] = Field(default=[0.8, 0.8, 0.8, 1.0], description="Base color RGBA [0.0 - 1.0]")
    roughness: float = Field(default=0.5, ge=0.0, le=1.0, description="Microfacet roughness [0.0 - 1.0]")
    metallic: float = Field(default=0.0, ge=0.0, le=1.0, description="Metallic reflectance [0.0 - 1.0]")
    color_hex: Optional[str] = Field(default=None, description="Hex color code (e.g. #C59A3F)")
    notes: Optional[str] = Field(default=None, description="Specific finish details (e.g. weathered, polished)")

    @field_validator("base_color_rgba")
    def validate_rgba(cls, v):
        if len(v) != 4 or not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError("base_color_rgba must be an RGBA list of 4 floats between 0.0 and 1.0")
        return v


class TextureRequirements(BaseModel):
    requires_pbr_maps: bool = Field(default=True, description="Whether full PBR texture maps are required")
    map_types: List[Literal["diffuse", "normal", "roughness", "metallic", "ao", "displacement"]] = Field(
        default_factory=lambda: ["diffuse", "normal", "roughness"],
        description="Required PBR map channels"
    )
    resolution: Literal[512, 1024, 2048, 4096] = Field(default=1024, description="Target texture resolution")
    uv_unwrapping_strategy: Literal["smart_project", "atlas", "seam_based", "lightmap"] = Field(
        default="smart_project", description="UV layout unwrapping strategy"
    )
    bake_vertex_colors: bool = Field(default=True, description="Whether to bake TripoSR vertex colors into texture maps")


class TopologyBudget(BaseModel):
    target_triangles: int = Field(default=15000, ge=100, le=500000, description="Target polygon triangle budget")
    max_triangles: int = Field(default=30000, ge=500, le=1000000, description="Maximum allowable triangle budget")
    mesh_style: Literal["low_poly", "game_ready", "high_poly_sculpt", "stylized"] = Field(
        default="game_ready", description="Target visual topology style"
    )
    watertight_required: bool = Field(default=True, description="Whether watertight/manifold geometry is mandatory")
    manifold_geometry_required: bool = Field(default=True, description="Reject non-manifold edges or vertices")


class GeometryCorrection(BaseModel):
    decimate_ratio: Optional[float] = Field(default=None, ge=0.01, le=0.99, description="Optional decimation ratio")
    smooth_shading: bool = Field(default=True, description="Enable smooth shading with auto-smooth angle")
    auto_smooth_angle_deg: float = Field(default=30.0, ge=0.0, le=180.0, description="Angle threshold for sharp edges")
    apply_subdivision: bool = Field(default=False, description="Apply Catmull-Clark subdivision")
    subdivision_levels: int = Field(default=1, ge=1, le=3, description="Subdivision levels")
    bevel_edges: bool = Field(default=False, description="Apply edge bevel modifier")
    bevel_width_m: float = Field(default=0.005, ge=0.0001, le=0.1, description="Bevel width in meters")


class ValidationInstruction(BaseModel):
    check_watertight: bool = Field(default=True, description="Verify 0 non-manifold edges")
    check_non_manifold_edges: bool = Field(default=True, description="Verify manifold boundary condition")
    check_uv_overlap: bool = Field(default=True, description="Verify non-overlapping UV chart area")
    min_face_count: int = Field(default=500, ge=10, description="Minimum acceptable faces")
    max_draw_calls: int = Field(default=4, ge=1, le=32, description="Target maximum draw calls/materials")


class ImprovementInstruction(BaseModel):
    aspect: Literal["geometry", "silhouette", "materials", "textures", "scale", "topology"] = Field(
        ..., description="Asset domain requiring enhancement"
    )
    issue_description: str = Field(..., description="Identified weakness or failure in generated asset")
    corrective_action: str = Field(..., description="Concrete step to improve the asset")
    priority: Literal["low", "medium", "high", "critical"] = Field(default="medium")


class TechnicalArtistPlan(BaseModel):
    """
    Comprehensive structured technical-artist production plan generated by Qwen 2.5 Coder.
    Defines structural decomposition, dimensions, material assignments, UV/texture rules,
    topology budgets, validation gates, and safe deterministic Blender operations.
    """
    version: str = Field(default="2.0", description="Plan schema version")
    asset_name: str = Field(..., description="Canonical name of the 3D asset")
    category: str = Field(default="prop", description="Asset category (weapon, furniture, prop, vehicle, etc.)")
    description: str = Field(default="", description="Detailed technical-artist summary")
    components: List[ComponentDefinition] = Field(default_factory=list, description="Sub-part structural breakdown")
    dimensions: DimensionConstraints = Field(default_factory=DimensionConstraints, description="Metric constraints")
    materials: List[MaterialSpecification] = Field(default_factory=list, description="PBR material assignments")
    textures: TextureRequirements = Field(default_factory=TextureRequirements, description="Texture map and UV targets")
    topology: TopologyBudget = Field(default_factory=TopologyBudget, description="Polygon and manifold constraints")
    geometry_correction: GeometryCorrection = Field(default_factory=GeometryCorrection, description="Post-processing ops")
    blender_commands: List[CommandType] = Field(default_factory=list, description="Safe executable Blender commands")
    validation: ValidationInstruction = Field(default_factory=ValidationInstruction, description="Acceptance criteria")
    improvement_notes: List[ImprovementInstruction] = Field(default_factory=list, description="Learned corrections")

    def to_command_plan(self) -> CommandPlan:
        """
        Converts the TechnicalArtistPlan into an executable CommandPlan for Blender.
        Combines explicit blender_commands with generated operations from dimensions and materials.
        """
        cmds: List[CommandType] = list(self.blender_commands)

        # If height is constrained and not in explicit commands, add SetHeightCommand
        has_height_cmd = any(isinstance(c, SetHeightCommand) for c in cmds)
        if not has_height_cmd and self.dimensions.target_height_m:
            cmds.append(SetHeightCommand(
                target_object=self.asset_name,
                height_meters=self.dimensions.target_height_m
            ))

        # If smooth shading is desired, add SmoothShadingCommand
        has_smooth_cmd = any(isinstance(c, SmoothShadingCommand) for c in cmds)
        if not has_smooth_cmd and self.geometry_correction.smooth_shading:
            cmds.append(SmoothShadingCommand(
                target_object=self.asset_name,
                enable=True,
                angle_degrees=self.geometry_correction.auto_smooth_angle_deg
            ))

        # If materials specified and not in explicit commands, add first material
        has_mat_cmd = any(isinstance(c, SetMaterialCommand) for c in cmds)
        if not has_mat_cmd and self.materials:
            primary_mat = self.materials[0]
            cmds.append(SetMaterialCommand(
                target_object=self.asset_name,
                preset=primary_mat.preset,
                base_color=primary_mat.base_color_rgba,
                metallic=primary_mat.metallic,
                roughness=primary_mat.roughness
            ))

        # If decimation specified
        if self.geometry_correction.decimate_ratio:
            has_decimate = any(isinstance(c, DecimateCommand) for c in cmds)
            if not has_decimate:
                cmds.append(DecimateCommand(
                    target_object=self.asset_name,
                    ratio=self.geometry_correction.decimate_ratio
                ))

        return CommandPlan(
            version=self.version,
            description=f"Technical Artist Plan for {self.asset_name}: {self.description}",
            commands=cmds
        )


# ==============================================================================
# Phase 2 Semantic Interpretation & Precision Correction Plan Schemas
# ==============================================================================

class ComponentSemanticAssignment(BaseModel):
    component_id: str = Field(..., description="ID of the candidate component node")
    assigned_semantic_label: str = Field(..., description="Semantic label assigned (e.g. blade, seat, crossguard, leg)")
    material_intent: Optional[str] = Field(default=None, description="Inferred PBR material intent (e.g. wood, metal, leather)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score")
    is_uncertain: bool = Field(default=True, description="True if confidence < 0.70")
    parent_component_id: Optional[str] = Field(default=None, description="Inferred parent assembly ID")
    reasoning: str = Field(default="", description="Geometric rationale for assignment")


class ComponentSemanticProposal(BaseModel):
    version: str = Field(default="2.0")
    asset_id: str = Field(..., description="Asset identifier")
    category: str = Field(default="prop")
    inference_source: str = Field(default="heuristic", description="Provenance: qwen3_vl_multimodal vs heuristic_fallback")
    assignments: List[ComponentSemanticAssignment] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    has_uncertain_components: bool = Field(default=True)
    summary: str = Field(default="")


class PrecisionToolCall(BaseModel):
    tool_name: Literal[
        "analyze_geometry",
        "decompose_components",
        "align_and_ground",
        "calibrate_scale",
        "enforce_symmetry",
        "decimate_mesh",
        "export_submeshes"
    ] = Field(..., description="Allowlisted tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Typed arguments matching tool schema")
    rationale: str = Field(default="", description="Why this tool call is needed")


class PrecisionCorrectionPlan(BaseModel):
    version: str = Field(default="2.0")
    plan_id: str = Field(default_factory=lambda: f"corr_plan_{uuid.uuid4().hex[:6]}")
    asset_name: str
    target_height_m: Optional[float] = None
    target_upright: bool = True
    target_grounded: bool = True
    tool_calls: List[PrecisionToolCall] = Field(default_factory=list, description="Strictly allowlisted tool calls")
    rationale: str = Field(default="", description="Overall engineering strategy")



# ==============================================================================
# Phase 3.1 Dynamic Autonomous Technical-Artist Editing Schemas
# ==============================================================================

class ComponentInvariant(BaseModel):
    component_id: str = Field(..., description="Component ID that must maintain physical dimensions/position")
    preserve_dimensions: bool = Field(default=True, description="Dimensions must stay within tolerance")
    preserve_position: bool = Field(default=True, description="Centroid must stay within tolerance")
    tolerance: float = Field(default=0.05, ge=0.0, le=1.0, description="Allowed deviation ratio (default 5%)")


class ComponentEditPlan(BaseModel):
    version: str = Field(default="3.1")
    plan_id: str = Field(default_factory=lambda: f"edit_plan_{uuid.uuid4().hex[:6]}")
    asset_id: str = Field(..., description="Target asset identifier")
    user_request: str = Field(default="", description="Original user prompt or edit request")
    target_component_ids: List[str] = Field(default_factory=list, description="Candidate IDs to modify")
    target_semantic_labels: List[str] = Field(default_factory=list, description="Target labels (e.g. backrest, blade, leg)")
    operation_type: Literal[
        "scale", "translate", "rotate", "mirror", "solidify", "subdivide",
        "boolean_difference", "boolean_union", "extrude", "bevel", "separate_component", "cleanup"
    ] = Field(default="scale", description="Allowlisted geometry operation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation-specific parameters")
    axis: Optional[Literal["x", "y", "z", "all"]] = Field(default="z", description="Primary transformation axis")
    scale_factor: Optional[float] = Field(default=1.0, ge=0.05, le=10.0, description="Target dimension scale multiplier")
    anchor: Optional[Literal["bottom", "top", "center", "origin"]] = Field(default="bottom", description="Stationary anchor point")
    translation_delta: Optional[List[float]] = Field(default=None, description="Translation vector [dx, dy, dz] in meters")
    invariants: List[ComponentInvariant] = Field(default_factory=list, description="Invariants that must remain preserved")
    risk_level: Literal["low", "medium", "high"] = Field(default="low")
    requires_user_confirmation: bool = Field(default=False, description="True if target component resolution is ambiguous")
    status: Literal["READY", "NEEDS_COMPONENT_CONFIRMATION", "REJECTED"] = Field(
        default="READY", description="Execution gate status"
    )
    candidate_components: List[Dict[str, Any]] = Field(
        default_factory=list, description="Candidate options for user confirmation"
    )
    rationale: str = Field(default="", description="Technical-artist reasoning")
    execution_provenance: Literal["qwen_coder_neural", "auxiliary_classifier_neural", "dynamic_rule_fallback", "failed"] = Field(
        default="dynamic_rule_fallback", description="Provenance of how the edit plan was generated"
    )
