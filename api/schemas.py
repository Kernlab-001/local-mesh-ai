"""
Pydantic Schemas for Local Neural 3D Studio API.
Includes strict boundary validation, length constraints, and pattern safety.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class CreateJobRequest(BaseModel):
    image_path: Optional[str] = Field(default=None, max_length=512, description="Absolute path to 2D concept image")
    prompt: Optional[str] = Field(default=None, max_length=1000, description="Text prompt to generate 3D asset from scratch")
    target_height: float = Field(default=1.8, ge=0.05, le=50.0)
    quality: str = Field(default="balanced", description="draft | balanced | high")
    preset: str = Field(default="prop", description="prop | character | weapon | furniture")
    remove_bg: bool = True
    sanitize: bool = True
    sharp_overlay: bool = True
    weld: bool = True
    pbr_material: bool = True
    accurate_colors: bool = Field(default=True, description="Aligns 3D mesh colors and vibrancy 1:1 with reference artwork")
    generate_lods: bool = False
    auto_rig: bool = False
    render_turntable: bool = False
    retopology_and_bake: bool = Field(default=False, description="Executes automated quad retopology decimation and bakes Normal/Diffuse maps")
    target_faces: int = Field(default=8000, ge=1000, le=100000, description="Target face count for retopology")
    target_blend: Optional[str] = Field(default=None, max_length=512)
    engine: Optional[str] = Field(default="auto", description="3D Engine: auto | triposr | shap_e | hybrid_sd_triposr | sf3d | trellis")
    strategy: Optional[str] = Field(default="hybrid_concept_art", description="Text-to-3D Strategy: hybrid_concept_art | direct_text_to_3d")
    max_poly_budget: Optional[int] = Field(default=150000, description="Maximum polygon triangle budget for geometry validator")
    route_override: Optional[str] = Field(default=None, description="Requested generation route override")
    route: Optional[str] = Field(default=None, description="Alias for route_override")
    hardware_profile: Optional[str] = Field(default="balanced", description="GPU execution profile: low_memory | balanced | quality")

class JobResponse(BaseModel):
    job_id: str
    input_mode: str = "image"
    status: str
    stage: str
    progress_pct: int
    elapsed_seconds: float
    output_mesh: Optional[str] = None
    high_poly_mesh: Optional[str] = None
    low_poly_mesh: Optional[str] = None
    output_blend: Optional[str] = None
    preview_png: Optional[str] = None
    turntable_mp4: Optional[str] = None
    baked_normal: Optional[str] = None
    baked_diffuse: Optional[str] = None
    game_ready_glb: Optional[str] = None
    mesh_url: Optional[str] = None
    preview_url: Optional[str] = None
    blend_url: Optional[str] = None
    concept_url: Optional[str] = None
    multiview_url: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = []

class EditJobRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=1000, description="Natural language modification prompt for Qwen")

class ExportJobRequest(BaseModel):
    format: str = Field(..., description="glb | fbx | obj")
    output_name: str = Field(
        default="model_export",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Safe filename identifier (alphanumeric, underscore, dash only)"
    )

# ==============================================================================
# Phase 2 Geometry & Component Analysis API Schemas
# ==============================================================================

class AnalyzeGeometryRequest(BaseModel):
    mesh_path: str = Field(..., description="Path to mesh to analyze")

class DecomposeComponentsRequest(BaseModel):
    mesh_path: str = Field(...)
    method: str = Field(default="auto", description="auto | connected | spatial | color")
    asset_id: Optional[str] = Field(default=None)
    min_faces: int = Field(default=100, ge=10, le=10000)
    output_dir: Optional[str] = Field(default=None)

class AlignAndGroundRequest(BaseModel):
    mesh_path: str = Field(...)
    output_path: Optional[str] = Field(default=None)
    ground_z: bool = Field(default=True)
    center_xy: bool = Field(default=True)
    normalize_orientation: bool = Field(default=True)
    target_height_m: Optional[float] = Field(default=None, ge=0.01, le=100.0)
    enforce_symmetry_plane: Optional[str] = Field(default=None)

class BoundedCorrectionRequest(BaseModel):
    mesh_path: str = Field(...)
    target_height_m: Optional[float] = Field(default=None, ge=0.01, le=100.0)
    target_upright: bool = Field(default=True)
    target_grounded: bool = Field(default=True)
    max_attempts: int = Field(default=3, ge=1, le=5)

class ExecuteToolRequest(BaseModel):
    tool_name: str = Field(..., description="Allowlisted tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict)



# ==============================================================================
# Phase 4 Unified Generation Pipeline Schemas
# ==============================================================================

class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=1000, description="Text description of the 3D asset to generate")
    route: Optional[str] = Field(default=None, description="Optional route override: text_to_3d_hybrid | text_to_3d_direct")
    strategy: Optional[str] = Field(default="hybrid_concept_art", description="Text-to-3D strategy: hybrid_concept_art | direct_text_to_3d")
    hardware_profile: str = Field(default="balanced", description="low_memory | balanced | quality")
    preset: str = Field(default="prop", description="prop | character | weapon | furniture")
    target_height: Optional[float] = Field(default=None, ge=0.05, le=50.0)
    accurate_colors: bool = Field(default=True, description="Aligns 3D mesh colors with reference artwork")
    render_turntable: bool = Field(default=False)

class GenerateImageRequest(BaseModel):
    image_path: str = Field(..., min_length=1, max_length=512, description="Local path to source 2D reference image")
    hardware_profile: str = Field(default="balanced", description="low_memory | balanced | quality")
    preset: str = Field(default="prop", description="prop | character | weapon | furniture")
    target_height: Optional[float] = Field(default=None, ge=0.05, le=50.0)
    accurate_colors: bool = Field(default=True, description="Aligns 3D mesh colors with reference artwork")
    render_turntable: bool = Field(default=False)

class JobProgressResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress_pct: int
    elapsed_seconds: float
    error: Optional[str] = None
    logs: List[str] = []

class RouteInfoResponse(BaseModel):
    route_id: str
    name: str
    modality: str
    status: str
    ready: bool
    description: str
    required_models: List[str]
    estimated_vram_gb: float
    notes: str

class AssetResponse(BaseModel):
    asset_id: str
    job_id: str
    created_at: str
    modality: str
    route: str
    hardware_profile: str
    execution_provenance: str
    mesh_path: Optional[str] = None
    preview_path: Optional[str] = None
    file_size_bytes: int = 0
    timings: Dict[str, float] = Field(default_factory=dict)


# ==============================================================================
# Phase 5 Blender Studio Integration Schemas
# ==============================================================================

class StageImageResponse(BaseModel):
    staged_path: str
    filename: str
    width: int
    height: int
    format: str
    file_size_bytes: int

class SceneObjectMetadata(BaseModel):
    name: str
    type: str = "MESH"
    vertex_count: int = 0
    face_count: int = 0
    dimensions_m: List[float] = Field(default_factory=list)
    bounding_box: Dict[str, Any] = Field(default_factory=dict)
    material_slots: List[str] = Field(default_factory=list)
    local_mesh_ai_asset_id: Optional[str] = None
    local_mesh_ai_job_id: Optional[str] = None

class SceneSyncRequest(BaseModel):
    scene_id: Optional[str] = "default_scene"
    scene_name: Optional[str] = None
    blender_version: Optional[str] = "4.2.0"
    units: Optional[str] = "METRIC"
    objects: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    active_object: Optional[Dict[str, Any]] = None
    selected_objects: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    scene_objects_summary: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    camera_info: Optional[Dict[str, Any]] = None

class SceneSyncResponse(BaseModel):
    status: str = "synchronized"
    timestamp: str
    scene_name: str = "default_scene"
    synced_objects: int = 0
    objects_synced_count: int = 0
    scene_version: int = 1
    message: str = "Scene synchronized"


# ==============================================================================
# Phase 6 Advanced AI Technical Artist & Editing Session Schemas
# ==============================================================================

class ComponentSummary(BaseModel):
    component_id: str
    semantic_label: str = "candidate_component"
    confidence: float = 0.5
    is_uncertain: bool = True
    dimensions: List[float] = Field(default_factory=list)
    centroid: List[float] = Field(default_factory=list)
    face_count: int = 0
    vertex_count: int = 0
    relative_z_position: float = 0.5
    color_hex: Optional[str] = None
    submesh_url: Optional[str] = None


class StageAssetRequest(BaseModel):
    asset_name: str = Field(..., min_length=1, max_length=128, description="Blender active object name")
    mesh_path: Optional[str] = Field(default=None, description="Local path to exported mesh (.glb/.obj)")
    mesh_base64: Optional[str] = Field(default=None, description="Base64 encoded mesh file bytes")
    file_format: str = Field(default="glb", description="glb | obj")
    object_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Transforms, slots, bounding dimensions from Blender")
    fingerprint: Optional[str] = Field(default=None, description="Unique asset hash or mesh fingerprint")


class StageAssetResponse(BaseModel):
    session_id: str
    asset_id: str
    asset_name: str
    mesh_path: str
    mesh_url: str
    vertex_count: int
    face_count: int
    dimensions: List[float]
    created_at: str
    message: str = "Asset staged successfully"


class AnalyzeAssetRequest(BaseModel):
    session_id: str = Field(..., description="Active editing session ID")
    decomposition_method: str = Field(default="auto", description="auto | partfield | geometric | connected")
    num_clusters: int = Field(default=4, ge=2, le=16)
    include_vlm: bool = Field(default=True, description="Run Qwen3-VL visual semantic grounding")
    user_description: Optional[str] = Field(default=None, description="Optional description of asset from user")


class AnalyzeAssetResponse(BaseModel):
    session_id: str
    asset_id: str
    asset_name: str
    total_components: int
    components: List[ComponentSummary]
    decomposition_provenance: str
    vlm_provenance: str
    render_paths: Dict[str, str] = Field(default_factory=dict)
    render_urls: Dict[str, str] = Field(default_factory=dict)
    mask_render_url: Optional[str] = None
    color_legend: Dict[str, str] = Field(default_factory=dict)
    summary: str


class GeneratePlanRequest(BaseModel):
    session_id: str = Field(..., description="Active editing session ID")
    user_instruction: str = Field(..., min_length=2, max_length=1000, description="Natural language technical-artist instruction")
    target_component_id: Optional[str] = Field(default=None, description="Optional explicit target component ID")


class GeneratePlanResponse(BaseModel):
    session_id: str
    plan_id: str
    plan: Dict[str, Any]
    status: str
    requires_user_confirmation: bool
    rationale: str


class ExecutePlanRequest(BaseModel):
    session_id: str = Field(..., description="Active editing session ID")
    plan_id: Optional[str] = Field(default=None, description="Plan ID to execute")
    confirmed_target_id: Optional[str] = Field(default=None, description="Explicit user confirmation for ambiguous target")
    plan_override: Optional[Dict[str, Any]] = Field(default=None, description="Optional user-reviewed plan parameters")


class ExecutePlanResponse(BaseModel):
    session_id: str
    status: str
    report: Dict[str, Any]
    output_mesh_url: str
    message: str


class RollbackRequest(BaseModel):
    session_id: str = Field(..., description="Active editing session ID")


class RollbackResponse(BaseModel):
    session_id: str
    status: str
    message: str
    output_mesh_url: str


class SessionStateResponse(BaseModel):
    session_id: str
    asset_id: str
    asset_name: str
    version: int
    status: str
    created_at: str
    updated_at: str
    total_components: int
    components: List[ComponentSummary]
    history_count: int
    has_pre_edit_snapshot: bool
    mesh_url: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list) 
