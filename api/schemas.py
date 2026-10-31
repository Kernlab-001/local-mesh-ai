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

