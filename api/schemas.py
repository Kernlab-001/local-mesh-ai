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
