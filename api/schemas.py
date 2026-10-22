"""
Pydantic Schemas for Local Neural 3D Studio API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class CreateJobRequest(BaseModel):
    image_path: Optional[str] = Field(default=None, description="Absolute path to 2D concept image")
    prompt: Optional[str] = Field(default=None, description="Text prompt to generate 3D asset from scratch")
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
    target_blend: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress_pct: int
    elapsed_seconds: float
    output_mesh: Optional[str] = None
    output_blend: Optional[str] = None
    preview_png: Optional[str] = None
    turntable_mp4: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = []

class EditJobRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=1000, description="Natural language modification prompt for Qwen")

class ExportJobRequest(BaseModel):
    format: str = Field(..., description="glb | fbx | obj")
    output_name: Optional[str] = "model_export"
