"""
Strict Pydantic Command Schemas & Allowlist for Qwen 2.5 Coder.
Protects Blender from arbitrary code execution by requiring
versioned, validated, strongly-typed JSON command plans.
"""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
import math

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
    preset: Optional[Literal["custom", "metal", "gold", "wood", "plastic", "stone", "glass"]] = "custom"
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
