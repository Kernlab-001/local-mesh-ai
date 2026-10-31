"""
Tests for Command Schemas, Operation Allowlist & Malicious Input Rejection.
"""

import pytest
from pydantic import ValidationError
from core.command_schemas import (
    CommandPlan,
    SetMaterialCommand,
    BevelCommand,
    SetHeightCommand,
    DecimateCommand
)

def test_valid_command_plan():
    raw = {
        "version": "1.0",
        "description": "Make gold chair",
        "commands": [
            {"op": "set_material", "target_object": "Chair", "preset": "gold", "metallic": 1.0, "roughness": 0.2},
            {"op": "bevel", "target_object": "Chair", "width": 0.02, "segments": 3},
            {"op": "set_height", "target_object": "Chair", "height_meters": 1.2}
        ]
    }
    plan = CommandPlan.model_validate(raw)
    assert len(plan.commands) == 3
    assert plan.commands[0].op == "set_material"
    assert plan.commands[0].preset == "gold"

def test_reject_unknown_operation():
    raw = {
        "version": "1.0",
        "commands": [
            {"op": "delete_all_files", "target_object": "System"}
        ]
    }
    with pytest.raises(ValidationError):
        CommandPlan.model_validate(raw)

def test_reject_malicious_code_payload():
    raw = {
        "version": "1.0",
        "commands": [
            {"op": "exec", "code": "import os; os.system('rmdir /s /q C:\\')"}
        ]
    }
    with pytest.raises(ValidationError):
        CommandPlan.model_validate(raw)

def test_reject_out_of_bounds_numbers():
    # Negative bevel width
    with pytest.raises(ValidationError):
        BevelCommand(target_object="Box", width=-0.5)

    # Decimate ratio > 1.0
    with pytest.raises(ValidationError):
        DecimateCommand(target_object="Box", ratio=2.5)

    # Target height <= 0
    with pytest.raises(ValidationError):
        SetHeightCommand(target_object="Box", height_meters=-1.0)

def test_technical_artist_plan_validation():
    from core.command_schemas import (
        TechnicalArtistPlan,
        ComponentDefinition,
        DimensionConstraints,
        MaterialSpecification,
        TextureRequirements,
        TopologyBudget,
        GeometryCorrection,
        ValidationInstruction,
        ImprovementInstruction,
        SetMaterialCommand,
        BevelCommand
    )

    plan_dict = {
        "version": "2.0",
        "asset_name": "SteelBroadsword",
        "category": "weapon",
        "description": "Double-edged steel broadsword with brass crossguard and brown leather grip",
        "components": [
            {"name": "blade", "role": "primary_structural", "material_preset": "steel"},
            {"name": "crossguard", "role": "functional", "parent_component": "blade", "material_preset": "brass"},
            {"name": "grip", "role": "functional", "parent_component": "crossguard", "material_preset": "leather"},
            {"name": "pommel", "role": "accessory", "parent_component": "grip", "material_preset": "brass"}
        ],
        "dimensions": {
            "target_height_m": 1.15,
            "target_width_m": 0.25,
            "target_depth_m": 0.05,
            "proportions": [0.22, 0.04, 1.0],
            "symmetry_axis": "y"
        },
        "materials": [
            {
                "material_id": "mat_steel_blade",
                "target_component": "blade",
                "preset": "steel",
                "base_color_rgba": [0.85, 0.85, 0.88, 1.0],
                "metallic": 0.95,
                "roughness": 0.2
            },
            {
                "material_id": "mat_brass_guard",
                "target_component": "crossguard",
                "preset": "brass",
                "base_color_rgba": [0.85, 0.65, 0.25, 1.0],
                "metallic": 0.85,
                "roughness": 0.35
            },
            {
                "material_id": "mat_leather_grip",
                "target_component": "grip",
                "preset": "leather",
                "base_color_rgba": [0.35, 0.2, 0.1, 1.0],
                "metallic": 0.0,
                "roughness": 0.8
            }
        ],
        "textures": {
            "requires_pbr_maps": True,
            "map_types": ["diffuse", "normal", "roughness", "metallic"],
            "resolution": 2048,
            "uv_unwrapping_strategy": "smart_project",
            "bake_vertex_colors": True
        },
        "topology": {
            "target_triangles": 12000,
            "max_triangles": 20000,
            "mesh_style": "game_ready",
            "watertight_required": True,
            "manifold_geometry_required": True
        },
        "geometry_correction": {
            "decimate_ratio": 0.5,
            "smooth_shading": True,
            "auto_smooth_angle_deg": 35.0,
            "bevel_edges": True,
            "bevel_width_m": 0.002
        },
        "blender_commands": [
            {"op": "bevel", "target_object": "SteelBroadsword", "width": 0.002, "segments": 2}
        ],
        "validation": {
            "check_watertight": True,
            "check_non_manifold_edges": True,
            "min_face_count": 1000
        },
        "improvement_notes": [
            {
                "aspect": "silhouette",
                "issue_description": "Crossguard should extend perpendicularly from blade axis",
                "corrective_action": "Align symmetry across y axis with 0.25m span",
                "priority": "high"
            }
        ]
    }

    plan = TechnicalArtistPlan.model_validate(plan_dict)
    assert plan.asset_name == "SteelBroadsword"
    assert len(plan.components) == 4
    assert len(plan.materials) == 3
    assert plan.topology.mesh_style == "game_ready"
    assert plan.dimensions.target_height_m == 1.15

    # Test conversion to Blender CommandPlan
    cmd_plan = plan.to_command_plan()
    assert len(cmd_plan.commands) >= 3
    # Check that SetHeightCommand was injected from dimensions
    ops = [c.op for c in cmd_plan.commands]
    assert "set_height" in ops
    assert "smooth_shading" in ops
    assert "set_material" in ops

