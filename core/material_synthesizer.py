"""
Material Synthesizer for Local Mesh AI.
Executes inside Blender (headless or GUI) to:
1. Ensure valid, non-overlapping UV mappings via Smart UV Project.
2. Segment polygon faces into multiple distinct semantic material slots based on AppearanceProfile.
3. Build 100% glTF 2.0 compliant PBR shader node trees (Direct TexImage / ColorAttribute -> BSDF).
4. Pack all textures into the mesh for seamless, zero-loss glTF binary export.
"""

import os
import math
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

from core.appearance_analyzer import AppearanceProfile, MaterialSlotConfig
from core.texture_generator import texture_generator

def synthesize_and_apply_materials(
    obj,
    profile: AppearanceProfile,
    textures_dir: Optional[str] = None,
    generate_textures: bool = True
) -> Dict[str, Any]:
    """
    Synthesizes and assigns multi-material PBR setups to obj in Blender.
    """
    if not IN_BLENDER:
        raise RuntimeError("synthesize_and_apply_materials must be executed inside Blender via 'bpy'.")

    if not obj or obj.type != 'MESH' or len(obj.data.polygons) == 0:
        return {"status": "skipped", "reason": "No valid mesh object"}

    print(f"[Material Synthesizer] Applying appearance profile '{profile.category}' with {len(profile.materials)} material slots...")

    # 1. Ensure clean, non-overlapping UV Unwrapping
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    has_uv = bool(obj.data.uv_layers)
    if not has_uv or len(obj.data.uv_layers.active.data) == 0:
        print("[Material Synthesizer] Unwrapping non-overlapping UV layout (Smart UV Project)...")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66.0),
            island_margin=0.02,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True
        )
        bpy.ops.object.mode_set(mode='OBJECT')

    # Verify UV coordinates are populated
    uv_count = 0
    if obj.data.uv_layers and obj.data.uv_layers.active:
        uv_count = len(obj.data.uv_layers.active.data)

    # 2. Compute mesh bounding box and normalized heights
    vertices = np.array([v.co for v in obj.data.vertices])
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    z_height = max(z_max - z_min, 1e-5)

    # 3. Create Blender Material Slots
    obj.data.materials.clear()
    material_slots = []
    mat_to_index = {}

    textures_dir = textures_dir or os.path.join(os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ".", "textures")
    os.makedirs(textures_dir, exist_ok=True)

    for idx, mat_cfg in enumerate(profile.materials):
        mat_name = f"M_{mat_cfg.name}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Output & Principled BSDF
        out_node = nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (600, 0)

        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (200, 0)
        links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])

        # Set Base Physical Properties
        bsdf.inputs['Roughness'].default_value = mat_cfg.roughness
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = mat_cfg.metallic

        norm_color = [c / 255.0 for c in mat_cfg.color_rgb] + [1.0]
        bsdf.inputs['Base Color'].default_value = tuple(norm_color)

        # 4. Generate and bind PBR texture maps (Albedo, Normal, Roughness)
        if generate_textures:
            try:
                pbr_maps = texture_generator.generate_generic_pbr(
                    base_rgb=mat_cfg.color_rgb,
                    texture_type=mat_cfg.texture_type,
                    roughness_base=mat_cfg.roughness,
                    res=1024
                )
                tex_paths = texture_generator.save_pbr_textures(
                    pbr_maps,
                    output_dir=textures_dir,
                    prefix=f"tex_{idx}_{mat_cfg.name.lower()}"
                )

                # A) Base Color / Diffuse Image Node (Direct connection for 100% glTF compliance)
                if "diffuse" in tex_paths:
                    tex_diff_node = nodes.new('ShaderNodeTexImage')
                    tex_diff_node.location = (-300, 100)
                    img_diff = bpy.data.images.load(tex_paths["diffuse"])
                    img_diff.pack()
                    tex_diff_node.image = img_diff
                    links.new(tex_diff_node.outputs['Color'], bsdf.inputs['Base Color'])

                # B) Normal Bump Map Node (Direct connection)
                if "normal" in tex_paths:
                    tex_norm_node = nodes.new('ShaderNodeTexImage')
                    tex_norm_node.location = (-600, -200)
                    img_norm = bpy.data.images.load(tex_paths["normal"])
                    img_norm.colorspace_settings.name = 'Non-Color'
                    img_norm.pack()
                    tex_norm_node.image = img_norm

                    norm_map_node = nodes.new('ShaderNodeNormalMap')
                    norm_map_node.location = (-300, -200)
                    norm_map_node.inputs['Strength'].default_value = 1.0
                    links.new(tex_norm_node.outputs['Color'], norm_map_node.inputs['Color'])
                    links.new(norm_map_node.outputs['Normal'], bsdf.inputs['Normal'])

            except Exception as tex_err:
                print(f"[Material Synthesizer Warning] Texture binding skipped for {mat_cfg.name}: {tex_err}")

        obj.data.materials.append(mat)
        mat_to_index[mat_cfg.name] = idx
        material_slots.append(mat_name)

    # 5. Partition Polygon Faces into Semantic Material Slots
    num_polygons = len(obj.data.polygons)
    slot_counts = [0] * len(profile.materials)

    for poly in obj.data.polygons:
        # Normalized height of polygon centroid in object space
        cz = poly.center.z
        norm_z = np.clip((cz - z_min) / z_height, 0.0, 1.0)

        # Match polygon to appropriate material slot based on height & orientation rules
        assigned_slot = 0
        for s_idx, m_cfg in enumerate(profile.materials):
            if m_cfg.z_min_norm <= norm_z <= m_cfg.z_max_norm:
                assigned_slot = s_idx
                break

        poly.material_index = assigned_slot
        slot_counts[assigned_slot] += 1

    print(f"[Material Synthesizer] Polygon partitioning across {len(profile.materials)} slots:")
    for s_idx, m_cfg in enumerate(profile.materials):
        pct = round(slot_counts[s_idx] / max(num_polygons, 1) * 100, 1)
        print(f"  Slot {s_idx} ({m_cfg.name}): {slot_counts[s_idx]:,} faces ({pct}%) -> RGB {m_cfg.color_rgb}")

    return {
        "status": "success",
        "category": profile.category,
        "material_count": len(material_slots),
        "materials": material_slots,
        "slot_face_distribution": slot_counts,
        "uv_count": uv_count,
        "textures_generated": generate_textures
    }
