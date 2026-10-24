"""
Headless Blender 5.2 LTS Asset Inspection and Rendering Validator.
Invoked via:
  blender --background --python tests/blender_verify_asset.py -- --mesh <path_to_glb> --render <path_to_png> --report <path_to_json>
"""

import os
import sys
import json
import argparse
import math

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None


def inspect_and_render(mesh_path: str, render_path: str, report_path: str):
    if bpy is None:
        print("[Blender Validator Error] bpy is not available outside Blender.")
        sys.exit(1)

    # 1. Reset scene to pristine state
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create scene collection if missing
    scene = bpy.context.scene
    available_engines = [e.identifier for e in scene.render.bl_rna.properties['engine'].enum_items]
    if 'BLENDER_EEVEE' in available_engines:
        scene.render.engine = 'BLENDER_EEVEE'
    elif 'BLENDER_EEVEE_NEXT' in available_engines:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    else:
        scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'

    # 2. Import GLB mesh
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found at: {mesh_path}")

    bpy.ops.import_scene.gltf(filepath=mesh_path)

    # Find imported mesh objects
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objs:
        raise ValueError("No mesh objects found after GLB import.")

    primary_obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = primary_obj
    primary_obj.select_set(True)

    # Gather mesh statistics
    mesh_data = primary_obj.data
    vert_count = len(mesh_data.vertices)
    poly_count = len(mesh_data.polygons)
    dims = [round(float(d), 4) for d in primary_obj.dimensions]

    # Gather material slots and principled BSDF properties
    mats_info = []
    for slot in primary_obj.material_slots:
        if slot.material:
            mat = slot.material
            mat_entry = {
                "name": mat.name,
                "use_nodes": mat.use_nodes,
                "has_texture": False,
                "roughness": None,
                "metallic": None
            }
            if mat.use_nodes and mat.node_tree:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        try:
                            if 'Roughness' in node.inputs:
                                mat_entry["roughness"] = round(float(node.inputs['Roughness'].default_value), 3)
                            if 'Metallic' in node.inputs:
                                mat_entry["metallic"] = round(float(node.inputs['Metallic'].default_value), 3)
                        except Exception:
                            pass
                    elif node.type == 'TEX_IMAGE' and node.image:
                        mat_entry["has_texture"] = True
            mats_info.append(mat_entry)

    # Ensure vertex color attributes are connected to Principled BSDF Base Color for accurate render
    has_color_attr = hasattr(mesh_data, "color_attributes") and len(mesh_data.color_attributes) > 0
    if has_color_attr:
        attr_name = mesh_data.color_attributes[0].name
        if not primary_obj.material_slots or not primary_obj.material_slots[0].material:
            v_mat = bpy.data.materials.new(name="M_VertexColors")
            v_mat.use_nodes = True
            primary_obj.data.materials.append(v_mat)
        else:
            v_mat = primary_obj.material_slots[0].material
            v_mat.use_nodes = True

        if v_mat.node_tree:
            nodes = v_mat.node_tree.nodes
            links = v_mat.node_tree.links
            bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf and not any(l.to_socket == bsdf.inputs['Base Color'] for l in links):
                attr_node = nodes.new(type="ShaderNodeAttribute")
                attr_node.attribute_name = attr_name
                attr_node.attribute_type = 'GEOMETRY'
                links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])

    # 3. Setup Camera & 3-Point Studio Lighting
    # Center object origin
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    primary_obj.location = (0, 0, dims[2] / 2.0)

    # Calculate bounding radius
    bbox_max = max(dims)
    cam_dist = max(bbox_max * 2.2, 1.2)

    # Camera
    cam_data = bpy.data.cameras.new("InspectCamera")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("InspectCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (cam_dist * 0.85, -cam_dist * 0.85, cam_dist * 0.6)
    # Point at center of object
    direction = mathutils.Vector((0, 0, dims[2] / 2.0)) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Key light
    key_light_data = bpy.data.lights.new("KeyLight", type='AREA')
    key_light_data.energy = 250.0
    key_light_data.size = 2.0
    key_light_obj = bpy.data.objects.new("KeyLight", key_light_data)
    scene.collection.objects.link(key_light_obj)
    key_light_obj.location = (cam_dist, -cam_dist * 0.5, cam_dist)

    # Fill light
    fill_light_data = bpy.data.lights.new("FillLight", type='AREA')
    fill_light_data.energy = 80.0
    fill_light_data.size = 3.0
    fill_light_obj = bpy.data.objects.new("FillLight", fill_light_data)
    scene.collection.objects.link(fill_light_obj)
    fill_light_obj.location = (-cam_dist, -cam_dist, cam_dist * 0.5)

    # Rim light
    rim_light_data = bpy.data.lights.new("RimLight", type='POINT')
    rim_light_data.energy = 150.0
    rim_light_obj = bpy.data.objects.new("RimLight", rim_light_data)
    scene.collection.objects.link(rim_light_obj)
    rim_light_obj.location = (0, cam_dist, cam_dist)

    # 4. Render to output path
    os.makedirs(os.path.dirname(os.path.abspath(render_path)), exist_ok=True)
    scene.render.filepath = os.path.abspath(render_path)
    bpy.ops.render.render(write_still=True)

    # 5. Output JSON report
    report = {
        "blender_verified": True,
        "mesh_path": mesh_path,
        "object_name": primary_obj.name,
        "vertex_count": vert_count,
        "polygon_count": poly_count,
        "dimensions_meters": dims,
        "height_m": dims[2],
        "material_count": len(mats_info),
        "materials": mats_info,
        "render_path": render_path
    }

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[Blender Validator] Verified: {vert_count} verts, {poly_count} polys, {len(mats_info)} materials. Rendered -> {render_path}")


if __name__ == "__main__":
    # Extract arguments passed after '--'
    argv = sys.argv
    if "--" in argv:
        args_to_parse = argv[argv.index("--") + 1:]
    else:
        args_to_parse = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to GLB mesh file")
    parser.add_argument("--render", required=True, help="Path to output preview PNG")
    parser.add_argument("--report", required=True, help="Path to output JSON report")
    args = parser.parse_args(args_to_parse)

    inspect_and_render(args.mesh, args.render, args.report)
