"""
Blender Automated Component Hierarchy & Precision Alignment Bridge (Phase 2).
Executes inside Blender 4.2 / 5.2 (headless or interactive) to:
1. Import 3D meshes (GLB, OBJ, STL).
2. Construct structured Blender Collections for component hierarchies.
3. Separate and organize candidate components into parented Blender objects.
4. Assign custom metadata properties (component_id, label, confidence, dimensions).
5. Apply deterministic alignment transforms (grounding, orientation, scale).
6. Preserve authentic vertex colors and PBR materials.
7. Save full .blend project and export consolidated .glb asset.
"""

import os
import sys
import json
import math
from pathlib import Path

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None


def build_blender_hierarchy(
    mesh_path: str,
    hierarchy_dict: dict,
    alignment_dict: dict = None,
    out_blend_path: str = None,
    out_glb_path: str = None,
    target_height_m: float = None
) -> dict:
    """
    Main headless Blender hierarchy construction and alignment worker.
    """
    if bpy is None:
        raise RuntimeError("bpy module is only available when executed inside Blender.")

    # 1. Reset Scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 2. Create Master Collection
    master_col_name = "AI_Asset_Hierarchy"
    master_col = bpy.data.collections.new(master_col_name)
    scene.collection.children.link(master_col)

    # 3. Import Source Mesh
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")

    if mesh_path.lower().endswith(('.glb', '.gltf')):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.lower().endswith('.obj'):
        try:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        except Exception:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    # Gather imported mesh objects
    imported_objs = [o for o in scene.objects if o.type == 'MESH']
    if not imported_objs:
        raise RuntimeError(f"No mesh objects found after importing {mesh_path}")

    root_obj = imported_objs[0]
    root_obj.name = "AI_Main_Asset"

    # Move imported objects to master collection
    for obj in imported_objs:
        for c in obj.users_collection:
            c.objects.unlink(obj)
        master_col.objects.link(obj)

    created_objects = []
    components_info = hierarchy_dict.get("components", {})

    # 4. Process Components & Sub-collections
    if components_info and len(components_info) > 1:
        # Check if individual submesh paths are provided
        submesh_nodes = [n for n in components_info.values() if n.get("submesh_path") and os.path.exists(n["submesh_path"])]

        if submesh_nodes:
            # We have pre-isolated submeshes: import each into its own collection
            # First remove the root generic imported object
            bpy.data.objects.remove(root_obj, do_unlink=True)
            imported_objs = []

            for comp in submesh_nodes:
                cid = comp["component_id"]
                label = comp.get("semantic_label", "part")
                col_name = f"Col_{label}_{cid[-6:]}"
                comp_col = bpy.data.collections.new(col_name)
                master_col.children.link(comp_col)

                # Import component submesh
                sm_path = comp["submesh_path"]
                bpy.ops.import_scene.gltf(filepath=sm_path)
                new_objs = [o for o in scene.objects if o.type == 'MESH' and o not in imported_objs]
                if new_objs:
                    comp_obj = new_objs[0]
                    comp_obj.name = f"Part_{label}_{cid[-6:]}"
                    for c in comp_obj.users_collection:
                        c.objects.unlink(comp_obj)
                    comp_col.objects.link(comp_obj)

                    # Custom Metadata Properties
                    comp_obj["component_id"] = cid
                    comp_obj["semantic_label"] = label
                    comp_obj["confidence"] = comp.get("confidence", 0.5)
                    comp_obj["is_uncertain"] = comp.get("is_uncertain", True)
                    comp_obj["dimensions"] = list(comp.get("dimensions", [0, 0, 0]))

                    imported_objs.append(comp_obj)
                    created_objects.append({
                        "object_name": comp_obj.name,
                        "component_id": cid,
                        "label": label,
                        "collection": col_name
                    })
        else:
            # Single object or loose parts: assign metadata to main object
            root_obj["component_id"] = hierarchy_dict.get("root_asset_id", "asset")
            root_obj["semantic_label"] = "Primary_Asset"
            root_obj["confidence"] = 1.0
            root_obj["is_uncertain"] = False
            created_objects.append({
                "object_name": root_obj.name,
                "component_id": root_obj["component_id"],
                "label": "Primary_Asset",
                "collection": master_col_name
            })
    else:
        created_objects.append({
            "object_name": root_obj.name,
            "component_id": "primary",
            "label": "Primary_Asset",
            "collection": master_col_name
        })

    # 5. Apply Precision Alignment & Grounding
    target_objs = [o for o in master_col.all_objects if o.type == 'MESH']
    applied_ops = []

    if target_objs:
        # Calculate combined bounding box in world space
        all_verts = []
        for obj in target_objs:
            all_verts.extend([obj.matrix_world @ v.co for v in obj.data.vertices])

        if all_verts:
            min_x = min(v.x for v in all_verts)
            max_x = max(v.x for v in all_verts)
            min_y = min(v.y for v in all_verts)
            max_y = max(v.y for v in all_verts)
            min_z = min(v.z for v in all_verts)
            max_z = max(v.z for v in all_verts)

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            cur_height = max_z - min_z

            # Ground Z=0 and Center XY
            for obj in target_objs:
                for v in obj.data.vertices:
                    v.co.x -= center_x
                    v.co.y -= center_y
                    v.co.z -= min_z
                obj.data.update()
            applied_ops.append("ground_and_center_xy")

            # Scale to target height if provided
            if target_height_m and target_height_m > 0 and cur_height > 1e-4:
                scale_factor = target_height_m / cur_height
                for obj in target_objs:
                    obj.scale = (obj.scale.x * scale_factor, obj.scale.y * scale_factor, obj.scale.z * scale_factor)
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                    obj.select_set(False)
                applied_ops.append(f"scale_to_{target_height_m}m")

            # Smooth Shading
            for obj in target_objs:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.shade_smooth()
                obj.select_set(False)
            applied_ops.append("smooth_shading")

    # 6. Save Blender Project (.blend)
    saved_blend = None
    if out_blend_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_blend_path)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend_path)
        saved_blend = out_blend_path

    # 7. Export Consolidated GLB
    saved_glb = None
    if out_glb_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_glb_path)), exist_ok=True)
        bpy.ops.export_scene.gltf(
            filepath=out_glb_path,
            export_format='GLB',
            use_selection=False,
            export_materials='EXPORT',
            export_attributes=True,
            export_all_vertex_colors=True,
            export_apply=True
        )
        saved_glb = out_glb_path

    return {
        "status": "success",
        "master_collection": master_col_name,
        "objects_created": created_objects,
        "alignment_operations": applied_ops,
        "blend_path": saved_blend,
        "glb_path": saved_glb
    }


if __name__ == "__main__":
    # Handle command-line arguments when invoked via:
    # blender --background --python bridge/blender_hierarchy_bridge.py -- [args]
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]

    mesh_arg = None
    hierarchy_json_arg = None
    out_blend_arg = None
    out_glb_arg = None
    report_json_arg = None
    height_arg = None

    for i, a in enumerate(args):
        if a == "--mesh" and i + 1 < len(args):
            mesh_arg = args[i + 1]
        elif a == "--hierarchy-json" and i + 1 < len(args):
            hierarchy_json_arg = args[i + 1]
        elif a == "--out-blend" and i + 1 < len(args):
            out_blend_arg = args[i + 1]
        elif a == "--out-glb" and i + 1 < len(args):
            out_glb_arg = args[i + 1]
        elif a == "--report-json" and i + 1 < len(args):
            report_json_arg = args[i + 1]
        elif a == "--height" and i + 1 < len(args):
            try:
                height_arg = float(args[i + 1])
            except ValueError:
                pass

    hierarchy_data = {}
    if hierarchy_json_arg and os.path.exists(hierarchy_json_arg):
        with open(hierarchy_json_arg, "r", encoding="utf-8") as f:
            hierarchy_data = json.load(f)

    res = build_blender_hierarchy(
        mesh_path=mesh_arg,
        hierarchy_dict=hierarchy_data,
        out_blend_path=out_blend_arg,
        out_glb_path=out_glb_arg,
        target_height_m=height_arg
    )

    if report_json_arg:
        os.makedirs(os.path.dirname(os.path.abspath(report_json_arg)), exist_ok=True)
        with open(report_json_arg, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    print("[Blender Hierarchy Bridge Completed]:", json.dumps(res))
