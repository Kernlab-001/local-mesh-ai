"""
Deterministic Blender Command Plan Executor.
Executes approved, validated CommandPlans inside Blender 4.2 / 5.2.
Guarantees zero arbitrary code execution.
"""

import bpy
import math
import os
import json

def execute_command_plan(plan_dict: dict, blend_path: str, output_path: str) -> dict:
    """
    Opens target blend file, applies safe commands, and saves result.
    """
    if os.path.exists(blend_path):
        bpy.ops.wm.open_mainfile(filepath=blend_path)
    else:
        raise FileNotFoundError(f"Blend file not found: {blend_path}")

    results = []
    commands = plan_dict.get("commands", [])

    for cmd in commands:
        op = cmd.get("op")
        target_name = cmd.get("target_object", "AI_Generated_Model")
        obj = bpy.data.objects.get(target_name)

        if not obj:
            # Fallback: pick first mesh object
            mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
            if mesh_objs:
                obj = mesh_objs[0]
            else:
                results.append({"op": op, "status": "failed", "error": f"Object '{target_name}' not found."})
                continue

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        try:
            if op == "set_material":
                _apply_material(obj, cmd)
            elif op == "bevel":
                _apply_bevel(obj, cmd)
            elif op == "set_height":
                _apply_height(obj, cmd)
            elif op == "set_location":
                _apply_location(obj, cmd)
            elif op == "set_rotation":
                _apply_rotation(obj, cmd)
            elif op == "smooth_shading":
                _apply_smooth_shading(obj, cmd)
            elif op == "add_subdivision":
                _apply_subdivision(obj, cmd)
            elif op == "decimate":
                _apply_decimate(obj, cmd)
            elif op == "duplicate":
                _apply_duplicate(obj, cmd)
            else:
                results.append({"op": op, "status": "skipped", "error": "Unknown op"})
                continue

            results.append({"op": op, "status": "success"})
        except Exception as e:
            results.append({"op": op, "status": "failed", "error": str(e)})

    # Save output blend file
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    return {"status": "success", "results": results, "saved_to": output_path}

def _apply_material(obj, cmd):
    preset = cmd.get("preset", "custom")
    metallic = cmd.get("metallic", 0.0)
    roughness = cmd.get("roughness", 0.5)
    base_color = cmd.get("base_color", [0.8, 0.8, 0.8, 1.0])

    if preset == "gold":
        base_color = [1.0, 0.766, 0.336, 1.0]
        metallic = 1.0
        roughness = 0.15
    elif preset == "metal":
        metallic = 0.95
        roughness = 0.25
    elif preset == "wood":
        base_color = [0.35, 0.2, 0.1, 1.0]
        metallic = 0.0
        roughness = 0.7
    elif preset == "plastic":
        metallic = 0.0
        roughness = 0.3
    elif preset == "stone":
        base_color = [0.4, 0.4, 0.4, 1.0]
        metallic = 0.0
        roughness = 0.85

    mat = obj.data.materials[0] if obj.data.materials else bpy.data.materials.new(name="M_Material")
    if not obj.data.materials:
        obj.data.materials.append(mat)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = base_color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness

def _apply_bevel(obj, cmd):
    width = cmd.get("width", 0.01)
    segments = cmd.get("segments", 3)
    mod = obj.modifiers.new(name="AI_Bevel", type='BEVEL')
    mod.width = width
    mod.segments = segments
    bpy.ops.object.modifier_apply(modifier=mod.name)

def _apply_height(obj, cmd):
    target_h = cmd.get("height_meters", 1.8)
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_z = min(v.z for v in bbox)
    max_z = max(v.z for v in bbox)
    cur_h = max_z - min_z
    if cur_h > 0:
        factor = target_h / cur_h
        obj.scale = (obj.scale.x * factor, obj.scale.y * factor, obj.scale.z * factor)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def _apply_location(obj, cmd):
    obj.location = (cmd.get("x", 0.0), cmd.get("y", 0.0), cmd.get("z", 0.0))

def _apply_rotation(obj, cmd):
    obj.rotation_euler = (
        math.radians(cmd.get("rx_deg", 0.0)),
        math.radians(cmd.get("ry_deg", 0.0)),
        math.radians(cmd.get("rz_deg", 0.0))
    )

def _apply_smooth_shading(obj, cmd):
    if cmd.get("enable", True):
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()

def _apply_subdivision(obj, cmd):
    mod = obj.modifiers.new(name="AI_Subdiv", type='SUBSURF')
    mod.levels = cmd.get("levels", 1)
    bpy.ops.object.modifier_apply(modifier=mod.name)

def _apply_decimate(obj, cmd):
    mod = obj.modifiers.new(name="AI_Decimate", type='DECIMATE')
    mod.ratio = cmd.get("ratio", 0.5)
    bpy.ops.object.modifier_apply(modifier=mod.name)

def _apply_duplicate(obj, cmd):
    bpy.ops.object.duplicate(linked=False)
    dup = bpy.context.active_object
    dup.name = cmd.get("new_name", f"{obj.name}_copy")
    dup.location.x += cmd.get("offset_x", 1.0)
    dup.location.y += cmd.get("offset_y", 0.0)
    dup.location.z += cmd.get("offset_z", 0.0)

if __name__ == "__main__":
    import sys
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
        plan_json_path = argv[0]
        blend_in = argv[1]
        blend_out = argv[2]
        with open(plan_json_path, "r") as f:
            plan = json.load(f)
        res = execute_command_plan(plan, blend_in, blend_out)
        print(json.dumps(res))
