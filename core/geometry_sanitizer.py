"""
Geometry Sanitizer for Blender 4.x / 5.x.
Cleans raw neural 3D meshes (Marching Cubes output from TripoSR).
Removes loose vertices/faces, caps holes, centers pivot, snaps to ground plane Z=0,
and applies smooth shading.
"""

import bpy
import math

def sanitize_mesh(obj, target_height_meters=1.8):
    if obj is None or obj.type != 'MESH':
        return False
        
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # 1. Clean floating loose geometry and cap non-manifold holes
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    try:
        bpy.ops.mesh.fill_holes(sides=4)
    except Exception as e:
        print(f"[Warning] Hole fill notice: {e}")
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 2. Smart Grounding: Snap lowest vertex to Z=0 and center pivot
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not vertices:
        return False
        
    min_x = min(v.x for v in vertices)
    max_x = max(v.x for v in vertices)
    min_y = min(v.y for v in vertices)
    max_y = max(v.y for v in vertices)
    min_z = min(v.z for v in vertices)
    max_z = max(v.z for v in vertices)
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Offset coordinates so bottom center is at (0, 0, 0)
    for v in obj.data.vertices:
        v.co.x -= center_x
        v.co.y -= center_y
        v.co.z -= min_z
        
    obj.data.update()
    
    # 3. Scaling to real-world target height
    current_height = max_z - min_z
    if current_height > 0 and target_height_meters:
        scale_factor = target_height_meters / current_height
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
    # 4. Smooth Shading for modern Blender
    bpy.ops.object.shade_smooth()
    
    # Try adding Smooth by Angle geometry modifier
    try:
        mod = obj.modifiers.new(name="SmoothByAngle", type='NODES')
        # Blender built-in geometry node asset
    except Exception:
        pass
        
    print(f"[Success] Mesh '{obj.name}' sanitized: Grounded at Z=0, scaled to {target_height_meters}m.")
    return True
