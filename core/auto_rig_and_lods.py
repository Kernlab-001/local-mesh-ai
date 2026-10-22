"""
Auto-Rigging & LOD Export Pipeline.
Generates 3 game engine LODs (LOD0: 100%, LOD1: 50%, LOD2: 20%)
and optionally sets up an armature.
"""

import bpy
import os

def generate_lods(obj, output_base_dir):
    if not obj or obj.type != 'MESH':
        return []
        
    os.makedirs(output_base_dir, exist_ok=True)
    lod_configs = [("LOD0", 1.0), ("LOD1", 0.5), ("LOD2", 0.2)]
    exported_files = []
    
    # Save original mesh state
    orig_name = obj.name
    
    for lod_name, ratio in lod_configs:
        # Create duplicate object for decimation
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate(linked=False)
        dup_obj = bpy.context.active_object
        dup_obj.name = f"{orig_name}_{lod_name}"
        
        if ratio < 1.0:
            dec_mod = dup_obj.modifiers.new(name="Decimate_LOD", type='DECIMATE')
            dec_mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier=dec_mod.name)
            
        export_file = os.path.join(output_base_dir, f"{dup_obj.name}.glb")
        
        # Select only this LOD object for export
        bpy.ops.object.select_all(action='DESELECT')
        dup_obj.select_set(True)
        bpy.ops.export_scene.gltf(
            filepath=export_file,
            use_selection=True,
            export_format='GLB'
        )
        
        exported_files.append(export_file)
        print(f"[LOD] Exported {lod_name} ({int(ratio*100)}% triangles) -> {export_file}")
        
        # Remove duplicate object from active scene
        bpy.data.objects.remove(dup_obj, do_unlink=True)
        
    # Re-select original
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return exported_files
