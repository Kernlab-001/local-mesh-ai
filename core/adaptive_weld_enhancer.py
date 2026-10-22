"""
Adaptive Weld & Boundary Enhancer.
Seals non-manifold boundaries and reinforces ultra-thin geometries
(blades, wires, leaves, thin edges).
"""

import bpy

def enhance_thin_structures(obj, weld_distance=0.001, add_solidify=False):
    if not obj or obj.type != 'MESH':
        return False
        
    bpy.context.view_layer.objects.active = obj
    
    # 1. Weld Modifier to fuse micro-gaps
    weld_mod = obj.modifiers.new(name="Weld_MicroGaps", type='WELD')
    if hasattr(weld_mod, 'merge_threshold'):
        weld_mod.merge_threshold = weld_distance
    elif hasattr(weld_mod, 'distance'):
        weld_mod.distance = weld_distance
    try:
        bpy.ops.object.modifier_apply(modifier=weld_mod.name)
    except Exception as e:
        print(f"[Weld Notice] {e}")
        
    # 2. Optional Solidify for paper-thin surfaces
    if add_solidify:
        solid_mod = obj.modifiers.new(name="Solidify_Thin", type='SOLIDIFY')
        solid_mod.thickness = 0.002  # 2mm thickness
        try:
            bpy.ops.object.modifier_apply(modifier=solid_mod.name)
        except Exception as e:
            print(f"[Solidify Notice] {e}")
            
    print(f"[Success] Applied weld and structure enhancements to '{obj.name}'.")
    return True
