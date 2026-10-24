"""
Blender Automated Post-Processing Bridge.
Executes inside Blender (headless or GUI) to import generated meshes,
sanitize geometry, apply PBR materials, ground the model, and render previews.
"""

import bpy
import sys
import os

# Add core path to sys.path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
gpuplat_dir = os.path.dirname(script_dir)
if gpuplat_dir not in sys.path:
    sys.path.insert(0, gpuplat_dir)

from core.geometry_sanitizer import sanitize_mesh
from core.uv_projection_overlay import apply_pbr_and_projection
from core.adaptive_weld_enhancer import enhance_thin_structures
from core.auto_rig_and_lods import generate_lods

def process_mesh(mesh_path, image_path=None, target_blend_path=None, output_dir=None, height=1.8, create_lods=False, preset="prop", sharp_overlay=True, prompt=None):
    print(f"[Blender Bridge] Loading mesh: {mesh_path} (Preset: {preset})")
    if prompt:
        print(f"[Blender Bridge] Appearance guidance prompt: '{prompt}'")
    
    # If target blend project provided, load it; otherwise clear default scene
    if target_blend_path and os.path.exists(target_blend_path):
        bpy.ops.wm.open_mainfile(filepath=target_blend_path)
        print(f"[Blender Bridge] Loaded existing project: {target_blend_path}")
    else:
        # Clear default cube, light, camera
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
    # Import generated mesh (.glb or .obj)
    if mesh_path.lower().endswith('.glb') or mesh_path.lower().endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.lower().endswith('.obj'):
        try:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        except Exception:
            bpy.ops.import_scene.obj(filepath=mesh_path)
            
    # Find imported object
    imported_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported_objs:
        imported_objs = [o for o in bpy.data.objects if o.type == 'MESH']
        
    if not imported_objs:
        print("[Error] No mesh objects found after import!")
        return False
        
    main_obj = imported_objs[0]
    main_obj.name = "AI_Generated_Model"
    
    # 1. Sanitize geometry and ground to Z=0
    sanitize_mesh(main_obj, target_height_meters=height)
    
    # 2. Weld micro-gaps
    enhance_thin_structures(main_obj, weld_distance=0.001)
    
    # 3. Apply Multi-Material Segmentation & PBR Shaders
    apply_pbr_and_projection(
        main_obj,
        image_path=image_path,
        preset=preset,
        sharp_overlay=sharp_overlay,
        prompt=prompt,
        output_dir=output_dir
    )
    
    # 4. Generate LODs if requested
    if create_lods and output_dir:
        lod_dir = os.path.join(output_dir, "lods")
        generate_lods(main_obj, lod_dir)
        
    # 5. Setup lighting & camera if not existing project
    if not target_blend_path:
        setup_studio_lighting()
        setup_camera(main_obj)
        
    # 6. Save updated .blend file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "project.blend")
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"[Blender Bridge] Saved complete Blender project: {save_path}")
        
        # Render preview image
        render_path = os.path.join(output_dir, "preview.png")
        bpy.context.scene.render.filepath = render_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        try:
            bpy.ops.render.render(write_still=True)
            print(f"[Blender Bridge] Rendered preview image: {render_path}")
        except Exception as e:
            print(f"[Render Notice] {e}")

        # Export sanitized, grounded, multi-material 3D asset (.glb)
        mesh_export_path = os.path.join(output_dir, "mesh.glb")
        try:
            bpy.ops.export_scene.gltf(
                filepath=mesh_export_path,
                export_format='GLB',
                use_selection=False,
                export_materials='EXPORT',
                export_attributes=True,
                export_all_vertex_colors=True,
                export_apply=True
            )
            export_size = os.path.getsize(mesh_export_path) if os.path.exists(mesh_export_path) else 0
            mat_count = len(main_obj.data.materials)
            print(f"[Blender Bridge] Exported PBR 3D asset: {mesh_export_path} ({export_size:,} bytes, {mat_count} materials)")
        except Exception as e:
            print(f"[Blender Bridge Mesh Export Notice] {e}")
            
    return True

def setup_studio_lighting():
    # 3-Point Light Rig
    light_data = bpy.data.lights.new(name="Key_Light", type='AREA')
    light_data.energy = 300
    light_data.size = 2.0
    key_obj = bpy.data.objects.new(name="Key_Light", object_data=light_data)
    bpy.context.collection.objects.link(key_obj)
    key_obj.location = (3.0, -3.0, 3.5)
    
    fill_data = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_data.energy = 150
    fill_data.size = 3.0
    fill_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = (-3.0, -2.5, 2.5)

def setup_camera(target_obj):
    cam_data = bpy.data.cameras.new(name="Render_Camera")
    cam_obj = bpy.data.objects.new(name="Render_Camera", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, -3.5, 1.2)
    cam_obj.rotation_euler = (1.4, 0, 0)
    bpy.context.scene.camera = cam_obj

if __name__ == "__main__":
    # Parse CLI arguments passed after '--'
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        
    mesh_file = None
    img_file = None
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "output")
    target_blend = None
    
    preset = "prop"
    sharp_overlay = True
    target_height = 1.8
    create_lods = False
    prompt_text = None
    
    for i, a in enumerate(args):
        if a == "--mesh" and i + 1 < len(args):
            mesh_file = args[i + 1]
        elif a == "--image" and i + 1 < len(args):
            img_file = args[i + 1]
        elif a == "--out" and i + 1 < len(args):
            out_dir = args[i + 1]
        elif a == "--blend" and i + 1 < len(args):
            target_blend = args[i + 1]
        elif a == "--preset" and i + 1 < len(args):
            preset = args[i + 1]
        elif a == "--prompt" and i + 1 < len(args):
            prompt_text = args[i + 1]
        elif a == "--height" and i + 1 < len(args):
            try:
                target_height = float(args[i + 1])
            except ValueError:
                pass
        elif a == "--lods":
            create_lods = True
        elif a == "--no-overlay":
            sharp_overlay = False
            
    if mesh_file:
        process_mesh(
            mesh_file,
            image_path=img_file,
            target_blend_path=target_blend,
            output_dir=out_dir,
            height=target_height,
            create_lods=create_lods,
            preset=preset,
            sharp_overlay=sharp_overlay,
            prompt=prompt_text
        )
