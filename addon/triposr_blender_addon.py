bl_info = {
    "name": "Local Neural 3D Studio",
    "author": "Antigravity Architecture & TripoSR",
    "version": (2, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AI 3D Generator",
    "description": "Local-first 3D asset generation & editing using TripoSR, SD-Turbo, Qwen 2.5 Coder & RTX 3050 GPU",
    "category": "3D View",
}

import bpy
import os
import sys
import json
import math
import urllib.request
import urllib.parse
from bpy.props import (
    StringProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    IntProperty
)

SERVER_URL = "http://127.0.0.1:8000"

# --- Property Group for Add-on State ---
class TripoSRProperties(bpy.types.PropertyGroup):
    input_mode: EnumProperty(
        name="Mode",
        items=[
            ("prompt", "Prompt to 3D", "Generate concept artwork & 3D asset from text prompt"),
            ("image", "Image to 3D", "Reconstruct 3D asset from an existing 2D image"),
        ],
        default="prompt"
    )
    prompt_text: StringProperty(
        name="Concept Prompt",
        description="Describe the 3D asset you want to create",
        default="medieval glowing battleaxe with carved wooden handle and runes"
    )
    image_path: StringProperty(
        name="Image Path",
        description="Select a 2D concept or reference image",
        subtype='FILE_PATH'
    )
    remove_bg: BoolProperty(
        name="Remove Background",
        description="Automatically remove image background before 3D reconstruction",
        default=True
    )
    preset: EnumProperty(
        name="Asset Preset",
        items=[
            ("prop", "Game Prop", "Optimized for static hard-surface game props"),
            ("weapon", "Weapon", "High edge sharpness & metallic blade/hilt reflectivity"),
            ("character", "Character", "Configured for organic humanoid characters & creatures"),
            ("furniture", "Furniture", "Centered pivot with sturdy ground alignment"),
        ],
        default="weapon"
    )
    quality: EnumProperty(
        name="Quality",
        items=[
            ("draft", "Draft (Fast)", "128 resolution, rapid 15s prototyping"),
            ("balanced", "Balanced (Recommended)", "256 resolution, optimal detail & VRAM"),
            ("high", "High Detail", "256 resolution with tighter chunking"),
        ],
        default="balanced"
    )
    target_height: FloatProperty(
        name="Target Height (m)",
        description="Final real-world height of the asset in meters",
        default=1.8,
        min=0.05,
        max=50.0
    )
    
    # Accurate Color & Processing Toggles
    accurate_colors: BoolProperty(
        name="Accurate Colors (Tripo AI)",
        description="Calibrates 3D mesh colors to match reference artwork with 1:1 fidelity",
        default=True
    )
    color_boost: FloatProperty(
        name="Color Vibrancy",
        description="Saturation multiplier for rich, non-washed-out colors",
        default=1.25,
        min=0.5,
        max=2.0
    )
    auto_switch_shading: BoolProperty(
        name="Auto Viewport Color Shading",
        description="Automatically switches 3D viewport to Material Preview mode so colors show up immediately",
        default=True
    )
    auto_ground: BoolProperty(name="Smart Grounding (Z=0)", default=True)
    center_pivot: BoolProperty(name="Center Pivot to Base", default=True)
    sanitize_mesh: BoolProperty(name="Sanitize Geometry", default=True)
    sharp_overlay: BoolProperty(name="Sharp Texture Overlay", default=True)
    adaptive_weld: BoolProperty(name="Adaptive Welding", default=True)
    pbr_material: BoolProperty(name="Generate PBR Shaders", default=True)
    generate_lods: BoolProperty(name="Generate LOD0, 1, 2", default=False)
    auto_rig: BoolProperty(name="Attach Rigify Metarig", default=False)
    
    # Destination Controls
    dest_mode: EnumProperty(
        name="Destination",
        items=[
            ("new_col", "New Isolated Collection", "Creates a fresh labeled collection"),
            ("active_col", "Active Collection", "Inserts directly into current collection"),
        ],
        default="new_col"
    )
    collection_name: StringProperty(name="Collection Name", default="AI_Generated_Assets")

    # State & Polling
    active_job_id: StringProperty(name="Active Job ID", default="")
    status_message: StringProperty(name="Status", default="Ready")
    stage_name: StringProperty(name="Stage", default="Idle")
    progress_pct: IntProperty(name="Progress", default=0, min=0, max=100)
    is_generating: BoolProperty(name="Is Generating", default=False)
    wireframe_active: BoolProperty(name="Wireframe", default=False)

# --- Timer for Non-Blocking Polling ---
def poll_job_timer():
    scene = bpy.context.scene
    props = scene.triposr_props
    if not props.is_generating or not props.active_job_id:
        return None

    try:
        url = f"{SERVER_URL}/jobs/{props.active_job_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode('utf-8'))
            props.stage_name = data.get("stage", "Processing")
            props.progress_pct = data.get("progress_pct", 0)
            status = data.get("status")

            if status == "COMPLETED":
                props.is_generating = False
                props.status_message = "Generation Complete!"
                props.stage_name = "Done"
                props.progress_pct = 100
                
                # Import output mesh with full PBR and viewport shading
                output_mesh = data.get("output_mesh")
                if output_mesh and os.path.exists(output_mesh):
                    _import_and_organize(output_mesh, props)
                return None
            elif status in ("FAILED", "CANCELLED"):
                props.is_generating = False
                props.status_message = f"{status}: {data.get('error', 'Unknown')}"
                return None
    except Exception as e:
        props.status_message = f"Connecting to server... ({str(e)[:30]})"

    return 1.0  # Poll every 1 second

def _import_and_organize(filepath, props):
    """
    Imports mesh, links to destination collection, builds calibrated PBR shader,
    and switches viewport to Material Preview mode so colors show up immediately!
    """
    # 1. Collection handling
    if props.dest_mode == "new_col":
        col_name = props.collection_name
        col = bpy.data.collections.get(col_name)
        if not col:
            col = bpy.data.collections.new(col_name)
            bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.context.collection

    # 2. Import GLTF
    bpy.ops.import_scene.gltf(filepath=filepath)
    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported:
        return

    main_obj = imported[0]

    # Move to target collection
    for obj in imported:
        if obj.name not in col.objects:
            col.objects.link(obj)
            try:
                bpy.context.scene.collection.objects.unlink(obj)
            except Exception:
                pass

    bpy.context.view_layer.objects.active = main_obj
    main_obj.select_set(True)

    # 3. Build Calibrated PBR Shading & Color Setup
    if props.pbr_material:
        _apply_in_viewport_pbr(main_obj, filepath, props)

    # 4. Viewport Shading Mode: Auto-switch to Material Preview
    if props.auto_switch_shading:
        _enable_viewport_color_shading()

def _apply_in_viewport_pbr(obj, mesh_path, props):
    """Constructs native Blender 5.2 / 4.2 PBR shader with calibrated colors."""
    mat_name = f"M_{obj.name}_PBR"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output & Principled BSDF
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (600, 0)

    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (200, 0)
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    # Preset Reflectivity
    p_lower = props.preset.lower()
    if "weapon" in p_lower:
        node_bsdf.inputs['Roughness'].default_value = 0.25
        if 'Metallic' in node_bsdf.inputs:
            node_bsdf.inputs['Metallic'].default_value = 0.55
    elif "character" in p_lower:
        node_bsdf.inputs['Roughness'].default_value = 0.50
    elif "furniture" in p_lower:
        node_bsdf.inputs['Roughness'].default_value = 0.45
    else:  # prop
        node_bsdf.inputs['Roughness'].default_value = 0.35

    # Check Color Attributes
    color_attr_name = 'Color'
    if obj.data.color_attributes:
        color_attr_name = obj.data.color_attributes[0].name

    # Color Attribute Node
    col_node = nodes.new('ShaderNodeAttribute')
    col_node.location = (-600, 100)
    col_node.attribute_name = color_attr_name

    # Color Enhancement Chain (Tripo AI Saturation & Vibrancy Boost)
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (-350, 100)
    hsv.inputs['Saturation'].default_value = props.color_boost
    hsv.inputs['Value'].default_value = 1.05
    links.new(col_node.outputs['Color'], hsv.inputs['Color'])

    bc = nodes.new('ShaderNodeBrightContrast')
    bc.location = (-150, 100)
    bc.inputs['Bright'].default_value = 0.02
    bc.inputs['Contrast'].default_value = 0.15
    links.new(hsv.outputs['Color'], bc.inputs['Color'])

    links.new(bc.outputs['Color'], node_bsdf.inputs['Base Color'])

    # Set scene color transform to Standard so it doesn't look washed out
    if hasattr(bpy.context.scene, 'view_settings'):
        bpy.context.scene.view_settings.view_transform = 'Standard'

def _enable_viewport_color_shading():
    """Switches 3D Viewport to Material Preview mode and Solid to Vertex Color."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # Switch to Material Preview mode
                        space.shading.type = 'MATERIAL'
                        # In Solid view, also show vertex colors
                        try:
                            space.shading.color_type = 'VERTEX'
                        except Exception:
                            pass

# --- Operators ---
class TRIPOSR_OT_Generate(bpy.types.Operator):
    bl_idname = "triposr.generate_asset"
    bl_label = "⚡ Generate 3D Asset"
    bl_description = "Submit prompt or image to local GPU orchestration server"

    def execute(self, context):
        props = context.scene.triposr_props
        payload = {
            "target_height": props.target_height,
            "quality": props.quality,
            "preset": props.preset,
            "remove_bg": props.remove_bg,
            "sanitize": props.sanitize_mesh,
            "sharp_overlay": props.sharp_overlay,
            "weld": props.adaptive_weld,
            "pbr_material": props.pbr_material,
            "accurate_colors": props.accurate_colors,
            "generate_lods": props.generate_lods,
            "auto_rig": props.auto_rig
        }

        if props.input_mode == "prompt":
            if not props.prompt_text or len(props.prompt_text.strip()) < 2:
                self.report({'ERROR'}, "Please enter a descriptive prompt.")
                return {'CANCELLED'}
            payload["prompt"] = props.prompt_text.strip()
        else:
            if not props.image_path or not os.path.exists(bpy.path.abspath(props.image_path)):
                self.report({'ERROR'}, "Please select a valid reference image.")
                return {'CANCELLED'}
            payload["image_path"] = os.path.abspath(bpy.path.abspath(props.image_path))

        try:
            req = urllib.request.Request(
                f"{SERVER_URL}/jobs",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                props.active_job_id = data["job_id"]
                props.is_generating = True
                props.status_message = "Job Queued on GPU..."
                props.stage_name = "Queued"
                props.progress_pct = 5
                bpy.app.timers.register(poll_job_timer)
                self.report({'INFO'}, f"Job submitted (ID: {props.active_job_id})")
        except Exception as e:
            self.report({'ERROR'}, f"Server connection failed: {e}. Is start_api.bat running?")
            return {'CANCELLED'}

        return {'FINISHED'}

class TRIPOSR_OT_Cancel(bpy.types.Operator):
    bl_idname = "triposr.cancel_asset"
    bl_label = "Cancel Job"

    def execute(self, context):
        props = context.scene.triposr_props
        if props.active_job_id:
            try:
                urllib.request.urlopen(f"{SERVER_URL}/jobs/{props.active_job_id}/cancel", data=b"", timeout=3)
                props.is_generating = False
                props.status_message = "Job Cancelled"
            except Exception:
                pass
        return {'FINISHED'}

class TRIPOSR_OT_ToggleWireframe(bpy.types.Operator):
    bl_idname = "triposr.toggle_wireframe"
    bl_label = "Toggle Wireframe"

    def execute(self, context):
        props = context.scene.triposr_props
        props.wireframe_active = not props.wireframe_active
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                obj.show_wire = props.wireframe_active
        return {'FINISHED'}

class TRIPOSR_OT_SwitchViewport(bpy.types.Operator):
    bl_idname = "triposr.switch_viewport"
    bl_label = "Show Colors (Material Mode)"
    bl_description = "Switches the 3D viewport to Material Preview mode to see accurate colors"

    def execute(self, context):
        _enable_viewport_color_shading()
        self.report({'INFO'}, "Switched Viewport to Material Preview mode.")
        return {'FINISHED'}

class TRIPOSR_OT_ResetDefaults(bpy.types.Operator):
    bl_idname = "triposr.reset_defaults"
    bl_label = "Reset Settings"

    def execute(self, context):
        props = context.scene.triposr_props
        props.target_height = 1.8
        props.preset = "weapon"
        props.quality = "balanced"
        props.auto_ground = True
        props.sanitize_mesh = True
        props.adaptive_weld = True
        props.pbr_material = True
        props.accurate_colors = True
        props.color_boost = 1.25
        return {'FINISHED'}

# --- Main N-Panel UI ---
class TRIPOSR_PT_MainPanel(bpy.types.Panel):
    bl_label = "Local Neural 3D Studio"
    bl_idname = "TRIPOSR_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AI 3D Generator"

    def draw(self, context):
        layout = self.layout
        props = context.scene.triposr_props

        # Mode Selector Tabs
        mode_box = layout.box()
        mode_row = mode_box.row(align=True)
        mode_row.prop(props, "input_mode", expand=True)

        # Section A: Input
        box = layout.box()
        if props.input_mode == "prompt":
            box.label(text="1. Text-to-3D Prompt", icon='FONT_DATA')
            box.prop(props, "prompt_text", text="")
            box.label(text="AI generates artwork & builds 3D mesh", icon='INFO')
        else:
            box.label(text="1. Concept Reference Image", icon='IMAGE_DATA')
            box.prop(props, "image_path", text="")
            row = box.row()
            row.prop(props, "remove_bg")

        # Section B: Preset & Config
        box = layout.box()
        box.label(text="2. Asset Configuration", icon='PREFERENCES')
        box.prop(props, "preset", text="Preset")
        box.prop(props, "quality", text="Quality")
        box.prop(props, "target_height", text="Height (m)")

        # Section C: Advanced Color & Shading
        col = layout.box()
        col.label(text="3. Colors & PBR Shading", icon='MATERIAL')
        c_row1 = col.row()
        c_row1.prop(props, "accurate_colors")
        c_row1.prop(props, "pbr_material")
        
        c_row2 = col.row()
        c_row2.prop(props, "color_boost", text="Vibrancy")
        c_row2.prop(props, "auto_switch_shading", text="Auto Preview")

        col.label(text="Geometry & Alignment:", icon='MODIFIER')
        g_row1 = col.row()
        g_row1.prop(props, "auto_ground")
        g_row1.prop(props, "center_pivot")
        
        g_row2 = col.row()
        g_row2.prop(props, "sanitize_mesh")
        g_row2.prop(props, "adaptive_weld")

        # Section D: Destination
        dest_box = layout.box()
        dest_box.label(text="4. Scene Destination", icon='OUTLINER_COLLECTION')
        dest_box.prop(props, "dest_mode", text="")
        if props.dest_mode == "new_col":
            dest_box.prop(props, "collection_name", text="Name")

        # Section E: Action & Progress
        layout.separator()
        if not props.is_generating:
            btn_text = "⚡ Generate 3D from Prompt" if props.input_mode == "prompt" else "⚡ Generate 3D from Image"
            layout.operator("triposr.generate_asset", text=btn_text, icon='SHADING_RENDERED')
        else:
            prog_box = layout.box()
            prog_box.label(text=f"Stage: {props.stage_name}", icon='TIME')
            prog_box.label(text=f"Progress: {props.progress_pct}%")
            prog_box.operator("triposr.cancel_asset", icon='CANCEL')

        # Section F: Inspection & Quick Color Viewport Switch
        stat_box = layout.box()
        stat_box.label(text="Quick Tools & Viewport:", icon='TOOL_SETTINGS')
        row_tools = stat_box.row()
        row_tools.operator("triposr.switch_viewport", icon='SHADING_TEXTURE')
        row_tools.operator("triposr.toggle_wireframe", icon='SHADING_WIRE')

        if context.active_object and context.active_object.type == 'MESH':
            mesh = context.active_object.data
            stat_box.label(text=f"Active: {context.active_object.name} ({len(mesh.polygons):,} polys)")

        layout.operator("triposr.reset_defaults", icon='LOOP_BACK')

# --- Registration ---
classes = (
    TripoSRProperties,
    TRIPOSR_OT_Generate,
    TRIPOSR_OT_Cancel,
    TRIPOSR_OT_ToggleWireframe,
    TRIPOSR_OT_SwitchViewport,
    TRIPOSR_OT_ResetDefaults,
    TRIPOSR_PT_MainPanel
)

def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            try:
                bpy.utils.unregister_class(cls)
                bpy.utils.register_class(cls)
            except Exception:
                pass
    bpy.types.Scene.triposr_props = bpy.props.PointerProperty(type=TripoSRProperties)

def unregister():
    if hasattr(bpy.types.Scene, "triposr_props"):
        try:
            del bpy.types.Scene.triposr_props
        except Exception:
            pass
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

if __name__ == "__main__":
    register()
