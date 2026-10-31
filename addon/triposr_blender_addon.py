# -*- coding: utf-8 -*-
# GNU General Public License v3.0 (GPLv3)
# Copyright (C) 2024-2026 Local Mesh AI Contributors & Blender Foundation
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

bl_info = {
    "name": "Local Neural 3D Studio",
    "author": "Antigravity Architecture & TripoSR",
    "version": (5, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AI 3D Generator",
    "description": "Local-first 3D asset generation & editing using TripoSR, SD-Turbo, Qwen 2.5 Coder & local GPU",
    "category": "3D View",
    "license": "GPLv3",
}

import bpy
import os
import sys
import json
import math
import subprocess
import atexit
import urllib.request
import urllib.parse
from bpy.props import (
    StringProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    IntProperty
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
_global_server_process = None

# --- Add-on Preferences (Persists across ALL Blender sessions & new files!) ---
class TripoSRAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    project_dir: StringProperty(
        name="Repo Root Directory",
        description="Path to your local-mesh-ai repository directory (folder containing run_server.py)",
        subtype='DIR_PATH',
        default=""
    )
    auto_start_server: BoolProperty(
        name="Auto-Start Server Console",
        description="Automatically open backend terminal console when generating or when Blender opens",
        default=True
    )
    server_url: StringProperty(
        name="Server URL",
        description="Local FastAPI orchestration server URL",
        default=DEFAULT_SERVER_URL
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Local Mesh AI — Global Preferences", icon='PREFERENCES')
        layout.prop(self, "project_dir")
        layout.prop(self, "auto_start_server")
        layout.prop(self, "server_url")
        
        root = self.project_dir
        if root:
            if os.path.exists(os.path.join(root, "run_server.py")):
                layout.label(text="✅ Repository root verified ('run_server.py' found)", icon='CHECKMARK')
            else:
                layout.label(text="⚠️ 'run_server.py' not found in this folder!", icon='ERROR')
        else:
            layout.label(text="ℹ️ Select the folder where you cloned or extracted local-mesh-ai", icon='INFO')

def get_addon_preferences():
    """Retrieves persistent user preferences for this add-on."""
    try:
        pref = bpy.context.preferences.addons.get(__name__)
        if pref and pref.preferences:
            return pref.preferences
        pref = bpy.context.preferences.addons.get("triposr_blender_addon")
        if pref and pref.preferences:
            return pref.preferences
    except Exception:
        pass
    return None

# --- Server Health & Auto-Launch Helpers ---
def check_server_health(url=DEFAULT_SERVER_URL):
    """Checks if the local backend orchestration server is responding."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False

def find_project_root():
    """Intelligently locates the local-mesh-ai project root directory."""
    # 1. Global Add-on Preferences (persists across new files and sessions)
    pref = get_addon_preferences()
    if pref and pref.project_dir and os.path.exists(os.path.join(pref.project_dir, "run_server.py")):
        return pref.project_dir

    # 2. Check Scene property if user set it in the panel
    try:
        if hasattr(bpy.context, "scene") and hasattr(bpy.context.scene, "triposr_props"):
            custom = bpy.context.scene.triposr_props.project_dir
            if custom and os.path.exists(os.path.join(custom, "run_server.py")):
                if pref:
                    pref.project_dir = custom
                return custom
    except Exception:
        pass

    # 3. Environment variable
    env_dir = os.environ.get("LOCAL_MESH_AI_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "run_server.py")):
        return env_dir

    # 4. Standard candidate paths
    script_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..")),
        script_dir,
        os.getcwd(),
        os.path.expanduser(r"~\local-mesh-ai"),
        r"E:\GPUPLAT",
        r"C:\local-mesh-ai",
        r"D:\local-mesh-ai"
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "run_server.py")):
            return path
    return ""

def launch_backend_console(project_dir=None):
    """
    Spawns run_server.py in a dedicated external terminal console window.
    The console stays open and streams live GPU logs without blocking Blender.
    """
    if not project_dir:
        project_dir = find_project_root()
    if not project_dir or not os.path.exists(project_dir):
        return False, "Project directory not found. Please set 'Repo Root' in the Server Settings panel or Preferences."

    server_script = os.path.join(project_dir, "run_server.py")
    if not os.path.exists(server_script):
        return False, f"'run_server.py' not found in {project_dir}"

    # Locate virtualenv python or system python
    py_candidates = [
        os.path.join(project_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(project_dir, ".venv", "bin", "python"),
        sys.executable,
        "python"
    ]
    py_exe = None
    for cand in py_candidates:
        if os.path.exists(cand):
            py_exe = cand
            break
    if not py_exe:
        py_exe = "python"

    try:
        creation_flags = 0
        if sys.platform == "win32":
            # Spawns a dedicated visible Command Prompt console window
            creation_flags = subprocess.CREATE_NEW_CONSOLE

        # Pass Blender's PID so the server automatically closes when Blender exits!
        blender_pid = os.getpid()
        cmd = [py_exe, server_script, "--parent-pid", str(blender_pid)]

        global _global_server_process
        _global_server_process = subprocess.Popen(
            cmd,
            cwd=project_dir,
            creationflags=creation_flags
        )
        return True, "Backend server console launched successfully!"
    except Exception as e:
        return False, f"Failed to launch server terminal: {e}"

def stop_backend_console():
    """Terminates the spawned backend server process and closes the console window."""
    global _global_server_process
    terminated = False
    if _global_server_process is not None:
        try:
            if _global_server_process.poll() is None:
                _global_server_process.terminate()
                terminated = True
        except Exception:
            pass
        _global_server_process = None
    return terminated

# Register cleanup so terminal closes when Blender exits
atexit.register(stop_backend_console)

def _on_preset_change(self, context):
    """Updates target height to canonical architectural defaults when preset is changed."""
    p = self.preset
    if p in ("furniture", "chair", "table"):
        self.target_height = 0.90
    elif p == "weapon":
        self.target_height = 1.00
    elif p == "character":
        self.target_height = 1.80
    elif p == "vehicle":
        self.target_height = 1.50
    elif p == "prop":
        self.target_height = 0.80

def _on_project_dir_change(self, context):
    """Syncs project_dir changes to global AddonPreferences."""
    pref = get_addon_preferences()
    if pref and self.project_dir:
        pref.project_dir = self.project_dir

# --- Property Group for Add-on State ---
class TripoSRProperties(bpy.types.PropertyGroup):
    # Server & Connectivity
    server_url: StringProperty(
        name="Server URL",
        description="Local FastAPI orchestration server URL",
        default=DEFAULT_SERVER_URL
    )
    auto_start_server: BoolProperty(
        name="Auto-Start Server Console",
        description="Automatically open backend terminal console when generating or when Blender opens",
        default=True
    )
    project_dir: StringProperty(
        name="Repo Root",
        description="Path to local-mesh-ai repository directory",
        subtype='DIR_PATH',
        default="",
        update=_on_project_dir_change
    )
    is_server_online: BoolProperty(
        name="Server Online",
        default=False
    )
    show_advanced_server: BoolProperty(
        name="Show Server Settings",
        default=False
    )

    # Input Settings
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
            ("weapon", "Weapon (0.9m)", "Optimized for weapons; sets reflective metallic hilt & blade"),
            ("prop", "Game Prop (0.5m)", "Optimized for static hard-surface game props"),
            ("character", "Character (1.8m)", "Configured for organic humanoid characters & creatures"),
            ("furniture", "Furniture (1.2m)", "Centered pivot with sturdy ground alignment"),
        ],
        default="weapon",
        update=_on_preset_change
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
        default=0.9,
        min=0.05,
        max=50.0
    )
    
    # Accurate Color & Processing Toggles
    accurate_colors: BoolProperty(
        name="Perceptual Color Transfer",
        description="Perceptual Lab Reinhard color distribution transfer matching reference artwork palette",
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
    retopology_and_bake: BoolProperty(
        name="AAA Quad Retopology & Bake",
        description="Produces clean manifold quads (5K-15K) and bakes Tangent-Space Normal + Albedo maps",
        default=False
    )
    target_faces: IntProperty(
        name="Target Quads",
        description="Target polygon count for retopologized mesh",
        default=8000,
        min=1000,
        max=50000
    )
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

    # Phase 3: Component Hierarchy & Precision Editing
    selected_component_edit_op: EnumProperty(
        name="Edit Operation",
        items=[
            ("scale_axis", "Scale Axis (Height/Width)", "Scales active component along selected axis"),
            ("subdivision", "Subdivision Surface", "Increases mesh density on active component"),
            ("bevel", "Bevel Edges", "Adds chamfered/beveled edge modifier"),
            ("solidify", "Solidify Shell", "Adds parametric thickness"),
        ],
        default="scale_axis"
    )
    edit_axis: EnumProperty(
        name="Axis",
        items=[
            ("X", "X (Width)", "Scale along width"),
            ("Y", "Y (Depth)", "Scale along depth"),
            ("Z", "Z (Height)", "Scale along height"),
        ],
        default="Z"
    )
    edit_factor: FloatProperty(
        name="Factor",
        description="Scaling or strength factor",
        default=1.2,
        min=0.1,
        max=5.0
    )

    # State & Polling
    active_job_id: StringProperty(name="Active Job ID", default="")
    status_message: StringProperty(name="Status", default="Ready")
    stage_name: StringProperty(name="Stage", default="Idle")
    progress_pct: IntProperty(name="Progress", default=0, min=0, max=100)
    is_generating: BoolProperty(name="Is Generating", default=False)
    wireframe_active: BoolProperty(name="Wireframe", default=False)

# --- Timer for Non-Blocking Job Polling ---
def poll_job_timer():
    scene = bpy.context.scene
    props = scene.triposr_props
    if not props.is_generating or not props.active_job_id:
        return None

    try:
        url = f"{props.server_url}/jobs/{props.active_job_id}"
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
                props.is_server_online = True
                
                # Import output mesh with full PBR and viewport shading
                output_mesh = data.get("output_mesh")
                if output_mesh and os.path.exists(output_mesh):
                    _import_and_organize(output_mesh, props, profile_data=data.get("appearance_profile"))
                return None
            elif status in ("FAILED", "CANCELLED"):
                props.is_generating = False
                props.status_message = f"{status}: {data.get('error', 'Unknown')}"
                return None
    except Exception as e:
        props.status_message = f"Connecting to server... ({str(e)[:30]})"

    return 1.0  # Poll every 1 second

def _import_and_organize(filepath, props, profile_data=None):
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

    # 3. Build Calibrated PBR Shading & Color Setup for all imported mesh objects
    for obj in imported:
        if props.pbr_material:
            _apply_in_viewport_pbr(obj, filepath, props, profile_data=profile_data)

    # 4. Viewport Shading Mode: Auto-switch to Material Preview
    if props.auto_switch_shading:
        _enable_viewport_color_shading()

def _inspect_material_slot(mat) -> dict:
    """
    Inspects a single material slot for shader node integrity, textures, and color.
    Detects default white materials, missing textures, unlinked shaders, or valid PBR setups.
    """
    if mat is None:
        return {"status": "INVALID_MISSING", "mat": None, "reason": "Material slot is empty"}

    if not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "Material does not use nodes"}

    # Find Principled BSDF
    bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "No Principled BSDF node"}

    # Check Output Material
    out_node = next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if not out_node or not out_node.inputs.get('Surface') or not out_node.inputs['Surface'].is_linked:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "Output Material Surface is not linked"}

    base_color_inp = bsdf.inputs.get('Base Color')
    if not base_color_inp:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "No Base Color input on BSDF"}

    mat_name_lower = mat.name.lower()
    is_default_named = any(d in mat_name_lower for d in ["defaultmaterial", "material.0", "m_geometry_0", "fallback"])

    # 1. Trace Base Color link
    tex_node = None
    if base_color_inp.is_linked:
        from_node = base_color_inp.links[0].from_node
        if from_node.type == 'TEX_IMAGE':
            tex_node = from_node
        elif from_node.type == 'MIX':
            # Check inputs of mix node for image texture
            for inp in from_node.inputs:
                if inp.is_linked and inp.links[0].from_node.type == 'TEX_IMAGE':
                    tex_node = inp.links[0].from_node
                    # Clean direct bypass: connect texture directly into Base Color
                    mat.node_tree.links.new(tex_node.outputs['Color'], base_color_inp)
                    break
        elif from_node.type in ('VERTEX_COLOR', 'ATTRIBUTE'):
            return {'status': 'VALID_COLORED', 'mat': mat, 'reason': f'Principled BSDF Base Color linked to {from_node.type} {from_node.name}'}

    # If texture node found, check image resource
    if tex_node:
        img = tex_node.image
        if not img or img.size[0] == 0:
            return {"status": "INVALID_BROKEN_TEXTURE", "mat": mat, "reason": f"Texture node '{tex_node.name}' has no image"}
        try:
            pix = img.pixels
            if len(pix) >= 16:
                step = max(1, len(pix) // 64)
                count = max(1, len(pix) // (step * 4))
                r = sum(pix[j] for j in range(0, min(len(pix), step * 4 * count), step * 4)) / count
                g = sum(pix[j+1] for j in range(0, min(len(pix), step * 4 * count), step * 4)) / count
                b = sum(pix[j+2] for j in range(0, min(len(pix), step * 4 * count), step * 4)) / count
                mat.diffuse_color = (r, g, b, 1.0)
                if r >= 0.98 and g >= 0.98 and b >= 0.98:
                    return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "diffuse_rgb": (r, g, b), "reason": "Texture image is entirely pure white"}
                return {"status": "VALID_TEXTURED", "mat": mat, "diffuse_rgb": (r, g, b), "reason": f"Valid texture '{img.name}' ({img.size[0]}x{img.size[1]})"}
        except Exception as e:
            return {"status": "INVALID_BROKEN_TEXTURE", "mat": mat, "reason": f"Error reading texture pixels: {e}"}

    # 2. Check unlinked Base Color default value
    if not base_color_inp.is_linked:
        bc = list(base_color_inp.default_value)
        # Check if neutral unshaded gray or flat white ([0.8, 0.8, 0.8] or [1.0, 1.0, 1.0])
        is_neutral_monochrome = (abs(bc[0] - bc[1]) < 0.025 and abs(bc[1] - bc[2]) < 0.025)
        is_high_luminance = (bc[0] >= 0.75 and bc[1] >= 0.75 and bc[2] >= 0.75)
        is_default_white = is_high_luminance and is_neutral_monochrome

        if is_default_named and (is_default_white or is_high_luminance):
            return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "diffuse_rgb": tuple(bc[:3]), "reason": f"Default untextured white/gray material ({[round(x, 2) for x in bc[:3]]})"}
        if is_default_white and len(mat.node_tree.nodes) <= 3:
            return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "diffuse_rgb": tuple(bc[:3]), "reason": f"Default monochrome white/gray material ({[round(x, 2) for x in bc[:3]]})"}

        mat.diffuse_color = (bc[0], bc[1], bc[2], 1.0)
        return {"status": "VALID_COLORED", "mat": mat, "diffuse_rgb": tuple(bc[:3]), "reason": f"Authored non-white base color ({[round(x, 2) for x in bc[:3]]})"}

    return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "reason": "Base color is linked to non-texture node producing flat appearance"}

def _apply_in_viewport_pbr(obj, mesh_path=None, props=None, profile_data=None):
    """
    Validates all material slots on obj.
    - Preserves valid textured or colored materials.
    - Repairs missing, unlinked, or default white materials using AppearanceProfile.
    - Ensures valid polygon material index assignments.
    - Reports detailed preservation and repair diagnostics.
    """
    report = {
        "object": obj.name if obj else "None",
        "preserved": [],
        "repaired": [],
        "replaced": [],
        "status": "noop"
    }

    if not obj or obj.type != 'MESH':
        report["status"] = "skipped_not_mesh"
        return report

    # Ensure repository root is on sys.path
    root = find_project_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)

    # 1. Inspect existing material slots
    inspections = [_inspect_material_slot(m) for m in obj.data.materials]
    valid_slots = [i for i in inspections if i["status"] in ("VALID_TEXTURED", "VALID_COLORED")]
    invalid_slots = [i for i in inspections if i["status"] not in ("VALID_TEXTURED", "VALID_COLORED")]

    # 2. Resolve AppearanceProfile to know intended materials and structure
    resolved_profile = None
    try:
        from core.appearance_analyzer import AppearanceProfile, appearance_analyzer
        if profile_data:
            if isinstance(profile_data, AppearanceProfile):
                resolved_profile = profile_data
            elif isinstance(profile_data, dict):
                resolved_profile = AppearanceProfile.from_dict(profile_data)
        elif mesh_path:
            p_file = os.path.join(os.path.dirname(mesh_path), "profile.json")
            if not os.path.exists(p_file):
                p_file = os.path.join(os.path.dirname(mesh_path), "reports", "profile.json")
            if os.path.exists(p_file):
                with open(p_file, "r", encoding="utf-8") as f:
                    resolved_profile = AppearanceProfile.from_dict(json.load(f))

        if resolved_profile is None and props and hasattr(props, "prompt_text") and props.prompt_text:
            resolved_profile = appearance_analyzer.analyze_appearance(prompt=props.prompt_text)
    except Exception as e:
        print(f"[Blender Addon] Profile resolution note: {e}")

    needs_multi_material = bool(resolved_profile and len(resolved_profile.materials) > 1)

    # 3. Decision: Can we preserve existing materials?
    has_authored_coverage = any(i.get('status') in ('VALID_TEXTURED', 'VALID_COLORED') for i in valid_slots)
    if len(valid_slots) > 0 and len(invalid_slots) == 0 and has_authored_coverage:
            report["status"] = "preserved"
            report["preserved"] = [i["mat"].name for i in valid_slots]
            print(f"[Blender Addon] Preserved {len(valid_slots)} valid authored PBR material slots on '{obj.name}'.")

            # Ensure polygon material indices are valid
            num_mats = len(obj.data.materials)
            for poly in obj.data.polygons:
                if poly.material_index < 0 or poly.material_index >= num_mats:
                    poly.material_index = 0

            if hasattr(bpy.context.scene, 'view_settings'):
                bpy.context.scene.view_settings.view_transform = 'Standard'
            return report

    # 4. Repair Triggered!
    # Mesh has missing materials, default white materials, or lacks multi-material partitioning
    print(f"[Blender Addon] Object '{obj.name}' requires material repair (Valid: {len(valid_slots)}, Invalid: {len(invalid_slots)}, Needs multi: {needs_multi_material}). Initiating PBR repair...")

    if resolved_profile is None:
        try:
            from core.appearance_analyzer import appearance_analyzer
            resolved_profile = appearance_analyzer.analyze_appearance(
                prompt=props.prompt_text if (props and hasattr(props, "prompt_text") and props.prompt_text) else "traditional dining chair with walnut wood frame and cream cushion"
            )
        except Exception:
            pass

    if resolved_profile:
        from core.material_synthesizer import synthesize_and_apply_materials
        textures_dir = os.path.join(os.path.dirname(mesh_path), "textures") if mesh_path else None
        
        synth_res = synthesize_and_apply_materials(
            obj=obj,
            profile=resolved_profile,
            textures_dir=textures_dir,
            generate_textures=True
        )

        report["status"] = "repaired"
        report["replaced"] = [i["mat"].name for i in invalid_slots if i.get("mat")]
        report["repaired"] = [m.name for m in resolved_profile.materials]
        print(f"[Blender Addon] Material Repair complete on '{obj.name}': {synth_res.get('material_count')} slots synthesized: {synth_res.get('materials')}")
    else:
        # Fallback single material
        report["status"] = "fallback"
        obj.data.materials.clear()
        mat = bpy.data.materials.new(name=f"M_{obj.name}_PBR")
        mat.use_nodes = True
        obj.data.materials.append(mat)
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.24, 0.15, 0.10, 1.0)
            mat.diffuse_color = (0.24, 0.15, 0.10, 1.0)

    # 5. Final index validation: verify every polygon references a valid material slot
    num_mats = len(obj.data.materials)
    if num_mats > 0:
        for poly in obj.data.polygons:
            if poly.material_index < 0 or poly.material_index >= num_mats:
                poly.material_index = 0

    if hasattr(bpy.context.scene, 'view_settings'):
        bpy.context.scene.view_settings.view_transform = 'Standard'

    return report

def _enable_viewport_color_shading():
    """Switches 3D Viewport to Material Preview mode and Solid to Material colors."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        try:
                            space.shading.color_type = 'MATERIAL'
                        except Exception:
                            pass
    if hasattr(bpy.context.scene, 'view_settings'):
        bpy.context.scene.view_settings.view_transform = 'Standard'

def _submit_job_to_server(context, payload):
    """Submits payload to the FastAPI backend and begins non-blocking polling."""
    props = context.scene.triposr_props
    try:
        req = urllib.request.Request(
            f"{props.server_url}/jobs",
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
            props.is_server_online = True
            bpy.app.timers.register(poll_job_timer)
            return True, f"Job submitted successfully (ID: {props.active_job_id})"
    except Exception as e:
        props.is_generating = False
        return False, f"Server request failed: {e}"

# --- Operators ---
class TRIPOSR_OT_StartServer(bpy.types.Operator):
    bl_idname = "triposr.start_server"
    bl_label = "Launch Server Terminal"
    bl_description = "Spawns a dedicated console window running the local orchestration server (run_server.py)"

    def execute(self, context):
        props = context.scene.triposr_props
        if check_server_health(props.server_url):
            props.is_server_online = True
            self.report({'INFO'}, "Backend server is already running and ready!")
            return {'FINISHED'}

        root = find_project_root()
        if not root:
            self.report({'ERROR'}, "Please set 'Repo Root' folder in Server Settings above first.")
            return {'CANCELLED'}

        success, msg = launch_backend_console(root)
        if not success:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        props.status_message = "Launching server console... Please wait."
        self.report({'INFO'}, "Server terminal opened. Waiting for GPU initialization...")

        def _verify_server():
            if check_server_health(props.server_url):
                props.is_server_online = True
                props.status_message = "Server Ready"
                return None
            return 1.0

        bpy.app.timers.register(_verify_server, first_interval=1.5)
        return {'FINISHED'}

class TRIPOSR_OT_StopServer(bpy.types.Operator):
    bl_idname = "triposr.stop_server"
    bl_label = "Stop Server Terminal"
    bl_description = "Terminates the backend server process and closes the terminal console"

    def execute(self, context):
        props = context.scene.triposr_props
        stopped = stop_backend_console()
        props.is_server_online = False
        props.status_message = "Server stopped"
        if stopped:
            self.report({'INFO'}, "Backend server console stopped.")
        else:
            self.report({'INFO'}, "Server process terminated.")
        return {'FINISHED'}

class TRIPOSR_OT_Generate(bpy.types.Operator):
    bl_idname = "triposr.generate_asset"
    bl_label = "⚡ Generate 3D Asset"
    bl_description = "Submit prompt or image to local GPU orchestration server (auto-launches server if offline)"

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
            "retopology_and_bake": props.retopology_and_bake,
            "target_faces": props.target_faces,
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

        # Check if server is running
        if check_server_health(props.server_url):
            props.is_server_online = True
            ok, msg = _submit_job_to_server(context, payload)
            if ok:
                self.report({'INFO'}, msg)
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, msg)
                return {'CANCELLED'}

        # Server is offline
        props.is_server_online = False
        if props.auto_start_server:
            root = find_project_root()
            if not root:
                self.report({'ERROR'}, "Repo Root not found. Please select your 'local-mesh-ai' folder in Server Settings above.")
                return {'CANCELLED'}

            success, msg = launch_backend_console(root)
            if not success:
                self.report({'ERROR'}, f"Could not auto-start backend: {msg}")
                return {'CANCELLED'}

            props.is_generating = True
            props.stage_name = "Starting Server"
            props.progress_pct = 2
            props.status_message = "Auto-launching server console... Waiting for GPU..."
            self.report({'INFO'}, "Server terminal opened. Submitting job as soon as online...")

            attempt = {"count": 0}

            def _wait_and_submit():
                attempt["count"] += 1
                if check_server_health(props.server_url):
                    props.is_server_online = True
                    ok, subm_msg = _submit_job_to_server(context, payload)
                    if ok:
                        props.status_message = "Job Queued on GPU..."
                    else:
                        props.status_message = f"Submission error: {subm_msg}"
                        props.is_generating = False
                    return None
                if attempt["count"] > 25:
                    props.is_generating = False
                    props.status_message = "Server launch timed out. Check the terminal console."
                    return None
                return 1.0

            bpy.app.timers.register(_wait_and_submit, first_interval=2.0)
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Local backend server is offline! Click 'Launch Terminal' above or enable Auto-Start.")
            return {'CANCELLED'}

class TRIPOSR_OT_Cancel(bpy.types.Operator):
    bl_idname = "triposr.cancel_asset"
    bl_label = "Cancel Job"

    def execute(self, context):
        props = context.scene.triposr_props
        if props.active_job_id:
            try:
                urllib.request.urlopen(f"{props.server_url}/jobs/{props.active_job_id}/cancel", data=b"", timeout=3)
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
    bl_description = "Switches the 3D viewport to Material Preview mode to inspect materials"

    def execute(self, context):
        _enable_viewport_color_shading()
        mat_count = 0
        target_objs = context.selected_objects if context.selected_objects else context.scene.objects
        props = context.scene.triposr_props if hasattr(context.scene, "triposr_props") else None
        for obj in target_objs:
            if obj.type == 'MESH':
                # Run inspection & repair to ensure default white materials are resolved
                _apply_in_viewport_pbr(obj, None, props)
                for mat in obj.data.materials:
                    if mat and getattr(mat, "use_nodes", False):
                        mat_count += 1
                        tex_node = next((n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image), None)
                        if tex_node and tex_node.image:
                            img = tex_node.image
                            try:
                                if img.pixels and len(img.pixels) >= 16:
                                    pix = img.pixels
                                    step = max(1, len(pix) // 64)
                                    count = max(1, len(pix) // (step * 4))
                                    r = sum(pix[j] for j in range(0, min(len(pix), step * 4 * count), step * 4)) / count
                                    g = sum(pix[j+1] for j in range(0, min(len(pix), step * 4 * count), step * 4)) / count
                                    b = sum(pix[j+2] for j in range(0, min(len(pix), step * 4 * count), step * 4)) / count
                                    mat.diffuse_color = (r, g, b, 1.0)
                            except Exception:
                                pass
                        else:
                            bsdf = mat.node_tree.nodes.get('Principled BSDF')
                            if bsdf and 'Base Color' in bsdf.inputs and not bsdf.inputs['Base Color'].is_linked:
                                bc = bsdf.inputs['Base Color'].default_value
                                mat.diffuse_color = (bc[0], bc[1], bc[2], 1.0)
        self.report({'INFO'}, f"Switched Viewport to Material Preview mode ({mat_count} material slots active).")
        return {'FINISHED'}

class TRIPOSR_OT_ResetDefaults(bpy.types.Operator):
    bl_idname = "triposr.reset_defaults"
    bl_label = "Reset Settings"

    def execute(self, context):
        props = context.scene.triposr_props
        props.target_height = 0.9
        props.preset = "weapon"
        props.quality = "balanced"
        props.auto_ground = True
        props.sanitize_mesh = True
        props.adaptive_weld = True
        props.pbr_material = True
        props.accurate_colors = True
        props.color_boost = 1.25
        props.auto_start_server = True
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

        # Section 0: Backend Server Status & Auto-Start
        srv_box = layout.box()
        srv_row = srv_box.row(align=True)
        if props.is_server_online:
            srv_row.label(text="Backend Server: 🟢 Online (Port 8000)", icon='CHECKMARK')
            srv_row.operator("triposr.stop_server", text="Stop", icon='CANCEL')
        else:
            srv_row.label(text="Backend Server: 🔴 Offline", icon='CANCEL')
            srv_row.operator("triposr.start_server", text="Launch Terminal", icon='CONSOLE')
        
        opt_row = srv_box.row()
        opt_row.prop(props, "auto_start_server", text="Auto-Launch Server on Generate")
        
        # Collapsible server & path settings
        adv_row = srv_box.row()
        adv_row.prop(props, "show_advanced_server", text="Server & Path Settings", icon='DOWNARROW_HLT' if props.show_advanced_server else 'RIGHTARROW')
        if props.show_advanced_server:
            s_col = srv_box.column()
            s_col.prop(props, "project_dir", text="Repo Root")
            root = find_project_root()
            if root and os.path.exists(os.path.join(root, "run_server.py")):
                s_col.label(text=f"✅ Detected: {os.path.basename(root)}", icon='CHECKMARK')
            else:
                s_col.label(text="⚠️ Please select your 'local-mesh-ai' folder!", icon='ERROR')
            s_col.prop(props, "server_url", text="Server URL")

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

        col.label(text="AAA Game-Ready Pipeline:", icon='SHADING_SOLID')
        g_row3 = col.row()
        g_row3.prop(props, "retopology_and_bake")
        if props.retopology_and_bake:
            col.prop(props, "target_faces", text="Target Quads")

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

        # Section F: Inspection & Quick Viewport Switch
        stat_box = layout.box()
        stat_box.label(text="Quick Tools & Viewport:", icon='TOOL_SETTINGS')
        row_tools = stat_box.row()
        row_tools.operator("triposr.switch_viewport", icon='SHADING_TEXTURE')
        row_tools.operator("triposr.toggle_wireframe", icon='SHADING_WIRE')

        if context.active_object and context.active_object.type == 'MESH':
            mesh = context.active_object.data
            stat_box.label(text=f"Active: {context.active_object.name} ({len(mesh.polygons):,} polys)")

        layout.operator("triposr.reset_defaults", icon='LOOP_BACK')


# --- Phase 3: Component Hierarchy & Precision Editing Operators and UI ---

class TRIPOSR_OT_ApplyComponentEdit(bpy.types.Operator):
    bl_idname = "triposr.apply_component_edit"
    bl_label = "Apply AI Component Edit"
    bl_description = "Executes the selected geometric modification on the active component in the hierarchy"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.triposr_props
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Please select a mesh component first!")
            return {'CANCELLED'}

        op = props.selected_component_edit_op
        factor = props.edit_factor
        axis = props.edit_axis

        if op == "scale_axis":
            if axis == "X":
                obj.scale.x *= factor
            elif axis == "Y":
                obj.scale.y *= factor
            elif axis == "Z":
                obj.scale.z *= factor
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            self.report({'INFO'}, f"Scaled {obj.name} along {axis} by {factor}x")
        elif op == "subdivision":
            mod = obj.modifiers.new(name="AI_Subsurf", type='SUBSURF')
            mod.levels = max(1, int(factor))
            self.report({'INFO'}, f"Added Subsurf modifier (level {mod.levels}) to {obj.name}")
        elif op == "bevel":
            mod = obj.modifiers.new(name="AI_Bevel", type='BEVEL')
            mod.width = 0.02 * factor
            self.report({'INFO'}, f"Added Bevel modifier to {obj.name}")
        elif op == "solidify":
            mod = obj.modifiers.new(name="AI_Solidify", type='SOLIDIFY')
            mod.thickness = 0.02 * factor
            self.report({'INFO'}, f"Added Solidify modifier to {obj.name}")

        return {'FINISHED'}


class TRIPOSR_OT_GroundActiveComponent(bpy.types.Operator):
    bl_idname = "triposr.ground_active_component"
    bl_label = "Snap Component to Ground (Z=0)"
    bl_description = "Translates active component so its lowest point rests precisely on Z=0"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import mathutils
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Please select a mesh component first!")
            return {'CANCELLED'}

        bbox = [obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box]
        min_z = min(v.z for v in bbox)
        obj.location.z -= min_z
        self.report({'INFO'}, f"Grounded {obj.name} to Z=0.0")
        return {'FINISHED'}


class TRIPOSR_PT_ComponentHierarchyPanel(bpy.types.Panel):
    bl_label = "AI Component Hierarchy & Precision Editing"
    bl_idname = "TRIPOSR_PT_component_hierarchy_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AI 3D Generator"
    bl_parent_id = "TRIPOSR_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.triposr_props

        col_hier = layout.box()
        col_hier.label(text="Component Hierarchy Tree:", icon='OUTLINER_OB_MESH')

        master_col = bpy.data.collections.get("AI_Asset_Hierarchy")
        if master_col and master_col.objects:
            for obj in master_col.objects:
                row = col_hier.row(align=True)
                is_active = (context.active_object == obj)
                icon = 'RADIOBUT_ON' if is_active else 'RADIOBUT_OFF'
                label_txt = f"{obj.name}"
                sem = obj.get("ai_semantic_label")
                if sem:
                    label_txt += f" [{sem}]"
                row.label(text=label_txt, icon=icon)
        else:
            col_hier.label(text="No AI_Asset_Hierarchy active.", icon='INFO')

        if context.active_object and context.active_object.type == 'MESH':
            obj = context.active_object
            edit_box = layout.box()
            edit_box.label(text=f"Active Target: {obj.name}", icon='EDITMODE_HLT')
            dims = obj.dimensions
            edit_box.label(text=f"Dimensions: {dims.x:.3f}m x {dims.y:.3f}m x {dims.z:.3f}m")

            edit_box.prop(props, "selected_component_edit_op", text="Action")
            if props.selected_component_edit_op == "scale_axis":
                row_axis = edit_box.row()
                row_axis.prop(props, "edit_axis", expand=True)
                edit_box.prop(props, "edit_factor", text="Scale Factor")
            elif props.selected_component_edit_op in ("bevel", "solidify"):
                edit_box.prop(props, "edit_factor", text="Thickness/Width")
            elif props.selected_component_edit_op == "subdivision":
                edit_box.prop(props, "edit_factor", text="Levels")

            row_btns = edit_box.row(align=True)
            row_btns.operator("triposr.apply_component_edit", text="Apply Edit", icon='CHECKMARK')
            row_btns.operator("triposr.ground_active_component", text="Ground (Z=0)", icon='SNAP_VERTEX')

# --- Background Server Polling Timer ---
def _periodic_server_check():
    """Lightweight background timer checking server status every 3.5 seconds."""
    try:
        if hasattr(bpy.context, "scene") and hasattr(bpy.context.scene, "triposr_props"):
            props = bpy.context.scene.triposr_props
            if not props.is_generating:
                props.is_server_online = check_server_health(props.server_url)
    except Exception:
        pass
    return 3.5

def _startup_server_check():
    """One-time startup check when Blender loads or add-on is enabled."""
    try:
        online = check_server_health()
        if hasattr(bpy.context, "scene") and hasattr(bpy.context.scene, "triposr_props"):
            props = bpy.context.scene.triposr_props
            props.is_server_online = online
            # If server offline and root exists, auto launch console
            root = find_project_root()
            if not online and props.auto_start_server and root:
                launch_backend_console(root)
    except Exception:
        pass
    return None

# --- Registration ---
classes = (
    TripoSRAddonPreferences,
    TripoSRProperties,
    TRIPOSR_OT_StartServer,
    TRIPOSR_OT_StopServer,
    TRIPOSR_OT_Generate,
    TRIPOSR_OT_Cancel,
    TRIPOSR_OT_ToggleWireframe,
    TRIPOSR_OT_SwitchViewport,
    TRIPOSR_OT_ResetDefaults,
    TRIPOSR_OT_ApplyComponentEdit,
    TRIPOSR_OT_GroundActiveComponent,
    TRIPOSR_PT_MainPanel,
    TRIPOSR_PT_ComponentHierarchyPanel
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
    
    # Initialize Repo Root from preferences if available
    try:
        pref = get_addon_preferences()
        root = find_project_root()
        if pref and root and not pref.project_dir:
            pref.project_dir = root
        if hasattr(bpy.context, "scene") and hasattr(bpy.context.scene, "triposr_props"):
            if root:
                bpy.context.scene.triposr_props.project_dir = root
    except Exception:
        pass

    if not bpy.app.timers.is_registered(_periodic_server_check):
        bpy.app.timers.register(_periodic_server_check, first_interval=2.0)
    if not bpy.app.timers.is_registered(_startup_server_check):
        bpy.app.timers.register(_startup_server_check, first_interval=1.0)

def unregister():
    # Terminate backend console if running
    try:
        stop_backend_console()
    except Exception:
        pass

    if bpy.app.timers.is_registered(_periodic_server_check):
        bpy.app.timers.unregister(_periodic_server_check)
    if bpy.app.timers.is_registered(_startup_server_check):
        bpy.app.timers.unregister(_startup_server_check)
        
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
