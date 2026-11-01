# -*- coding: utf-8 -*-
# GNU General Public License v3.0 (GPLv3)
# Copyright (C) 2024-2026 Local Mesh AI Contributors & Blender Foundation
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

bl_info = {
    "name": "Local Mesh AI — Blender Studio",
    "author": "Antigravity Architecture & Kernlab-001",
    "version": (5, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > LOCAL MESH AI",
    "description": "Unified local-first AI 3D asset generation, scene synchronization & interactive editing studio",
    "category": "LOCAL MESH AI",
    "license": "GPLv3",
}

import bpy
import os
import sys
import json
import math
import time
import queue
import threading
import subprocess
import webbrowser
import atexit
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from bpy.props import (
    StringProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    PointerProperty
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
_global_server_process = None

# Background polling shared state (thread-safe)
_job_state = {
    "job_id": None,
    "status": "IDLE",
    "stage": "Idle",
    "progress": 0,
    "message": "Ready",
    "output_mesh": None,
    "error": None,
    "start_time": 0.0,
    "elapsed": 0.0,
    "is_active": False,
    "new_completion": False
}
_job_lock = threading.Lock()


# ==============================================================================
# Add-on Preferences
# ==============================================================================
class LocalMeshAddonPreferences(bpy.types.AddonPreferences):
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
        layout.label(text="Local Mesh AI Studio — Global Preferences", icon='PREFERENCES')
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


# ==============================================================================
# Server Health, Discovery & Process Helpers
# ==============================================================================
def check_server_health(url=DEFAULT_SERVER_URL, timeout=0.8):
    """Checks if the local backend orchestration server is responding."""
    try:
        t0 = time.time()
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.time() - t0) * 1000.0
            if resp.status == 200:
                data = {}
                try:
                    data = json.loads(resp.read().decode('utf-8'))
                except Exception:
                    pass
                return True, latency_ms, data
    except Exception as e:
        return False, 0.0, str(e)
    return False, 0.0, "Unknown response"


def find_project_root():
    """Intelligently locates the local-mesh-ai project root directory."""
    pref = get_addon_preferences()
    if pref and pref.project_dir and os.path.exists(os.path.join(pref.project_dir, "run_server.py")):
        return pref.project_dir

    try:
        if hasattr(bpy.context, "scene") and hasattr(bpy.context.scene, "local_mesh_props"):
            custom = bpy.context.scene.local_mesh_props.project_dir
            if custom and os.path.exists(os.path.join(custom, "run_server.py")):
                if pref:
                    pref.project_dir = custom
                return custom
    except Exception:
        pass

    env_dir = os.environ.get("LOCAL_MESH_AI_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "run_server.py")):
        return env_dir

    script_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..")),
        script_dir,
        os.getcwd(),
        os.path.expanduser(r"~\local-mesh-ai"),
        r"E:\local-mesh-ai-v6.0",
        r"E:\GPUPLAT",
        r"C:\local-mesh-ai",
        r"D:\local-mesh-ai"
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "run_server.py")):
            return path
    return ""


def launch_backend_console(project_dir=None):
    """Spawns run_server.py in a dedicated external terminal console window."""
    if not project_dir:
        project_dir = find_project_root()
    if not project_dir or not os.path.exists(project_dir):
        return False, "Project directory not found. Please set 'Repo Root' in Preferences."

    server_script = os.path.join(project_dir, "run_server.py")
    if not os.path.exists(server_script):
        return False, f"'run_server.py' not found in {project_dir}"

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
            creation_flags = subprocess.CREATE_NEW_CONSOLE

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
    """Terminates the spawned backend server process."""
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

atexit.register(stop_backend_console)


# ==============================================================================
# Preset Defaults Callbacks
# ==============================================================================
def _on_preset_change(self, context):
    """Updates target height to canonical architectural defaults."""
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
        self.target_height = 0.50

def _on_project_dir_change(self, context):
    """Syncs project_dir changes to global AddonPreferences."""
    pref = get_addon_preferences()
    if pref and self.project_dir:
        pref.project_dir = self.project_dir


# ==============================================================================
# Property Group for Local Mesh AI State
# ==============================================================================
class LocalMeshProperties(bpy.types.PropertyGroup):
    # --- Connection & Server ---
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
    is_server_online: BoolProperty(name="Server Online", default=False)
    server_latency_ms: FloatProperty(name="Server Latency (ms)", default=0.0)
    server_gpu_info: StringProperty(name="GPU Info", default="Detecting...")
    server_vram_info: StringProperty(name="VRAM Info", default="Detecting...")
    show_advanced_server: BoolProperty(name="Show Server Settings", default=False)

    # --- Generation Inputs ---
    input_mode: EnumProperty(
        name="Input Mode",
        items=[
            ("prompt", "Text-to-3D", "Generate concept artwork & 3D asset from text prompt"),
            ("image", "Image-to-3D", "Reconstruct 3D asset from an existing 2D image"),
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
    generation_route: EnumProperty(
        name="Route",
        description="Active neural route dispatched by pipeline",
        items=[
            ("text_to_3d_pipeline", "Text-to-3D (SD-Turbo + TripoSR)", "Text prompt to concept diffusion followed by 3D reconstruction"),
            ("text_to_3d_direct", "Text-to-3D Direct (Shap-E)", "Direct implicit text-to-3D generation"),
            ("image_to_3d_triposr", "Image-to-3D (TripoSR)", "Single reference image direct mesh reconstruction"),
        ],
        default="text_to_3d_pipeline"
    )
    hardware_profile: EnumProperty(
        name="Hardware Profile",
        description="GPU memory budget and execution profile",
        items=[
            ("balanced", "Balanced (6 GB RTX 3050)", "Optimal memory vs speed for 6GB VRAM GPUs"),
            ("low_memory", "Low Memory (4 GB)", "Strict aggressive VRAM caching & chunking"),
            ("quality", "Quality (8+ GB)", "Full precision and maximum resolution"),
        ],
        default="balanced"
    )
    preset: EnumProperty(
        name="Asset Preset",
        items=[
            ("weapon", "Weapon (1.0m)", "Optimized for weapons; metallic edge treatment"),
            ("prop", "Game Prop (0.5m)", "Optimized for static hard-surface game props"),
            ("character", "Character (1.8m)", "Configured for organic humanoid characters & creatures"),
            ("furniture", "Furniture (0.9m)", "Centered pivot with sturdy ground alignment"),
            ("vehicle", "Vehicle (1.5m)", "Scale & bounding profile tuned for vehicles"),
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
        default=1.0,
        min=0.05,
        max=50.0
    )
    remove_bg: BoolProperty(name="Remove Background", default=True)
    sanitize_mesh: BoolProperty(name="Sanitize Geometry", default=True)
    adaptive_weld: BoolProperty(name="Adaptive Welding", default=True)
    pbr_material: BoolProperty(name="Generate PBR Shaders", default=True)
    accurate_colors: BoolProperty(name="Perceptual Color Transfer", default=True)
    sharp_overlay: BoolProperty(name="Sharp Texture Overlay", default=True)
    color_boost: FloatProperty(name="Color Vibrancy", default=1.25, min=0.5, max=2.0)
    auto_switch_shading: BoolProperty(name="Auto Viewport Color Shading", default=True)
    auto_ground: BoolProperty(name="Smart Grounding (Z=0)", default=True)
    center_pivot: BoolProperty(name="Center Pivot to Base", default=True)
    retopology_and_bake: BoolProperty(name="AAA Quad Retopology & Bake", default=False)
    target_faces: IntProperty(name="Target Quads", default=8000, min=1000, max=50000)
    generate_lods: BoolProperty(name="Generate LODs", default=False)
    auto_rig: BoolProperty(name="Attach Rigify Metarig", default=False)

    # --- Scene Destination ---
    dest_mode: EnumProperty(
        name="Destination",
        items=[
            ("new_col", "New Collection", "Creates or links into dedicated asset collection"),
            ("active_col", "Active Collection", "Inserts directly into current collection"),
        ],
        default="new_col"
    )
    collection_name: StringProperty(name="Collection Name", default="AI_Generated_Assets")

    # --- Job Monitor & Polling ---
    active_job_id: StringProperty(name="Active Job ID", default="")
    status_message: StringProperty(name="Status", default="Ready")
    stage_name: StringProperty(name="Stage", default="Idle")
    progress_pct: IntProperty(name="Progress", default=0, min=0, max=100)
    job_status: StringProperty(name="Job Status", default="IDLE")
    job_elapsed_sec: FloatProperty(name="Elapsed Time (s)", default=0.0)
    is_generating: BoolProperty(name="Is Generating", default=False)
    last_completed_mesh: StringProperty(name="Completed Mesh Path", default="")
    last_completed_polycount: IntProperty(name="Last Polycount", default=0)

    # --- Asset Catalogue ---
    selected_asset_path: StringProperty(name="Selected Asset", subtype='FILE_PATH', default="")
    asset_search_filter: StringProperty(name="Filter", default="")

    # --- Scene Sync State ---
    sync_status: StringProperty(name="Sync Status", default="Not synchronized")
    last_sync_time: StringProperty(name="Last Sync", default="Never")
    synced_object_count: IntProperty(name="Synced Objects", default=0)

    # --- Component Hierarchy & Editing (Phase 3 retained) ---
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
    edit_factor: FloatProperty(name="Factor", default=1.2, min=0.1, max=5.0)
    wireframe_active: BoolProperty(name="Wireframe", default=False)


# Backwards compatibility alias
TripoSRProperties = LocalMeshProperties


# ==============================================================================
# Non-Blocking Background Job Polling Worker & Main-Thread Timer
# ==============================================================================
def _background_poll_worker(server_url, job_id):
    """
    Background worker thread: repeatedly queries backend /jobs/{job_id}
    without blocking Blender's main event loop.
    Safe: Never calls bpy from this thread!
    """
    global _job_state
    url = f"{server_url}/jobs/{job_id}"
    t_start = time.time()

    while True:
        with _job_lock:
            if not _job_state["is_active"] or _job_state["job_id"] != job_id:
                break

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                status = data.get("status", "PROCESSING")
                stage = data.get("stage", "Processing")
                progress = int(data.get("progress_pct", 0))
                output_mesh = data.get("output_mesh")
                error = data.get("error")

                with _job_lock:
                    _job_state["status"] = status
                    _job_state["stage"] = stage
                    _job_state["progress"] = progress
                    _job_state["elapsed"] = time.time() - t_start
                    _job_state["output_mesh"] = output_mesh
                    _job_state["error"] = error

                    if status in ("COMPLETED", "FAILED", "CANCELLED"):
                        _job_state["is_active"] = False
                        if status == "COMPLETED":
                            _job_state["new_completion"] = True
                        break
        except Exception as e:
            with _job_lock:
                _job_state["message"] = f"Network poll note: {str(e)[:30]}"

        time.sleep(0.4)


def poll_active_job_timer():
    """
    Executes exclusively on Blender's MAIN thread via bpy.app.timers.
    Safely transfers background job state to bpy properties and triggers
    automatic asset import upon completion.
    """
    scene = bpy.context.scene
    props = getattr(scene, "local_mesh_props", None) or getattr(scene, "triposr_props", None)
    if not props:
        return None

    with _job_lock:
        job_id = _job_state["job_id"]
        status = _job_state["status"]
        stage = _job_state["stage"]
        progress = _job_state["progress"]
        elapsed = _job_state["elapsed"]
        output_mesh = _job_state["output_mesh"]
        error = _job_state["error"]
        new_completion = _job_state["new_completion"]
        is_active = _job_state["is_active"]

        props.stage_name = stage
        props.progress_pct = progress
        props.job_status = status
        props.job_elapsed_sec = elapsed

        if new_completion:
            _job_state["new_completion"] = False
            props.is_generating = False
            props.status_message = f"Generation Completed in {elapsed:.1f}s!"
            props.stage_name = "Completed"
            props.progress_pct = 100
            props.last_completed_mesh = output_mesh or ""

            # Auto import on Blender main thread
            if output_mesh and os.path.exists(output_mesh):
                _import_and_organize(output_mesh, props)
            return None

        if status == "FAILED":
            props.is_generating = False
            props.status_message = f"Failed: {error or 'Unknown error'}"
            return None

        if status == "CANCELLED":
            props.is_generating = False
            props.status_message = "Generation Cancelled."
            return None

    if not is_active:
        return None

    return 0.3  # Poll main thread timer every 300ms


# ==============================================================================
# Mesh Import & Organization (Main Thread)
# ==============================================================================
def _import_and_organize(filepath, props, profile_data=None):
    """
    Imports mesh, links into dedicated AI_Generated_Assets collection,
    applies calibrated PBR shader, centers pivot, and switches to Material Preview.
    """
    if not os.path.exists(filepath):
        print(f"[Blender Studio] Output file not found: {filepath}")
        return

    # 1. Collection handling
    col_name = props.collection_name if props.dest_mode == "new_col" else None
    if col_name:
        col = bpy.data.collections.get(col_name)
        if not col:
            col = bpy.data.collections.new(col_name)
            bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.context.collection

    # 2. Deselect all
    for o in bpy.context.selected_objects:
        o.select_set(False)

    # 3. Import GLTF / OBJ
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".gltf", ".glb"):
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif ext == ".obj":
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=filepath)
            else:
                bpy.ops.import_scene.obj(filepath=filepath)
    except Exception as e:
        print(f"[Blender Studio] Import error: {e}")
        return

    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported:
        return

    main_obj = imported[0]
    total_polys = sum(len(o.data.polygons) for o in imported)
    props.last_completed_polycount = total_polys

    # Link to destination collection
    for obj in imported:
        if col and obj.name not in col.objects:
            col.objects.link(obj)
            try:
                bpy.context.scene.collection.objects.unlink(obj)
            except Exception:
                pass

    bpy.context.view_layer.objects.active = main_obj
    main_obj.select_set(True)

    # 4. Apply PBR shaders & accurate colors
    for obj in imported:
        if props.pbr_material:
            _apply_in_viewport_pbr(obj, filepath, props, profile_data=profile_data)

    # 5. Smart Grounding / Alignment if requested
    if props.auto_ground or props.center_pivot:
        _ground_and_center_object(main_obj, ground=props.auto_ground, center=props.center_pivot)

    # 6. Auto-switch shading to Material Preview
    if props.auto_switch_shading:
        _enable_viewport_color_shading()

    # 7. Frame selected in active 3D view
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx = bpy.context.copy()
            ctx['area'] = area
            ctx['region'] = area.regions[-1]
            try:
                bpy.ops.view3d.view_selected(ctx)
            except Exception:
                pass


def _ground_and_center_object(obj, ground=True, center=True):
    """Sets pivot to bottom base and snaps Z to 0.0."""
    try:
        import mathutils
        bbox = [obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box]
        min_z = min(v.z for v in bbox)
        if ground:
            obj.location.z -= min_z
    except Exception:
        pass


# ==============================================================================
# PBR Material Inspection & Repair (Preserved from Phase 4)
# ==============================================================================
def _inspect_material_slot(mat) -> dict:
    """Inspects a single material slot for shader node integrity."""
    if mat is None:
        return {"status": "INVALID_MISSING", "mat": None, "reason": "Material slot is empty"}

    if not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "Material does not use nodes"}

    bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "No Principled BSDF node"}

    out_node = next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if not out_node or not out_node.inputs.get('Surface') or not out_node.inputs['Surface'].is_linked:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "Output Material Surface is not linked"}

    base_color_inp = bsdf.inputs.get('Base Color')
    if not base_color_inp:
        return {"status": "INVALID_UNLINKED", "mat": mat, "reason": "No Base Color input on BSDF"}

    mat_name_lower = mat.name.lower()
    is_default_named = any(d in mat_name_lower for d in ["defaultmaterial", "material.0", "m_geometry_0", "fallback"])

    tex_node = None
    if base_color_inp.is_linked:
        from_node = base_color_inp.links[0].from_node
        if from_node.type == 'TEX_IMAGE':
            tex_node = from_node
        elif from_node.type == 'MIX':
            for inp in from_node.inputs:
                if inp.is_linked and inp.links[0].from_node.type == 'TEX_IMAGE':
                    tex_node = inp.links[0].from_node
                    mat.node_tree.links.new(tex_node.outputs['Color'], base_color_inp)
                    break
        elif from_node.type in ('VERTEX_COLOR', 'ATTRIBUTE'):
            return {'status': 'VALID_COLORED', 'mat': mat, 'reason': f'Principled BSDF Base Color linked to {from_node.type} {from_node.name}'}

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

    if not base_color_inp.is_linked:
        bc = list(base_color_inp.default_value)
        is_neutral_monochrome = (abs(bc[0] - bc[1]) < 0.025 and abs(bc[1] - bc[2]) < 0.025)
        is_high_luminance = (bc[0] >= 0.75 and bc[1] >= 0.75 and bc[2] >= 0.75)
        is_default_white = is_high_luminance and is_neutral_monochrome

        if is_default_named and (is_default_white or is_high_luminance):
            return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "diffuse_rgb": tuple(bc[:3]), "reason": f"Default untextured white/gray material ({[round(x, 2) for x in bc[:3]]})"}
        if is_default_white and len(mat.node_tree.nodes) <= 3:
            return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "diffuse_rgb": tuple(bc[:3]), "reason": f"Default monochrome white/gray material ({[round(x, 2) for x in bc[:3]]})"}

        mat.diffuse_color = (bc[0], bc[1], bc[2], 1.0)
        return {"status": "VALID_COLORED", "mat": mat, "diffuse_rgb": tuple(bc[:3]), "reason": f"Authored non-white base color ({[round(x, 2) for x in bc[:3]]})"}

    return {"status": "INVALID_WHITE_OR_DEFAULT", "mat": mat, "reason": "Base color is linked to non-texture node"}


def _apply_in_viewport_pbr(obj, mesh_path=None, props=None, profile_data=None):
    """Validates and repairs material slots on obj using AppearanceProfile."""
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

    root = find_project_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)

    inspections = [_inspect_material_slot(m) for m in obj.data.materials]
    valid_slots = [i for i in inspections if i["status"] in ("VALID_TEXTURED", "VALID_COLORED")]
    invalid_slots = [i for i in inspections if i["status"] not in ("VALID_TEXTURED", "VALID_COLORED")]

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
        print(f"[Blender Studio] Profile resolution note: {e}")

    has_authored_coverage = any(i.get('status') in ('VALID_TEXTURED', 'VALID_COLORED') for i in valid_slots)
    if len(valid_slots) > 0 and len(invalid_slots) == 0 and has_authored_coverage:
        report["status"] = "preserved"
        report["preserved"] = [i["mat"].name for i in valid_slots]
        num_mats = len(obj.data.materials)
        for poly in obj.data.polygons:
            if poly.material_index < 0 or poly.material_index >= num_mats:
                poly.material_index = 0
        return report

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
    else:
        report["status"] = "fallback"
        obj.data.materials.clear()
        mat = bpy.data.materials.new(name=f"M_{obj.name}_PBR")
        mat.use_nodes = True
        obj.data.materials.append(mat)
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.24, 0.15, 0.10, 1.0)
            mat.diffuse_color = (0.24, 0.15, 0.10, 1.0)

    num_mats = len(obj.data.materials)
    if num_mats > 0:
        for poly in obj.data.polygons:
            if poly.material_index < 0 or poly.material_index >= num_mats:
                poly.material_index = 0

    return report


def _enable_viewport_color_shading():
    """Switches 3D Viewport to Material Preview mode."""
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


# ==============================================================================
# OPERATORS: Connection & Health (Panel 1)
# ==============================================================================
class LOCALMESH_OT_RefreshConnection(bpy.types.Operator):
    bl_idname = "localmesh.refresh_connection"
    bl_label = "Refresh Connection"
    bl_description = "Pings the backend server to check health, latency, and GPU status"

    def execute(self, context):
        props = context.scene.local_mesh_props
        online, latency, data = check_server_health(props.server_url)
        props.is_server_online = online
        props.server_latency_ms = latency
        if online and isinstance(data, dict):
            props.status_message = f"Connected ({latency:.1f}ms)"
            gpu = data.get("gpu", {})
            if isinstance(gpu, dict):
                props.server_gpu_info = gpu.get("device_name", "GPU Active")
                avail = gpu.get("available_vram_gb", 0)
                tot = gpu.get("total_vram_gb", 0)
                props.server_vram_info = f"{avail:.1f} / {tot:.1f} GB"
            self.report({'INFO'}, f"Server online ({latency:.1f}ms)")
        else:
            props.status_message = "Server offline"
            props.server_gpu_info = "Offline"
            props.server_vram_info = "Offline"
            self.report({'WARNING'}, "Local backend server is offline")
        return {'FINISHED'}


class LOCALMESH_OT_DisconnectBackend(bpy.types.Operator):
    bl_idname = "localmesh.disconnect_backend"
    bl_label = "Disconnect"
    bl_description = "Disconnects session state from backend server"

    def execute(self, context):
        props = context.scene.local_mesh_props
        props.is_server_online = False
        props.status_message = "Disconnected"
        self.report({'INFO'}, "Disconnected from backend.")
        return {'FINISHED'}


class LOCALMESH_OT_StartServer(bpy.types.Operator):
    bl_idname = "localmesh.start_server"
    bl_label = "Start Backend Terminal"
    bl_description = "Spawns dedicated terminal console running run_server.py"

    def execute(self, context):
        props = context.scene.local_mesh_props
        online, _, _ = check_server_health(props.server_url)
        if online:
            props.is_server_online = True
            self.report({'INFO'}, "Server is already running!")
            return {'FINISHED'}

        root = find_project_root()
        if not root:
            self.report({'ERROR'}, "Repo Root not found. Set it in Preferences.")
            return {'CANCELLED'}

        ok, msg = launch_backend_console(root)
        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        props.status_message = "Launching server console..."
        self.report({'INFO'}, "Server console opened. Initializing...")

        def _verify():
            online, lat, _ = check_server_health(props.server_url)
            if online:
                props.is_server_online = True
                props.server_latency_ms = lat
                props.status_message = "Server Ready"
                return None
            return 1.0

        bpy.app.timers.register(_verify, first_interval=1.5)
        return {'FINISHED'}


class LOCALMESH_OT_StopServer(bpy.types.Operator):
    bl_idname = "localmesh.stop_server"
    bl_label = "Stop Backend"
    bl_description = "Terminates the backend server console"

    def execute(self, context):
        props = context.scene.local_mesh_props
        stop_backend_console()
        props.is_server_online = False
        props.status_message = "Server stopped"
        self.report({'INFO'}, "Backend server stopped.")
        return {'FINISHED'}


class LOCALMESH_OT_OpenDashboard(bpy.types.Operator):
    bl_idname = "localmesh.open_dashboard"
    bl_label = "Open Dashboard"
    bl_description = "Opens FastAPI interactive API documentation in web browser"

    def execute(self, context):
        props = context.scene.local_mesh_props
        url = f"{props.server_url}/docs"
        webbrowser.open(url)
        self.report({'INFO'}, f"Opened {url}")
        return {'FINISHED'}


class LOCALMESH_OT_RunDiagnostics(bpy.types.Operator):
    bl_idname = "localmesh.run_diagnostics"
    bl_label = "Run Diagnostics"
    bl_description = "Queries backend for hardware profiles and route readiness"

    def execute(self, context):
        props = context.scene.local_mesh_props
        try:
            req = urllib.request.Request(f"{props.server_url}/api/v1/hardware/profiles")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                telemetry = data.get("_telemetry", {})
                device = telemetry.get("device_name", "GPU")
                avail_vram = telemetry.get("available_vram_gb", 0)
                recom = telemetry.get("recommended_profile", "balanced")
                self.report({'INFO'}, f"GPU: {device} | Avail VRAM: {avail_vram:.2f}GB | Recommended: {recom}")
                props.status_message = f"Diagnostics: {recom.upper()} mode recommended"
        except Exception as e:
            self.report({'ERROR'}, f"Diagnostics failed: {e}")
        return {'FINISHED'}


# ==============================================================================
# OPERATORS: Generation (Panel 2 & 3)
# ==============================================================================
class LOCALMESH_OT_StageImage(bpy.types.Operator):
    bl_idname = "localmesh.stage_image"
    bl_label = "Stage Selected Image"
    bl_description = "Safely stages the selected reference image via /api/v1/stage-image"

    def execute(self, context):
        props = context.scene.local_mesh_props
        img_path = bpy.path.abspath(props.image_path)
        if not img_path or not os.path.exists(img_path):
            self.report({'ERROR'}, "Please select a valid image file first.")
            return {'CANCELLED'}

        try:
            import mimetypes
            content_type = mimetypes.guess_type(img_path)[0] or "application/octet-stream"
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            boundary = "----WebKitFormBoundaryLocalMeshBlender"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(img_path)}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

            req = urllib.request.Request(
                f"{props.server_url}/api/v1/stage-image",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                staged_path = data.get("staged_path")
                dims = f"{data.get('width')}x{data.get('height')}"
                self.report({'INFO'}, f"Staged successfully ({dims}): {os.path.basename(staged_path)}")
                props.status_message = f"Image Staged ({dims})"
        except Exception as e:
            self.report({'ERROR'}, f"Failed to stage image: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class LOCALMESH_OT_GenerateAsset(bpy.types.Operator):
    bl_idname = "localmesh.generate_asset"
    bl_label = "⚡ Generate 3D Asset"
    bl_description = "Dispatches generation job to local GPU backend via /jobs"

    def execute(self, context):
        props = context.scene.local_mesh_props
        online, _, _ = check_server_health(props.server_url)

        if not online:
            if props.auto_start_server:
                bpy.ops.localmesh.start_server()
                self.report({'WARNING'}, "Backend offline. Auto-launching terminal... Retry in 5 seconds.")
                return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Backend server is offline! Start server first.")
                return {'CANCELLED'}

        # Build payload matching Phase 4 API contracts
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
            "auto_rig": props.auto_rig,
            "hardware_profile": props.hardware_profile,
            "route_override": props.generation_route
        }

        if props.input_mode == "prompt":
            prompt = props.prompt_text.strip()
            if not prompt:
                self.report({'ERROR'}, "Please enter a descriptive prompt.")
                return {'CANCELLED'}
            payload["prompt"] = prompt
        else:
            img_path = bpy.path.abspath(props.image_path)
            if not img_path or not os.path.exists(img_path):
                self.report({'ERROR'}, "Please select a valid reference image.")
                return {'CANCELLED'}
            payload["image_path"] = os.path.abspath(img_path)

        try:
            req = urllib.request.Request(
                f"{props.server_url}/jobs",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                job_id = data.get("job_id")
                props.active_job_id = job_id
                props.is_generating = True
                props.status_message = "Job Queued on GPU..."
                props.stage_name = "Queued"
                props.progress_pct = 5
                props.job_status = "QUEUED"
                props.job_elapsed_sec = 0.0

                # Initialize background polling thread & register main-thread timer
                global _job_state
                with _job_lock:
                    _job_state["job_id"] = job_id
                    _job_state["status"] = "QUEUED"
                    _job_state["stage"] = "Queued"
                    _job_state["progress"] = 5
                    _job_state["elapsed"] = 0.0
                    _job_state["output_mesh"] = None
                    _job_state["error"] = None
                    _job_state["is_active"] = True
                    _job_state["new_completion"] = False

                t = threading.Thread(
                    target=_background_poll_worker,
                    args=(props.server_url, job_id),
                    daemon=True
                )
                t.start()

                bpy.app.timers.register(poll_active_job_timer, first_interval=0.3)
                self.report({'INFO'}, f"Dispatched Job {job_id[:8]} to GPU pipeline")
                return {'FINISHED'}
        except Exception as e:
            props.is_generating = False
            self.report({'ERROR'}, f"Failed to submit job: {e}")
            return {'CANCELLED'}


class LOCALMESH_OT_CancelJob(bpy.types.Operator):
    bl_idname = "localmesh.cancel_job"
    bl_label = "Cancel Generation"
    bl_description = "Cancels active generation job on backend"

    def execute(self, context):
        props = context.scene.local_mesh_props
        if props.active_job_id:
            try:
                req = urllib.request.Request(f"{props.server_url}/jobs/{props.active_job_id}/cancel", data=b"")
                urllib.request.urlopen(req, timeout=3.0)
                props.is_generating = False
                props.status_message = "Job Cancelled"
                props.job_status = "CANCELLED"
                with _job_lock:
                    _job_state["is_active"] = False
                    _job_state["status"] = "CANCELLED"
                self.report({'INFO'}, "Generation cancelled.")
            except Exception as e:
                self.report({'WARNING'}, f"Cancel request note: {e}")
        return {'FINISHED'}


# ==============================================================================
# OPERATORS: Asset Catalogue (Panel 4)
# ==============================================================================
class LOCALMESH_OT_RefreshAssets(bpy.types.Operator):
    bl_idname = "localmesh.refresh_assets"
    bl_label = "Refresh Catalogue"
    bl_description = "Queries generated assets from backend and local output directory"

    def execute(self, context):
        props = context.scene.local_mesh_props
        root = find_project_root()
        count = 0
        if root:
            out_meshes = Path(root) / "output" / "meshes"
            if out_meshes.exists():
                found = list(out_meshes.glob("*/*.glb")) + list(out_meshes.glob("*/*.obj"))
                count = len(found)
                if found:
                    props.selected_asset_path = str(found[-1])
        props.status_message = f"Found {count} asset(s) in output"
        self.report({'INFO'}, f"Asset catalogue refreshed ({count} assets found)")
        return {'FINISHED'}


class LOCALMESH_OT_ImportAsset(bpy.types.Operator):
    bl_idname = "localmesh.import_asset"
    bl_label = "Import into Scene"
    bl_description = "Imports selected asset file into Blender scene"

    def execute(self, context):
        props = context.scene.local_mesh_props
        path = bpy.path.abspath(props.selected_asset_path)
        if not path or not os.path.exists(path):
            self.report({'ERROR'}, "Selected asset file does not exist.")
            return {'CANCELLED'}

        _import_and_organize(path, props)
        self.report({'INFO'}, f"Imported {os.path.basename(path)}")
        return {'FINISHED'}


class LOCALMESH_OT_OpenOutputFolder(bpy.types.Operator):
    bl_idname = "localmesh.open_output_folder"
    bl_label = "Open Output Directory"
    bl_description = "Opens the generated assets folder in File Explorer"

    def execute(self, context):
        root = find_project_root()
        out_dir = Path(root) / "output" if root else Path.cwd() / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(out_dir))
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])
        return {'FINISHED'}


# ==============================================================================
# OPERATORS: Scene Synchronization (Panel 6)
# ==============================================================================
class LOCALMESH_OT_SyncScene(bpy.types.Operator):
    bl_idname = "localmesh.sync_scene"
    bl_label = "Sync Scene to Backend"
    bl_description = "Serializes scene objects and metadata and pushes to /api/v1/blender/sync-scene"

    def execute(self, context):
        props = context.scene.local_mesh_props
        objects_meta = []
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                mesh = obj.data
                mat_names = [m.name for m in mesh.materials if m]
                objects_meta.append({
                    "name": obj.name,
                    "type": obj.type,
                    "location": [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
                    "rotation": [float(obj.rotation_euler.x), float(obj.rotation_euler.y), float(obj.rotation_euler.z)],
                    "scale": [float(obj.scale.x), float(obj.scale.y), float(obj.scale.z)],
                    "dimensions": [float(obj.dimensions.x), float(obj.dimensions.y), float(obj.dimensions.z)],
                    "poly_count": len(mesh.polygons),
                    "material_names": mat_names
                })

        payload = {
            "scene_name": context.scene.name,
            "objects": objects_meta,
            "units": context.scene.unit_settings.system
        }

        try:
            req = urllib.request.Request(
                f"{props.server_url}/api/v1/blender/sync-scene",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                synced_cnt = data.get("synced_objects", len(objects_meta))
                props.synced_object_count = synced_cnt
                now_str = time.strftime("%H:%M:%S")
                props.last_sync_time = now_str
                props.sync_status = f"✅ Synced ({synced_cnt} objects at {now_str})"
                self.report({'INFO'}, f"Synced {synced_cnt} scene objects to backend")
                return {'FINISHED'}
        except Exception as e:
            props.sync_status = f"❌ Sync failed: {str(e)[:30]}"
            self.report({'ERROR'}, f"Failed to sync scene: {e}")
            return {'CANCELLED'}


class LOCALMESH_OT_PullSceneStatus(bpy.types.Operator):
    bl_idname = "localmesh.pull_scene_status"
    bl_label = "Pull Backend Scene Status"
    bl_description = "Retrieves current synchronized scene state from /api/v1/blender/scene-status"

    def execute(self, context):
        props = context.scene.local_mesh_props
        try:
            req = urllib.request.Request(f"{props.server_url}/api/v1/blender/scene-status")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                cnt = data.get("object_count", 0)
                ver = data.get("scene_version", 1)
                t_str = data.get("last_sync", "unknown")
                props.sync_status = f"Backend holds v{ver} ({cnt} objects, {t_str})"
                self.report({'INFO'}, f"Backend scene v{ver} contains {cnt} registered objects")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to query scene status: {e}")
        return {'FINISHED'}


# ==============================================================================
# OPERATORS: Interactive Mesh Controls (Panel 7)
# ==============================================================================
class LOCALMESH_OT_OriginToBottom(bpy.types.Operator):
    bl_idname = "localmesh.origin_to_bottom"
    bl_label = "Origin to Bottom / Snap Ground"
    bl_description = "Positions object origin at its lowest point and snaps Z to 0.0"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object first!")
            return {'CANCELLED'}

        import mathutils
        bbox = [obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box]
        min_z = min(v.z for v in bbox)
        obj.location.z -= min_z
        self.report({'INFO'}, f"Grounded {obj.name} to Z=0.0")
        return {'FINISHED'}


class LOCALMESH_OT_CenterPivotOrigin(bpy.types.Operator):
    bl_idname = "localmesh.center_pivot_origin"
    bl_label = "Center to Origin"
    bl_description = "Centers active object's origin to its geometry center"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not context.active_object:
            self.report({'WARNING'}, "Select an object first!")
            return {'CANCELLED'}
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        self.report({'INFO'}, "Origin centered to bounds.")
        return {'FINISHED'}


class LOCALMESH_OT_ScaleToHeight(bpy.types.Operator):
    bl_idname = "localmesh.scale_to_height"
    bl_label = "Scale to Target Height"
    bl_description = "Uniformly scales active object to match target height in meters"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        props = context.scene.local_mesh_props
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object first!")
            return {'CANCELLED'}

        cur_h = obj.dimensions.z
        if cur_h > 1e-4:
            scale_fac = props.target_height / cur_h
            obj.scale *= scale_fac
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            self.report({'INFO'}, f"Scaled {obj.name} to {props.target_height:.2f}m")
        return {'FINISHED'}


class LOCALMESH_OT_RecalculateNormals(bpy.types.Operator):
    bl_idname = "localmesh.recalculate_normals"
    bl_label = "Recalculate Outside Normals"
    bl_description = "Recalculates outside normals and cleans inverted face orientation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object first!")
            return {'CANCELLED'}

        orig_mode = obj.mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode=orig_mode)
        self.report({'INFO'}, f"Recalculated outside normals on {obj.name}")
        return {'FINISHED'}


class LOCALMESH_OT_CleanLooseGeometry(bpy.types.Operator):
    bl_idname = "localmesh.clean_loose_geometry"
    bl_label = "Clean Loose Geometry"
    bl_description = "Deletes degenerate zero-area faces, loose vertices and edges"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object first!")
            return {'CANCELLED'}

        orig_mode = obj.mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.dissolve_degenerate()
        bpy.ops.object.mode_set(mode=orig_mode)
        self.report({'INFO'}, f"Cleaned loose geometry on {obj.name}")
        return {'FINISHED'}


# Backwards compatibility operator wrappers
class TRIPOSR_OT_StartServer(LOCALMESH_OT_StartServer):
    bl_idname = "triposr.start_server"

class TRIPOSR_OT_StopServer(LOCALMESH_OT_StopServer):
    bl_idname = "triposr.stop_server"

class TRIPOSR_OT_Generate(LOCALMESH_OT_GenerateAsset):
    bl_idname = "triposr.generate_asset"

class TRIPOSR_OT_Cancel(LOCALMESH_OT_CancelJob):
    bl_idname = "triposr.cancel_asset"

class TRIPOSR_OT_ToggleWireframe(bpy.types.Operator):
    bl_idname = "triposr.toggle_wireframe"
    bl_label = "Toggle Wireframe"

    def execute(self, context):
        props = context.scene.local_mesh_props
        props.wireframe_active = not props.wireframe_active
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                obj.show_wire = props.wireframe_active
        return {'FINISHED'}

class TRIPOSR_OT_SwitchViewport(bpy.types.Operator):
    bl_idname = "triposr.switch_viewport"
    bl_label = "Show Colors (Material Mode)"

    def execute(self, context):
        _enable_viewport_color_shading()
        mat_count = 0
        target_objs = context.selected_objects if context.selected_objects else context.scene.objects
        props = context.scene.local_mesh_props
        for obj in target_objs:
            if obj.type == 'MESH':
                _apply_in_viewport_pbr(obj, None, props)
                for mat in obj.data.materials:
                    if mat and getattr(mat, "use_nodes", False):
                        mat_count += 1
        self.report({'INFO'}, f"Switched Viewport to Material Preview ({mat_count} slots active).")
        return {'FINISHED'}

class TRIPOSR_OT_ResetDefaults(bpy.types.Operator):
    bl_idname = "triposr.reset_defaults"
    bl_label = "Reset Settings"

    def execute(self, context):
        props = context.scene.local_mesh_props
        props.target_height = 1.0
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

class TRIPOSR_OT_ApplyComponentEdit(bpy.types.Operator):
    bl_idname = "triposr.apply_component_edit"
    bl_label = "Apply AI Component Edit"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.local_mesh_props
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
            self.report({'INFO'}, f"Added Subsurf modifier to {obj.name}")
        elif op == "bevel":
            mod = obj.modifiers.new(name="AI_Bevel", type='BEVEL')
            mod.width = 0.02 * factor
            self.report({'INFO'}, f"Added Bevel modifier to {obj.name}")
        elif op == "solidify":
            mod = obj.modifiers.new(name="AI_Solidify", type='SOLIDIFY')
            mod.thickness = 0.02 * factor
            self.report({'INFO'}, f"Added Solidify modifier to {obj.name}")

        return {'FINISHED'}

class TRIPOSR_OT_GroundActiveComponent(LOCALMESH_OT_OriginToBottom):
    bl_idname = "triposr.ground_active_component"


# ==============================================================================
# UI PANELS: 7 Structured Sidebar Panels (Category: "LOCAL MESH AI")
# ==============================================================================
class LOCALMESH_PT_BasePanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LOCAL MESH AI"


class LOCALMESH_PT_Connection(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 1 — Connection & Health"""
    bl_idname = "LOCALMESH_PT_connection"
    bl_label = "1. Connection & Health"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        # Live status header
        box = layout.box()
        row = box.row(align=True)
        if props.is_server_online:
            row.label(text=f"🟢 Online ({props.server_latency_ms:.1f}ms)", icon='CHECKMARK')
            row.operator("localmesh.refresh_connection", text="", icon='FILE_REFRESH')
            row.operator("localmesh.disconnect_backend", text="Disconnect", icon='UNLINKED')
        else:
            row.label(text="🔴 Backend Offline", icon='CANCEL')
            row.operator("localmesh.refresh_connection", text="Retry", icon='FILE_REFRESH')

        # Server controls
        btn_row = box.row(align=True)
        if props.is_server_online:
            btn_row.operator("localmesh.open_dashboard", text="Dashboard", icon='URL')
            btn_row.operator("localmesh.run_diagnostics", text="Diagnostics", icon='HELP')
            btn_row.operator("localmesh.stop_server", text="Stop Server", icon='QUIT')
        else:
            btn_row.operator("localmesh.start_server", text="Start Backend Console", icon='CONSOLE')

        # GPU & Telemetry details
        if props.is_server_online:
            t_col = box.column(align=True)
            t_col.label(text=f"Device: {props.server_gpu_info}", icon='DESKTOP')
            t_col.label(text=f"VRAM: {props.server_vram_info}", icon='MEMORY')

        # Expandable Server & Path Settings
        exp_row = layout.row()
        exp_row.prop(props, "show_advanced_server", text="Server Settings", icon='DOWNARROW_HLT' if props.show_advanced_server else 'RIGHTARROW')
        if props.show_advanced_server:
            s_box = layout.box()
            s_box.prop(props, "server_url", text="URL")
            s_box.prop(props, "project_dir", text="Repo Root")
            s_box.prop(props, "auto_start_server", text="Auto-Launch on Generate")


class LOCALMESH_PT_Generate(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 2 — Generate AI 3D Asset"""
    bl_idname = "LOCALMESH_PT_generate"
    bl_label = "2. Generate AI 3D Asset"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        # Mode Selector
        mode_row = layout.row(align=True)
        mode_row.prop(props, "input_mode", expand=True)

        in_box = layout.box()
        if props.input_mode == "prompt":
            in_box.label(text="Text Prompt:", icon='FONT_DATA')
            in_box.prop(props, "prompt_text", text="")
        else:
            in_box.label(text="Reference Image:", icon='IMAGE_DATA')
            in_box.prop(props, "image_path", text="")
            row = in_box.row(align=True)
            row.operator("localmesh.stage_image", text="Upload / Stage", icon='EXPORT')
            row.prop(props, "remove_bg", text="Remove BG")

        # Route & Pipeline configuration
        cfg_box = layout.box()
        cfg_box.label(text="Pipeline Route & Hardware:", icon='PREFERENCES')
        cfg_box.prop(props, "generation_route", text="Route")
        cfg_box.prop(props, "hardware_profile", text="Profile")

        # Preset & Geometric Constraints
        p_row = cfg_box.row(align=True)
        p_row.prop(props, "preset", text="Preset")
        p_row.prop(props, "quality", text="Quality")
        cfg_box.prop(props, "target_height", text="Target Height (m)")

        # Shading & Materials toggles
        col_mat = layout.box()
        col_mat.label(text="Materials & Viewport:", icon='MATERIAL')
        r1 = col_mat.row()
        r1.prop(props, "pbr_material", text="PBR Shaders")
        r1.prop(props, "accurate_colors", text="Color Calibrated")
        r2 = col_mat.row()
        r2.prop(props, "color_boost", text="Vibrancy")
        r2.prop(props, "auto_switch_shading", text="Auto Preview")

        # Generate Action Button
        layout.separator()
        if not props.is_generating:
            btn_text = "⚡ Generate 3D from Prompt" if props.input_mode == "prompt" else "⚡ Generate 3D from Image"
            layout.operator("localmesh.generate_asset", text=btn_text, icon='SHADING_RENDERED')
        else:
            prog_box = layout.box()
            prog_box.label(text=f"Processing: {props.stage_name}...", icon='TIME')
            prog_box.label(text=f"Progress: {props.progress_pct}%")
            prog_box.operator("localmesh.cancel_job", text="Cancel Job", icon='CANCEL')


class LOCALMESH_PT_JobMonitor(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 3 — Job Monitor"""
    bl_idname = "LOCALMESH_PT_job_monitor"
    bl_label = "3. Job Monitor"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        box = layout.box()
        if props.active_job_id:
            box.label(text=f"Active Job: {props.active_job_id[:12]}...", icon='SCRIPTPLUGINS')
            box.label(text=f"Status: {props.job_status} | Elapsed: {props.job_elapsed_sec:.1f}s", icon='INFO')
            box.label(text=f"Stage: {props.stage_name}", icon='TIME')

            # Progress Bar representation
            row_bar = box.row()
            row_bar.prop(props, "progress_pct", text="Progress", slider=True)

            if props.is_generating:
                box.operator("localmesh.cancel_job", text="Cancel Current Generation", icon='CANCEL')
            elif props.job_status == "COMPLETED" and props.last_completed_mesh:
                res_box = box.box()
                res_box.label(text="✅ Generation Finished!", icon='CHECKMARK')
                res_box.label(text=f"Polys: {props.last_completed_polycount:,}")
                res_box.label(text=f"Asset: {os.path.basename(props.last_completed_mesh)}")
        else:
            box.label(text="No active generation jobs.", icon='INFO')
            box.label(text=f"Status: {props.status_message}")


class LOCALMESH_PT_Assets(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 4 — Generated Assets Catalogue"""
    bl_idname = "LOCALMESH_PT_assets"
    bl_label = "4. Generated Assets"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        r_top = layout.row(align=True)
        r_top.operator("localmesh.refresh_assets", text="Refresh Catalogue", icon='FILE_REFRESH')
        r_top.operator("localmesh.open_output_folder", text="Open Output Folder", icon='FOLDER_REDIRECT')

        box = layout.box()
        box.prop(props, "selected_asset_path", text="Asset Path")

        if props.selected_asset_path and os.path.exists(bpy.path.abspath(props.selected_asset_path)):
            p = bpy.path.abspath(props.selected_asset_path)
            sz_mb = os.path.getsize(p) / (1024 * 1024)
            box.label(text=f"Size: {sz_mb:.2f} MB | {os.path.basename(p)}", icon='FILE_3D')
            box.operator("localmesh.import_asset", text="Import into Scene", icon='IMPORT')
        else:
            box.label(text="Select an asset file to inspect or import", icon='INFO')


class LOCALMESH_PT_SceneInfo(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 5 — Scene Information"""
    bl_idname = "LOCALMESH_PT_scene_info"
    bl_label = "5. Scene Information"
    bl_order = 5

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Scene overview counts
        mesh_count = sum(1 for o in scene.objects if o.type == 'MESH')
        light_count = sum(1 for o in scene.objects if o.type == 'LIGHT')
        cam_count = sum(1 for o in scene.objects if o.type == 'CAMERA')

        stat_box = layout.box()
        stat_box.label(text=f"Scene Objects: {len(scene.objects)} (Meshes: {mesh_count}, Lights: {light_count}, Cams: {cam_count})", icon='SCENE_DATA')

        # Active object inspector
        act = context.active_object
        if act and act.type == 'MESH':
            act_box = layout.box()
            act_box.label(text=f"Active Mesh: {act.name}", icon='OBJECT_DATAMODE')
            mesh = act.data
            act_box.label(text=f"Polys: {len(mesh.polygons):,} | Verts: {len(mesh.vertices):,}")
            dims = act.dimensions
            act_box.label(text=f"Dims: {dims.x:.2f}m x {dims.y:.2f}m x {dims.z:.2f}m")
            mats = [m.name for m in mesh.materials if m]
            act_box.label(text=f"Materials ({len(mats)}): {', '.join(mats) if mats else 'None'}")
        else:
            layout.label(text="Select a mesh object to view geometry stats.", icon='INFO')


class LOCALMESH_PT_SceneSync(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 6 — Two-Way Scene Synchronization"""
    bl_idname = "LOCALMESH_PT_scene_sync"
    bl_label = "6. Two-Way Scene Synchronization"
    bl_order = 6

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        box = layout.box()
        box.label(text=f"Status: {props.sync_status}", icon='SYNTAX_OFF')
        box.label(text=f"Last Sync: {props.last_sync_time} ({props.synced_object_count} objects)", icon='TIME')

        row_sync = box.row(align=True)
        row_sync.operator("localmesh.sync_scene", text="Sync Scene to Backend", icon='EXPORT')
        row_sync.operator("localmesh.pull_scene_status", text="Pull Status", icon='IMPORT')


class LOCALMESH_PT_MeshControls(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    """Panel 7 — Interactive Mesh Controls & Bounded Edits"""
    bl_idname = "LOCALMESH_PT_mesh_controls"
    bl_label = "7. Interactive Mesh Controls"
    bl_order = 7

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        # Alignment and Transformation
        box_align = layout.box()
        box_align.label(text="Transform & Alignment:", icon='ORIENTATION_GIMBAL')
        row_a = box_align.row(align=True)
        row_a.operator("localmesh.origin_to_bottom", text="Snap Ground (Z=0)", icon='SNAP_VERTEX')
        row_a.operator("localmesh.center_pivot_origin", text="Center Origin", icon='PIVOT_CURSOR')
        box_align.operator("localmesh.scale_to_height", text="Scale to Target Height", icon='CON_SIZELIMIT')

        # Cleanup & Normals
        box_clean = layout.box()
        box_clean.label(text="Geometry Cleanup:", icon='BRUSH_DATA')
        row_c = box_clean.row(align=True)
        row_c.operator("localmesh.recalculate_normals", text="Fix Normals", icon='NORMALS_FACE')
        row_c.operator("localmesh.clean_loose_geometry", text="Clean Loose", icon='X')

        # Viewport Shading
        box_view = layout.box()
        box_view.label(text="Viewport Tools:", icon='RESTRICT_VIEW_OFF')
        row_v = box_view.row(align=True)
        row_v.operator("triposr.switch_viewport", text="Material Preview", icon='SHADING_TEXTURE')
        row_v.operator("triposr.toggle_wireframe", text="Wireframe", icon='SHADING_WIRE')

        # Retained Component Hierarchy & Precision Editing
        act = context.active_object
        if act and act.type == 'MESH':
            edit_box = layout.box()
            edit_box.label(text=f"Component Edit on '{act.name}':", icon='EDITMODE_HLT')
            edit_box.prop(props, "selected_component_edit_op", text="Operation")
            if props.selected_component_edit_op == "scale_axis":
                row_ax = edit_box.row()
                row_ax.prop(props, "edit_axis", expand=True)
                edit_box.prop(props, "edit_factor", text="Scale Factor")
            elif props.selected_component_edit_op in ("bevel", "solidify"):
                edit_box.prop(props, "edit_factor", text="Thickness")
            elif props.selected_component_edit_op == "subdivision":
                edit_box.prop(props, "edit_factor", text="Levels")

            edit_box.operator("triposr.apply_component_edit", text="Apply Component Edit", icon='CHECKMARK')


# Legacy Main Panel retained under "AI 3D Generator" for backwards compatibility
class TRIPOSR_PT_MainPanel(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    bl_label = "Local Neural 3D Studio"
    bl_idname = "TRIPOSR_PT_main_panel"
    bl_category = "AI 3D Generator"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Local Mesh AI Studio migrated to 'LOCAL MESH AI' tab", icon='INFO')
        layout.operator("localmesh.generate_asset", text="Generate 3D Asset", icon='SHADING_RENDERED')
        layout.operator("triposr.switch_viewport", icon='SHADING_TEXTURE')


# ==============================================================================
# Periodic Background Server Health Check Timer
# ==============================================================================
def _periodic_server_check():
    """Checks server status periodically without blocking Blender."""
    try:
        scene = bpy.context.scene
        props = getattr(scene, "local_mesh_props", None) or getattr(scene, "triposr_props", None)
        if props and not props.is_generating:
            online, latency, _ = check_server_health(props.server_url, timeout=0.5)
            props.is_server_online = online
            if online:
                props.server_latency_ms = latency
    except Exception:
        pass
    return 4.0


# ==============================================================================
# Registration & Unregistration
# ==============================================================================
classes = (
    LocalMeshAddonPreferences,
    LocalMeshProperties,
    LOCALMESH_OT_RefreshConnection,
    LOCALMESH_OT_DisconnectBackend,
    LOCALMESH_OT_StartServer,
    LOCALMESH_OT_StopServer,
    LOCALMESH_OT_OpenDashboard,
    LOCALMESH_OT_RunDiagnostics,
    LOCALMESH_OT_StageImage,
    LOCALMESH_OT_GenerateAsset,
    LOCALMESH_OT_CancelJob,
    LOCALMESH_OT_RefreshAssets,
    LOCALMESH_OT_ImportAsset,
    LOCALMESH_OT_OpenOutputFolder,
    LOCALMESH_OT_SyncScene,
    LOCALMESH_OT_PullSceneStatus,
    LOCALMESH_OT_OriginToBottom,
    LOCALMESH_OT_CenterPivotOrigin,
    LOCALMESH_OT_ScaleToHeight,
    LOCALMESH_OT_RecalculateNormals,
    LOCALMESH_OT_CleanLooseGeometry,
    TRIPOSR_OT_StartServer,
    TRIPOSR_OT_StopServer,
    TRIPOSR_OT_Generate,
    TRIPOSR_OT_Cancel,
    TRIPOSR_OT_ToggleWireframe,
    TRIPOSR_OT_SwitchViewport,
    TRIPOSR_OT_ResetDefaults,
    TRIPOSR_OT_ApplyComponentEdit,
    TRIPOSR_OT_GroundActiveComponent,
    LOCALMESH_PT_Connection,
    LOCALMESH_PT_Generate,
    LOCALMESH_PT_JobMonitor,
    LOCALMESH_PT_Assets,
    LOCALMESH_PT_SceneInfo,
    LOCALMESH_PT_SceneSync,
    LOCALMESH_PT_MeshControls,
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

    bpy.types.Scene.local_mesh_props = bpy.props.PointerProperty(type=LocalMeshProperties)
    bpy.types.Scene.triposr_props = bpy.props.PointerProperty(type=LocalMeshProperties)

    try:
        root = find_project_root()
        pref = get_addon_preferences()
        if pref and root and not pref.project_dir:
            pref.project_dir = root
        if hasattr(bpy.context, "scene"):
            props = getattr(bpy.context.scene, "local_mesh_props", None)
            if props and root:
                props.project_dir = root
    except Exception:
        pass

    if not bpy.app.timers.is_registered(_periodic_server_check):
        bpy.app.timers.register(_periodic_server_check, first_interval=1.5)


def unregister():
    try:
        stop_backend_console()
    except Exception:
        pass

    if bpy.app.timers.is_registered(_periodic_server_check):
        bpy.app.timers.unregister(_periodic_server_check)
    if bpy.app.timers.is_registered(poll_active_job_timer):
        bpy.app.timers.unregister(poll_active_job_timer)

    if hasattr(bpy.types.Scene, "local_mesh_props"):
        try:
            del bpy.types.Scene.local_mesh_props
        except Exception:
            pass
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
