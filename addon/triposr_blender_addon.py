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



import queue

_addon_result_queue = queue.Queue()


def _process_addon_result_queue():
    """Pumps background worker results exclusively on Blender's main thread."""
    while not _addon_result_queue.empty():
        try:
            cb, arg, is_error = _addon_result_queue.get_nowait()
            if cb:
                try:
                    cb(arg)
                except Exception as ex:
                    print(f"[LocalMeshAI] Error in main-thread callback: {ex}")
            _addon_result_queue.task_done()
        except queue.Empty:
            break
        except Exception as e:
            print(f"[LocalMeshAI] Error pumping result queue: {e}")
    return 0.05  # Check queue every 50ms


def run_in_background(task_func, on_success=None, on_error=None):
    """
    Executes a network or compute task on a background daemon thread without blocking Blender.
    Worker threads NEVER call bpy API functions directly; results are placed onto a thread-safe queue.
    """
    def _worker():
        try:
            res = task_func()
            if on_success:
                _addon_result_queue.put((on_success, res, False))
        except Exception as e:
            if on_error:
                _addon_result_queue.put((on_error, e, True))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


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
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        os.path.expanduser(r"~\local-mesh-ai"),
        os.path.expanduser(r"~\.local-mesh-ai")
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
    # --- Staged Asset & Image Tracking ---
    staged_image_path: StringProperty(name="Staged Image Path", default="")

    # --- Phase 6 AI Technical Artist Properties ---
    ai_session_id: StringProperty(name="Session ID", default="")
    ai_artist_instruction: StringProperty(
        name="Instruction",
        default="Make the backrest 20% taller without modifying the seat",
        description="Natural-language 3D editing request"
    )
    ai_artist_target_comp: StringProperty(name="Target Component", default="")
    ai_artist_status: StringProperty(name="AI Artist Status", default="Ready")
    ai_plan_preview: StringProperty(name="Plan Preview", default="")
    ai_plan_id: StringProperty(name="Plan ID", default="")
    ai_plan_target: StringProperty(name="Target", default="")
    ai_plan_operation: StringProperty(name="Operation", default="")
    ai_plan_factor: FloatProperty(name="Factor", default=1.2)
    ai_plan_axis: StringProperty(name="Axis", default="z")
    ai_plan_invariants: StringProperty(name="Invariants", default="")
    ai_has_plan: BoolProperty(name="Has Plan", default=False)
    ai_needs_confirmation: BoolProperty(name="Needs Confirmation", default=False)
    ai_is_busy: BoolProperty(name="AI Busy", default=False)
    ai_component_count: IntProperty(name="Component Count", default=0)
    ai_candidate_components_json: StringProperty(name="Candidates JSON", default="[]")
    ai_last_history: StringProperty(name="Last Result", default="")

    # --- Phase 7 Finishing Studio Properties ---
    finishing_status: StringProperty(name="Finishing Status", default="Ready")
    finishing_is_busy: BoolProperty(name="Finishing Busy", default=False)
    mesh_health_score: FloatProperty(name="Health Score", default=100.0)
    mesh_health_summary: StringProperty(name="Health Summary", default="Not inspected")
    mesh_is_game_ready: BoolProperty(name="Is Game-Ready", default=False)
    mesh_issues_json: StringProperty(name="Mesh Issues JSON", default="[]")

    finishing_target_tris: IntProperty(name="Target Triangles", default=5000, min=100, max=100000)
    finishing_reduction_ratio: FloatProperty(name="Reduction Ratio", default=0.5, min=0.01, max=0.99)
    finishing_retopo_method: EnumProperty(
        name="Retopo Method",
        items=[
            ("quadriflow", "Blender QuadriFlow (Quads)", "True 4-sided polygon flow using native Blender QuadriFlow"),
            ("decimation", "Quadric Decimation (Triangles)", "Edge-preserving quadric error decimation"),
            ("voxel", "Voxel Remesh", "Uniform voxel surface reconstruction"),
        ],
        default="quadriflow"
    )
    finishing_retopo_quads: IntProperty(name="Target Quads", default=4000, min=100, max=50000)
    finishing_preserve_sharp: BoolProperty(name="Preserve Sharp Edges", default=True)
    finishing_symmetry: EnumProperty(
        name="Symmetry",
        items=[
            ("none", "None", "No symmetry enforcement"),
            ("x", "X-Axis", "Enforce bilateral symmetry along X"),
            ("y", "Y-Axis", "Enforce bilateral symmetry along Y"),
            ("z", "Z-Axis", "Enforce bilateral symmetry along Z"),
        ],
        default="none"
    )

    finishing_uv_method: EnumProperty(
        name="UV Method",
        items=[
            ("smart_project", "Smart UV Project", "Automated multi-angle projection unwrap"),
            ("angle_based", "Angle-Based", "Conformal angle-preserving unwrap"),
            ("conformal", "Conformal", "Area-preserving conformal unwrap"),
        ],
        default="smart_project"
    )
    finishing_uv_margin: FloatProperty(name="Island Margin", default=0.02, min=0.001, max=0.2)
    finishing_uv_summary: StringProperty(name="UV Summary", default="No UV data")

    finishing_mat_preset: EnumProperty(
        name="Material Preset",
        items=[
            ("wood_walnut", "Walnut Wood", "Dark walnut wood with subtle gloss"),
            ("wood_oak", "Oak Wood", "Warm natural oak wood"),
            ("metal_chrome", "Polished Chrome", "High-reflectivity polished mirror metal"),
            ("metal_rough", "Brushed Steel", "Industrial brushed metallic surface"),
            ("fabric_cotton", "Cotton Fabric", "Soft matte upholstered woven textile"),
            ("leather_aged", "Aged Leather", "Textured vintage brown leather"),
            ("plastic_glossy", "Glossy Plastic", "Smooth colored hard plastic shell"),
            ("stone_granite", "Granite Stone", "Rough architectural granite stone"),
        ],
        default="wood_walnut"
    )
    finishing_mat_metallic: FloatProperty(name="Metallic", default=0.0, min=0.0, max=1.0)
    finishing_mat_roughness: FloatProperty(name="Roughness", default=0.5, min=0.0, max=1.0)

    finishing_bake_res: EnumProperty(
        name="Bake Resolution",
        items=[
            ("512", "512 x 512 (Draft)", "Fast draft resolution baking"),
            ("1024", "1024 x 1024 (Standard)", "Standard production PBR map resolution"),
            ("2048", "2048 x 2048 (High-Res)", "High-resolution texture baking"),
        ],
        default="1024"
    )
    finishing_bake_maps: StringProperty(name="Bake Maps", default="normal,base_color,ao,roughness,metallic")
    finishing_bake_high_poly: StringProperty(
        name="High-Poly Object",
        description="Source high-detail mesh object for detail baking",
        default=""
    )
    finishing_bake_low_poly: StringProperty(
        name="Low-Poly Object",
        description="Target low-poly mesh object receiving baked textures",
        default=""
    )

    finishing_export_glb: BoolProperty(name="GLB / glTF", default=True)
    finishing_export_fbx: BoolProperty(name="FBX", default=True)
    finishing_export_obj: BoolProperty(name="OBJ + MTL", default=True)
    finishing_export_summary: StringProperty(name="Export Summary", default="")

    # --- Phase 9 System Health, Learning & Lifecycle ---
    health_status_text: StringProperty(name="Health Status", default="Not Checked")
    health_issue_count: IntProperty(name="Health Issues", default=0)
    health_is_busy: BoolProperty(name="Health Busy", default=False)
    learning_consent_opted_in: BoolProperty(name="Opt-In Local Learning", default=False)
    active_model_name: StringProperty(name="Active Model", default="Qwen2.5-Coder 3B (GGUF Q4_K_M)")
    recovery_preview_id: StringProperty(name="Recovery Preview ID", default="")
    recovery_expires_at: FloatProperty(name="Recovery Expiration", default=0.0)
    candidate_model_name: StringProperty(name="Candidate Adapter", default="None")
    model_eval_score: FloatProperty(name="Model Eval Score", default=0.0)
    model_eval_passed: BoolProperty(name="Model Eval Passed", default=False)
    recovery_preview_ready: BoolProperty(name="Recovery Preview Ready", default=False)
    recovery_preview_text: StringProperty(name="Recovery Preview Text", default="")
    recovery_affected_files: StringProperty(name="Recovery Affected Files", default="")
    preflight_free_vram: FloatProperty(name="Preflight Free VRAM", default=0.0)
    preflight_feasibility: StringProperty(name="Preflight Feasibility", default="unknown")

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
            try:
                c_req = urllib.request.Request(f"{props.server_url}/api/v1/learning/consent")
                with urllib.request.urlopen(c_req, timeout=3.0) as c_resp:
                    c_data = json.loads(c_resp.read().decode("utf-8"))
                    props.learning_consent_opted_in = bool(c_data.get("opted_in", False))
            except Exception:
                pass
            try:
                m_req = urllib.request.Request(f"{props.server_url}/api/v1/learning/models")
                with urllib.request.urlopen(m_req, timeout=3.0) as m_resp:
                    m_list = json.loads(m_resp.read().decode("utf-8"))
                    if isinstance(m_list, list):
                        cand_found = False
                        for m_item in m_list:
                            if m_item.get("is_active"):
                                props.active_model_name = m_item.get("name") or m_item.get("model_id")
                            elif m_item.get("model_type") == "lora_adapter" and not cand_found:
                                props.candidate_model_name = m_item.get("name") or m_item.get("model_id")
                                cand_found = True
                        if not cand_found:
                            props.candidate_model_name = "None"
            except Exception:
                pass
            try:
                pref_body = json.dumps({"model_name": "local-technical-artist-lora", "force_cpu": False}).encode("utf-8")
                p_req = urllib.request.Request(
                    f"{props.server_url}/api/v1/learning/preflight",
                    data=pref_body,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(p_req, timeout=3.0) as p_resp:
                    p_data = json.loads(p_resp.read().decode("utf-8"))
                    props.preflight_free_vram = float(p_data.get("free_vram_mb") or 0.0)
                    props.preflight_feasibility = str(p_data.get("hardware_feasibility", "unknown"))
            except Exception:
                pass
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
                props.staged_image_path = staged_path
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
            staged = getattr(props, "staged_image_path", "")
            if staged and os.path.exists(staged):
                payload["image_path"] = os.path.abspath(staged)
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



# ==============================================================================
# Phase 6 Helpers: Mesh Transfer & In-Place Geometry Replacement
# ==============================================================================
def _export_active_object_mesh(obj):
    """Exports active selected mesh into a temporary working GLB or OBJ file."""
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"staged_{uuid.uuid4().hex[:6]}.glb")

    prev_active = bpy.context.view_layer.objects.active
    prev_selected = [o for o in bpy.context.selected_objects]

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.export_scene.gltf(filepath=temp_path, use_selection=True, export_format='GLB')
    except Exception:
        temp_path = os.path.join(temp_dir, f"staged_{uuid.uuid4().hex[:6]}.obj")
        try:
            bpy.ops.wm.obj_export(filepath=temp_path, export_selected_objects=True)
        except Exception:
            bpy.ops.export_scene.obj(filepath=temp_path, use_selection=True)
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        for o in prev_selected:
            if o.name in bpy.context.view_layer.objects:
                o.select_set(True)
        if prev_active and prev_active.name in bpy.context.view_layer.objects:
            bpy.context.view_layer.objects.active = prev_active

    return temp_path


def _update_blender_object_geometry(obj, new_mesh_filepath):
    """In-place updates target Blender object geometry on the main thread, preserving transforms and materials."""
    if not obj or obj.type != 'MESH' or not os.path.exists(new_mesh_filepath):
        return False

    orig_objs = set(bpy.context.scene.objects)
    try:
        if new_mesh_filepath.lower().endswith((".glb", ".gltf")):
            bpy.ops.import_scene.gltf(filepath=new_mesh_filepath)
        else:
            try:
                bpy.ops.wm.obj_import(filepath=new_mesh_filepath)
            except Exception:
                bpy.ops.import_scene.obj(filepath=new_mesh_filepath)

        new_objs = set(bpy.context.scene.objects) - orig_objs
        mesh_objs = [o for o in new_objs if o.type == 'MESH']
        if mesh_objs:
            temp_mesh_obj = mesh_objs[0]
            old_data = obj.data
            obj.data = temp_mesh_obj.data
            if not obj.data.materials and old_data.materials:
                for mat in old_data.materials:
                    obj.data.materials.append(mat)
            if old_data.users == 0:
                bpy.data.meshes.remove(old_data)

        for o in new_objs:
            if o != obj and o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        return True
    except Exception as e:
        print(f"[LocalMeshAI] Error applying updated mesh to object: {e}")
        return False


# ==============================================================================
# OPERATORS: Phase 6 AI Technical Artist
# ==============================================================================
class LOCALMESH_OT_StageActiveAsset(bpy.types.Operator):
    bl_idname = "localmesh.stage_active_asset"
    bl_label = "Stage Active Asset for AI"
    bl_description = "Exports active Blender mesh and initializes an AI Technical Artist editing session"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Please select an active 3D mesh object.")
            return {'CANCELLED'}

        props.ai_is_busy = True
        props.ai_artist_status = f"Staging '{act.name}'..."

        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url
        obj_name = act.name
        bbox = {"dimensions": [float(d) for d in act.dimensions]}

        def _task():
            req_data = {
                "asset_name": obj_name,
                "mesh_path": temp_mesh,
                "file_format": Path(temp_mesh).suffix.lstrip("."),
                "object_metadata": bbox
            }
            req = urllib.request.Request(
                f"{server_url}/api/v1/blender/stage-asset",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _on_success(data):
            props.ai_session_id = data.get("session_id", "")
            props.ai_artist_status = f"Staged '{obj_name}' ({data.get('face_count', 0)} faces)"
            props.ai_is_busy = False
            self.report({'INFO'}, f"Asset '{obj_name}' staged (Session {props.ai_session_id})")

        def _on_error(err):
            props.ai_is_busy = False
            props.ai_artist_status = f"Staging failed: {err}"
            self.report({'ERROR'}, f"Staging failed: {err}")

        run_in_background(_task, _on_success, _on_error)
        return {'FINISHED'}


class LOCALMESH_OT_AnalyzeActiveAsset(bpy.types.Operator):
    bl_idname = "localmesh.analyze_active_asset"
    bl_label = "Analyze Components & Grounding"
    bl_description = "Performs PartField component decomposition, 6-view rendering, and visual understanding"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        server_url = props.server_url
        props.ai_is_busy = True
        props.ai_artist_status = "Analyzing geometry & neural components..."

        # Stage active mesh synchronously in worker if session_id is not yet initialized (eliminates race condition)
        session_id_to_use = props.ai_session_id
        temp_mesh_export = _export_active_object_mesh(act) if (not session_id_to_use and act and act.type == 'MESH') else None
        act_name = act.name if act else "Active_Mesh"

        def _task():
            nonlocal session_id_to_use
            if not session_id_to_use:
                if not temp_mesh_export:
                    raise RuntimeError("No active session or active mesh object found.")
                stage_payload = {
                    "asset_name": act_name,
                    "mesh_path": temp_mesh_export,
                    "file_format": Path(temp_mesh_export).suffix.lstrip("."),
                    "object_metadata": {}
                }
                stage_req = urllib.request.Request(
                    f"{server_url}/api/v1/blender/stage-asset",
                    data=json.dumps(stage_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(stage_req, timeout=30.0) as s_resp:
                    s_data = json.loads(s_resp.read().decode("utf-8"))
                    session_id_to_use = s_data["session_id"]

            req_data = {
                "session_id": session_id_to_use,
                "decomposition_method": "auto",
                "num_clusters": 4,
                "include_vlm": True
            }
            req = urllib.request.Request(
                f"{server_url}/api/v1/artist/analyze",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _on_success(data):
            props.ai_component_count = data.get("total_components", 0)
            comps = data.get("components", [])
            props.ai_candidate_components_json = json.dumps(comps)
            props.ai_is_busy = False
            props.ai_artist_status = f"Analyzed: {len(comps)} components ({data.get('decomposition_provenance', 'auto')})"
            self.report({'INFO'}, f"Analysis complete: {len(comps)} components decomposed.")

        def _on_error(err):
            props.ai_is_busy = False
            props.ai_artist_status = f"Analysis failed: {err}"
            self.report({'ERROR'}, f"Analysis error: {err}")

        run_in_background(_task, _on_success, _on_error)
        return {'FINISHED'}


class LOCALMESH_OT_GenerateArtistPlan(bpy.types.Operator):
    bl_idname = "localmesh.generate_artist_plan"
    bl_label = "Generate AI Edit Plan"
    bl_description = "Converts natural-language instruction into structured Pydantic plan via Qwen Coder"

    def execute(self, context):
        props = context.scene.local_mesh_props
        if not props.ai_session_id:
            self.report({'ERROR'}, "Stage or analyze asset before generating edit plan.")
            return {'CANCELLED'}

        instruction = props.ai_artist_instruction.strip()
        if not instruction:
            self.report({'ERROR'}, "Please enter an instruction (e.g. 'Make backrest 20% taller').")
            return {'CANCELLED'}

        props.ai_is_busy = True
        props.ai_artist_status = "Qwen Coder generating edit plan..."
        session_id = props.ai_session_id
        server_url = props.server_url
        explicit_target = props.ai_artist_target_comp.strip() or None

        def _task():
            req_data = {
                "session_id": session_id,
                "user_instruction": instruction,
                "target_component_id": explicit_target
            }
            req = urllib.request.Request(
                f"{server_url}/api/v1/artist/plan",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _on_success(data):
            props.ai_is_busy = False
            plan = data.get("plan", {})
            props.ai_plan_id = data.get("plan_id", "")
            targets = plan.get("target_semantic_labels", []) or plan.get("target_component_ids", [])
            props.ai_plan_target = ", ".join(targets) if targets else "unresolved"
            props.ai_plan_operation = plan.get("operation_type", "scale")
            props.ai_plan_factor = float(plan.get("scale_factor", 1.0))
            props.ai_plan_axis = plan.get("axis", "z")

            invs = [inv.get("component_id", "") for inv in plan.get("invariants", [])]
            props.ai_plan_invariants = ", ".join(invs) if invs else "All other components"
            props.ai_plan_preview = plan.get("rationale", data.get("rationale", ""))

            needs_conf = (data.get("status") == "NEEDS_COMPONENT_CONFIRMATION" or plan.get("requires_user_confirmation", False))
            props.ai_needs_confirmation = needs_conf
            props.ai_has_plan = True
            props.ai_artist_status = f"Plan Ready: {props.ai_plan_operation} on {props.ai_plan_target}"
            self.report({'INFO'}, f"Edit plan generated: {props.ai_plan_operation} (Status: {data.get('status')})")

        def _on_error(err):
            props.ai_is_busy = False
            props.ai_artist_status = f"Planning failed: {err}"
            self.report({'ERROR'}, f"Planning error: {err}")

        run_in_background(_task, _on_success, _on_error)
        return {'FINISHED'}


class LOCALMESH_OT_ConfirmTargetComponent(bpy.types.Operator):
    bl_idname = "localmesh.confirm_target_component"
    bl_label = "Select Candidate Target"
    bl_description = "Selects this candidate component to resolve planning ambiguity"

    target_cid: bpy.props.StringProperty(name="Target CID")

    def execute(self, context):
        props = context.scene.local_mesh_props
        props.ai_artist_target_comp = self.target_cid
        props.ai_needs_confirmation = False
        self.report({'INFO'}, f"Confirmed target component: {self.target_cid}")
        # Regenerate plan immediately with confirmed target
        bpy.ops.localmesh.generate_artist_plan()
        return {'FINISHED'}


class LOCALMESH_OT_ApplyArtistPlan(bpy.types.Operator):
    bl_idname = "localmesh.execute_artist_plan"
    bl_label = "⚡ Apply Edit Plan"
    bl_description = "Executes approved plan, verifies invariants, and updates active Blender object geometry"

    def execute(self, context):
        props = context.scene.local_mesh_props
        if not props.ai_session_id or not props.ai_has_plan:
            self.report({'ERROR'}, "No active approved plan to apply.")
            return {'CANCELLED'}

        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select the target mesh object in the 3D viewport.")
            return {'CANCELLED'}

        props.ai_is_busy = True
        props.ai_artist_status = "Executing deterministic edit & verifying geometry..."
        session_id = props.ai_session_id
        plan_id = props.ai_plan_id
        server_url = props.server_url

        def _task():
            req_data = {
                "session_id": session_id,
                "plan_id": plan_id,
                "confirmed_target_id": props.ai_artist_target_comp or None
            }
            req = urllib.request.Request(
                f"{server_url}/api/v1/artist/execute",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))

            # Download the resulting mesh
            mesh_url = res_data.get("output_mesh_url")
            if not mesh_url.startswith("http"):
                mesh_url = f"{server_url}{mesh_url}"
            import tempfile
            dl_path = os.path.join(tempfile.gettempdir(), f"edited_{session_id}_{uuid.uuid4().hex[:6]}.glb")
            urllib.request.urlretrieve(mesh_url, dl_path)
            res_data["_downloaded_mesh_path"] = dl_path
            return res_data

        def _on_success(data):
            props.ai_is_busy = False
            status = data.get("status", "SUCCESS")
            report = data.get("report", {})
            dl_path = data.get("_downloaded_mesh_path")

            if dl_path and os.path.exists(dl_path):
                applied = _update_blender_object_geometry(act, dl_path)
                if applied:
                    self.report({'INFO'}, f"Edit applied: {report.get('summary', 'Updated mesh geometry')}")

            props.ai_last_history = report.get("summary", "")
            props.ai_artist_status = f"Result: {status}"
            props.ai_has_plan = False

        def _on_error(err):
            props.ai_is_busy = False
            props.ai_artist_status = f"Execution failed: {err}"
            self.report({'ERROR'}, f"Execution error: {err}")

        run_in_background(_task, _on_success, _on_error)
        return {'FINISHED'}


class LOCALMESH_OT_RollbackArtistEdit(bpy.types.Operator):
    bl_idname = "localmesh.rollback_artist_edit"
    bl_label = "↺ Rollback Edit"
    bl_description = "Restores previous pre-edit mesh geometry snapshot on active object"

    def execute(self, context):
        props = context.scene.local_mesh_props
        if not props.ai_session_id:
            self.report({'ERROR'}, "No active editing session found.")
            return {'CANCELLED'}

        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select the target mesh object.")
            return {'CANCELLED'}

        props.ai_is_busy = True
        props.ai_artist_status = "Rolling back to previous snapshot..."
        session_id = props.ai_session_id
        server_url = props.server_url

        def _task():
            req_data = {"session_id": session_id}
            req = urllib.request.Request(
                f"{server_url}/api/v1/artist/rollback",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))

            mesh_url = res_data.get("output_mesh_url")
            if not mesh_url.startswith("http"):
                mesh_url = f"{server_url}{mesh_url}"
            import tempfile
            dl_path = os.path.join(tempfile.gettempdir(), f"rollback_{session_id}_{uuid.uuid4().hex[:6]}.glb")
            urllib.request.urlretrieve(mesh_url, dl_path)
            res_data["_downloaded_mesh_path"] = dl_path
            return res_data

        def _on_success(data):
            props.ai_is_busy = False
            dl_path = data.get("_downloaded_mesh_path")
            if dl_path and os.path.exists(dl_path):
                _update_blender_object_geometry(act, dl_path)
            props.ai_artist_status = "Rolled Back to Pre-Edit Snapshot"
            self.report({'INFO'}, "Successfully restored pre-edit snapshot.")

        def _on_error(err):
            props.ai_is_busy = False
            props.ai_artist_status = f"Rollback failed: {err}"
            self.report({'ERROR'}, f"Rollback error: {err}")

        run_in_background(_task, _on_success, _on_error)
        return {'FINISHED'}


# ==============================================================================
# PANEL 8: AI Technical Artist & Interactive Workspace
# ==============================================================================
class LOCALMESH_PT_AITechnicalArtist(LOCALMESH_PT_BasePanel, bpy.types.Panel):
    bl_label = "AI Technical Artist & Component Editing"
    bl_idname = "LOCALMESH_PT_ai_technical_artist"
    bl_order = 8

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props
        act = context.active_object

        # 1. Active Target Asset Info
        box_asset = layout.box()
        if act and act.type == 'MESH':
            row = box_asset.row()
            row.label(text=f"Target: {act.name}", icon='OBJECT_DATAMODE')
            row.label(text=f"{len(act.data.polygons)} faces", icon='MESH_DATA')
            row_actions = box_asset.row(align=True)
            row_actions.operator("localmesh.stage_active_asset", text="Stage for AI", icon='EXPORT')
            row_actions.operator("localmesh.analyze_active_asset", text="Analyze & Decompose", icon='NODETREE')
        else:
            box_asset.label(text="Select a Mesh object to edit", icon='INFO')
            return

        # 2. Decomposed Components Hierarchy
        if props.ai_component_count > 0:
            box_comp = layout.box()
            box_comp.label(text=f"Detected Components ({props.ai_component_count}):", icon='OUTLINER_OB_MESH')
            try:
                comps = json.loads(props.ai_candidate_components_json or "[]")
                for c in comps[:6]:
                    c_row = box_comp.row(align=True)
                    cid = c.get("component_id", "")
                    lbl = c.get("semantic_label", cid)
                    conf = c.get("confidence", 0.5)
                    color = c.get("color_hex")
                    dims = c.get("dimensions", [0, 0, 0])
                    dim_str = f"{dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}m"
                    c_row.label(text=f"{lbl} ({dim_str}, {int(conf*100)}%)")
                    op = c_row.operator("localmesh.confirm_target_component", text="Target", icon='RESTRICT_SELECT_OFF')
                    op.target_cid = cid
            except Exception:
                pass

        # 3. Natural Language Instruction Input
        box_inst = layout.box()
        box_inst.label(text="Natural-Language Technical Artist Request:", icon='CONSOLE')
        box_inst.prop(props, "ai_artist_instruction", text="")
        if props.ai_artist_target_comp:
            box_inst.label(text=f"Explicit Target: {props.ai_artist_target_comp}", icon='CHECKMARK')

        btn_row = box_inst.row()
        btn_row.operator("localmesh.generate_artist_plan", text="Generate Edit Plan", icon='LIGHT_SUN')

        # 4. Plan Preview & Confirmation Gating
        if props.ai_has_plan:
            box_plan = layout.box()
            box_plan.label(text="AI Proposed Plan:", icon='SCRIPT')
            box_plan.label(text=f"Target: {props.ai_plan_target} | Op: {props.ai_plan_operation}")
            box_plan.label(text=f"Factor: {props.ai_plan_factor:.2f} | Axis: {props.ai_plan_axis}")
            box_plan.label(text=f"Invariants: {props.ai_plan_invariants}")
            box_plan.label(text=props.ai_plan_preview)

            if props.ai_needs_confirmation:
                box_plan.alert = True
                box_plan.label(text="Ambiguous Target: Select Component Above!", icon='ERROR')

            row_exec = box_plan.row(align=True)
            row_exec.operator("localmesh.execute_artist_plan", text="⚡ Apply Edit Plan", icon='CHECKMARK')
            row_exec.operator("localmesh.rollback_artist_edit", text="↺ Rollback", icon='LOOP_BACK')

        # 5. Status & Telemetry
        box_stat = layout.box()
        box_stat.label(text=f"Status: {props.ai_artist_status}", icon='TIME')
        if props.ai_last_history:
            box_stat.label(text=props.ai_last_history)

# Consolidated under single unified sidebar tab "LOCAL MESH AI"


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

# ==============================================================================
# Phase 7 Professional 3D Finishing Operators & Studio Panel
# ==============================================================================

class LOCALMESH_OT_InspectMeshHealth(bpy.types.Operator):
    bl_idname = "localmesh.inspect_mesh_health"
    bl_label = "Inspect Mesh Health"
    bl_description = "Evaluates geometric, topological, manifold, and game-readiness metrics"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select an active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Inspecting mesh health..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/validate",
                data=json.dumps({"mesh_path": temp_mesh}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            score = float(data.get("health_score", 100.0))
            props.mesh_health_score = score
            props.mesh_is_game_ready = bool(data.get("is_game_ready", False))
            issues = data.get("issues_found", [])
            props.mesh_issues_json = json.dumps(issues)
            props.mesh_health_summary = f"Score: {score:.1f}% | Non-manifold: {data.get('non_manifold_edges', 0)} | Zero-area: {data.get('zero_area_faces', 0)}"
            props.finishing_status = f"Inspection complete: {score:.1f}% health."
            if issues:
                self.report({'INFO'}, f"Health score {score:.1f}%. Issues: {len(issues)}")
            else:
                self.report({'INFO'}, f"Mesh is healthy ({score:.1f}%).")

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"Inspection failed: {err}"
            self.report({'ERROR'}, f"Validation error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_RepairActiveMesh(bpy.types.Operator):
    bl_idname = "localmesh.repair_active_mesh"
    bl_label = "Repair Mesh Geometry"
    bl_description = "Safely removes zero-area faces, duplicate coincident vertices, and fixes normals"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select an active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Repairing geometry..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/repair",
                data=json.dumps({"mesh_path": temp_mesh}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=40.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            repaired_path = data.get("output_mesh_path")
            if repaired_path and os.path.exists(repaired_path):
                _update_blender_object_geometry(act, repaired_path)
            repairs = data.get("repairs_applied", [])
            props.finishing_status = f"Repair complete ({len(repairs)} fixes applied)."
            self.report({'INFO'}, f"Repairs applied: {len(repairs)}")

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"Repair failed: {err}"
            self.report({'ERROR'}, f"Repair error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_ExecuteRetopology(bpy.types.Operator):
    bl_idname = "localmesh.execute_retopology"
    bl_label = "Execute QuadriFlow Retopology"
    bl_description = "Applies native Blender QuadriFlow remeshing for clean, continuous 4-sided polygon flow"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select an active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Executing QuadriFlow remeshing..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url
        target_f = props.finishing_retopo_quads
        sym = None if props.finishing_symmetry == "none" else props.finishing_symmetry
        sharp = props.finishing_preserve_sharp

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/retopology",
                data=json.dumps({
                    "mesh_path": temp_mesh,
                    "target_faces": target_f,
                    "symmetry_axis": sym,
                    "preserve_sharp": sharp
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=90.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            out_p = data.get("output_mesh_path")
            if out_p and os.path.exists(out_p):
                _update_blender_object_geometry(act, out_p)
            actual_f = data.get("actual_faces", 0)
            method = data.get("method", "retopology")
            props.finishing_status = f"Retopology complete ({actual_f} faces, {method})."
            self.report({'INFO'}, f"Retopology complete: {actual_f} faces.")

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"Retopology failed: {err}"
            self.report({'ERROR'}, f"Retopology error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_UnwrapActiveMesh(bpy.types.Operator):
    bl_idname = "localmesh.unwrap_active_mesh"
    bl_label = "Unwrap & Pack UVs"
    bl_description = "Unwraps UVs with island margin padding to prevent texture bleeding"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select an active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Unwrapping UV coordinates..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url
        m_method = props.finishing_uv_method
        m_margin = props.finishing_uv_margin

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/unwrap",
                data=json.dumps({
                    "mesh_path": temp_mesh,
                    "method": m_method,
                    "margin": m_margin
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            out_p = data.get("output_mesh_path")
            if out_p and os.path.exists(out_p):
                _update_blender_object_geometry(act, out_p)
            uv_rep = data.get("uv_report", {})
            util = uv_rep.get("uv_utilization_percent", 0.0)
            props.finishing_uv_summary = f"{util:.1f}% UV utilization, {uv_rep.get('zero_area_uv_count', 0)} zero-area"
            props.finishing_status = f"UV unwrap complete ({util:.1f}% coverage)."
            self.report({'INFO'}, f"UV unwrap complete: {util:.1f}% utilization.")

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"UV unwrap failed: {err}"
            self.report({'ERROR'}, f"UV unwrap error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_ApplyPBRMaterials(bpy.types.Operator):
    bl_idname = "localmesh.apply_pbr_materials"
    bl_label = "Apply PBR Preset"
    bl_description = "Constructs Principled BSDF PBR material node tree with metallic and roughness parameters"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select an active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Applying PBR material preset..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url
        preset = props.finishing_mat_preset

        def _task():
            mat_spec = {
                "material_name": f"M_{preset}",
                "preset": preset,
                "metallic": props.finishing_mat_metallic,
                "roughness": props.finishing_mat_roughness
            }
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/materials",
                data=json.dumps({"mesh_path": temp_mesh, "materials": [mat_spec]}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            out_p = data.get("mesh_path")
            if out_p and os.path.exists(out_p):
                _update_blender_object_geometry(act, out_p)
            props.finishing_status = f"Applied PBR preset '{preset}'."
            self.report({'INFO'}, f"Applied PBR preset '{preset}'.")

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"Material failed: {err}"
            self.report({'ERROR'}, f"Material error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_BakeTexturesCycles(bpy.types.Operator):
    bl_idname = "localmesh.bake_textures_cycles"
    bl_label = "Bake PBR Maps (Cycles)"
    bl_description = "Executes Selected-to-Active baking for Normal, Diffuse, AO, Roughness, and Metallic"

    def execute(self, context):
        props = context.scene.local_mesh_props
        hp_name = props.finishing_bake_high_poly.strip()
        lp_name = props.finishing_bake_low_poly.strip()

        hp_obj = context.scene.objects.get(hp_name) if hp_name else None
        lp_obj = context.scene.objects.get(lp_name) if lp_name else None

        if not hp_obj or not lp_obj:
            selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
            if len(selected_meshes) >= 2:
                act = context.active_object
                if act in selected_meshes:
                    lp_obj = act
                    candidates = [o for o in selected_meshes if o != act]
                    hp_obj = candidates[0]
                else:
                    hp_obj = selected_meshes[0]
                    lp_obj = selected_meshes[1]

        if not hp_obj or not lp_obj:
            self.report({'ERROR'}, "Specify both High-Poly and Low-Poly mesh objects (or select 2 mesh objects in viewport).")
            return {'CANCELLED'}

        if hp_obj == lp_obj:
            self.report({'ERROR'}, "High-poly and Low-poly must be distinct, separate mesh objects.")
            return {'CANCELLED'}

        if hp_obj.type != 'MESH' or lp_obj.type != 'MESH':
            self.report({'ERROR'}, "Both selected objects must be meshes.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = f"Baking Cycles PBR maps ({hp_obj.name} -> {lp_obj.name})..."
        hp_mesh_path = _export_active_object_mesh(hp_obj)
        lp_mesh_path = _export_active_object_mesh(lp_obj)
        server_url = props.server_url
        res = int(props.finishing_bake_res)

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/bake",
                data=json.dumps({
                    "high_poly_path": hp_mesh_path,
                    "low_poly_path": lp_mesh_path,
                    "bake_resolution": res,
                    "bake_maps": ["normal", "base_color", "ao", "roughness", "metallic"]
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            maps = data.get("baked_maps", {})
            props.finishing_status = f"Baking complete ({len(maps)} maps baked at {res}x{res})."
            self.report({'INFO'}, f"Baked {len(maps)} texture maps.")

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"Bake failed: {err}"
            self.report({'ERROR'}, f"Baking error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_GenerateLODSet(bpy.types.Operator):
    bl_idname = "localmesh.generate_lod_set"
    bl_label = "Generate LOD0-LOD3 Set"
    bl_description = "Produces multi-tier Level-of-Detail assets with edge-preserving silhouette decimation"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Generating LOD0-LOD3 set..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/lods",
                data=json.dumps({"mesh_path": temp_mesh}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            summary = data.get("summary", "LODs generated")
            props.finishing_status = summary
            self.report({'INFO'}, summary)

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"LOD generation failed: {err}"
            self.report({'ERROR'}, f"LOD error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_ExportFinishedAsset(bpy.types.Operator):
    bl_idname = "localmesh.export_finished_asset"
    bl_label = "Export & Validate (GLB / FBX / OBJ)"
    bl_description = "Exports multi-format package with structural re-import verification and asset_manifest.json"

    def execute(self, context):
        props = context.scene.local_mesh_props
        act = context.active_object
        if not act or act.type != 'MESH':
            self.report({'ERROR'}, "Select active mesh object.")
            return {'CANCELLED'}

        props.finishing_is_busy = True
        props.finishing_status = "Exporting and verifying asset packages..."
        temp_mesh = _export_active_object_mesh(act)
        server_url = props.server_url

        fmts = []
        if props.finishing_export_glb:
            fmts.append("glb")
        if props.finishing_export_fbx:
            fmts.append("fbx")
        if props.finishing_export_obj:
            fmts.append("obj")
        if not fmts:
            fmts = ["glb"]

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/finishing/export",
                data=json.dumps({
                    "mesh_path": temp_mesh,
                    "asset_name": act.name,
                    "formats": fmts
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.finishing_is_busy = False
            exp_files = data.get("exported_files", {})
            m_path = data.get("manifest_path", "")
            summary = f"Exported {len(exp_files)} formats ({', '.join(exp_files.keys())}). Manifest: {os.path.basename(m_path)}"
            props.finishing_export_summary = summary
            props.finishing_status = summary
            self.report({'INFO'}, summary)

        def _error(err):
            props.finishing_is_busy = False
            props.finishing_status = f"Export failed: {err}"
            self.report({'ERROR'}, f"Export error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_PT_FinishingStudio(bpy.types.Panel):
    bl_label = "Panel 9 — 3D Finishing, Retopo & PBR Studio"
    bl_idname = "LOCALMESH_PT_FinishingStudio"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LOCAL MESH AI'

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props
        act = context.active_object

        # Status row
        box = layout.box()
        row = box.row(align=True)
        if props.finishing_is_busy:
            row.label(text=props.finishing_status, icon='TIME')
        else:
            row.label(text=f"Status: {props.finishing_status}", icon='CHECKMARK')

        # Section 1: Mesh Health & Validation
        box_health = layout.box()
        box_health.label(text="1. Mesh Health & Validation", icon='HELP')
        col = box_health.column(align=True)
        col.operator("localmesh.inspect_mesh_health", icon='VIEWZOOM')
        if props.mesh_health_summary != "Not inspected":
            col.label(text=props.mesh_health_summary, icon='INFO')
            if props.mesh_is_game_ready:
                col.label(text="✅ Certified Game-Ready", icon='CHECKMARK')
            else:
                col.label(text="⚠️ Optimization recommended", icon='ERROR')
        col.operator("localmesh.repair_active_mesh", icon='TOOL_SETTINGS')

        # Section 2: Retopology & Low-Poly Simplification
        box_retopo = layout.box()
        box_retopo.label(text="2. Retopology & Low-Poly", icon='MESH_DATA')
        col = box_retopo.column(align=True)
        col.prop(props, "finishing_retopo_quads")
        col.prop(props, "finishing_symmetry")
        col.prop(props, "finishing_preserve_sharp")
        col.operator("localmesh.execute_retopology", icon='MOD_REMESH')

        # Section 3: Professional UV Unwrapping
        box_uv = layout.box()
        box_uv.label(text="3. UV Unwrapping & Packing", icon='UV')
        col = box_uv.column(align=True)
        col.prop(props, "finishing_uv_method")
        col.prop(props, "finishing_uv_margin")
        col.operator("localmesh.unwrap_active_mesh", icon='UV_DATA')
        if props.finishing_uv_summary != "No UV data":
            col.label(text=props.finishing_uv_summary, icon='INFO')

        # Section 4: PBR Materials & Presets
        box_mat = layout.box()
        box_mat.label(text="4. PBR Materials & Shaders", icon='MATERIAL')
        col = box_mat.column(align=True)
        col.prop(props, "finishing_mat_preset")
        col.prop(props, "finishing_mat_metallic", slider=True)
        col.prop(props, "finishing_mat_roughness", slider=True)
        col.operator("localmesh.apply_pbr_materials", icon='NODE_MATERIAL')

        # Section 5: Cycles Texture Baking
        box_bake = layout.box()
        box_bake.label(text="5. Cycles Selected-to-Active Bake", icon='RENDER_STILL')
        col = box_bake.column(align=True)
        col.prop_search(props, "finishing_bake_high_poly", context.scene, "objects", text="High-Poly")
        col.prop_search(props, "finishing_bake_low_poly", context.scene, "objects", text="Low-Poly")
        col.prop(props, "finishing_bake_res")
        col.operator("localmesh.bake_textures_cycles", icon='TEXTURE')

        # Section 6: LODs & Multi-Format Export
        box_exp = layout.box()
        box_exp.label(text="6. LODs & Game-Ready Export", icon='EXPORT')
        col = box_exp.column(align=True)
        col.operator("localmesh.generate_lod_set", icon='MOD_MULTIRES')
        row_fmt = col.row(align=True)
        row_fmt.prop(props, "finishing_export_glb")
        row_fmt.prop(props, "finishing_export_fbx")
        row_fmt.prop(props, "finishing_export_obj")
        col.operator("localmesh.export_finished_asset", icon='PACKAGE')
        if props.finishing_export_summary:
            col.label(text=props.finishing_export_summary, icon='CHECKMARK')


# ==============================================================================
# Phase 9 Operators & Panel: Autonomous Health & Model Lifecycle
# ==============================================================================

class LOCALMESH_OT_RunSystemDiagnostics(bpy.types.Operator):
    bl_idname = "localmesh.run_system_diagnostics"
    bl_label = "Diagnose System Health"
    bl_description = "Collects real-time hardware, GPU, memory, and Blender status"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url
        props.health_is_busy = True

        def _task():
            req = urllib.request.Request(f"{server_url}/api/v1/health/diagnostics")
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            props.health_is_busy = False
            overall = data.get("overall_status", "healthy")
            issues = data.get("active_issues", [])
            props.health_status_text = overall.capitalize()
            props.health_issue_count = len(issues)
            self.report({'INFO'}, f"System Health: {overall} ({len(issues)} issues)")

        def _error(err):
            props.health_is_busy = False
            props.health_status_text = "Check Failed"
            self.report({'ERROR'}, f"Diagnostics error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_PreviewRecovery(bpy.types.Operator):
    bl_idname = "localmesh.preview_recovery"
    bl_label = "1. Preview Recovery (Dry Run)"
    bl_description = "Performs safe dry run of recovery action to inspect affected files without deleting"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/health/recover",
                data=json.dumps({
                    "action_type": "clean_staging_dir",
                    "approved": False,
                    "dry_run": True
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            affected = data.get("affected_files", [])
            preview_id = data.get("preview_id", "")
            expires_at = data.get("expires_at", 0.0)
            props.recovery_preview_ready = True
            props.recovery_preview_id = preview_id
            props.recovery_expires_at = float(expires_at)
            props.recovery_preview_text = f"Dry Run: {len(affected)} files identified (ID: {preview_id[:8]}...)" if preview_id else f"Dry Run: {len(affected)} files identified"
            props.recovery_affected_files = "; ".join([os.path.basename(f) for f in affected[:10]])
            if len(affected) > 10:
                props.recovery_affected_files += f" (+{len(affected)-10} more)"
            self.report({'INFO'}, f"Recovery Preview: {len(affected)} files identified (ID: {preview_id[:8]})")

        def _error(err):
            props.recovery_preview_ready = False
            self.report({'ERROR'}, f"Preview recovery error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_ExecuteApprovedRecovery(bpy.types.Operator):
    bl_idname = "localmesh.execute_approved_recovery"
    bl_label = "2. Execute Approved Recovery"
    bl_description = "Executes explicit user-approved remediation on verified staging files"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url

        def _task():
            payload = {
                "action_type": "clean_staging_dir",
                "approved": True
            }
            if props.recovery_preview_id:
                payload["preview_id"] = props.recovery_preview_id
            req = urllib.request.Request(
                f"{server_url}/api/v1/health/recover",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            msg = data.get("message", "Remediation executed successfully")
            props.recovery_preview_ready = False
            props.recovery_preview_text = ""
            props.health_status_text = "Healthy"
            props.health_issue_count = 0
            self.report({'INFO'}, msg)

        def _error(err):
            self.report({'ERROR'}, f"Approved recovery error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_AutoSelfHeal(bpy.types.Operator):
    bl_idname = "localmesh.auto_self_heal"
    bl_label = "Safe Recovery Preview"
    bl_description = "Previews safe remediation actions"

    def execute(self, context):
        return bpy.ops.localmesh.preview_recovery()


class LOCALMESH_OT_ClearLearningData(bpy.types.Operator):
    bl_idname = "localmesh.clear_learning_data"
    bl_label = "Purge All Learning Data"
    bl_description = "Permanently removes all locally collected feedback, training pairs, and telemetry"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/learning/clear",
                data=json.dumps({}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            purged = data.get("purged_count", 0)
            self.report({'INFO'}, f"Purged {purged} learning samples.")

        def _error(err):
            self.report({'ERROR'}, f"Clear learning data error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_EvaluateCandidateModel(bpy.types.Operator):
    bl_idname = "localmesh.evaluate_candidate_model"
    bl_label = "Evaluate Candidate Model"
    bl_description = "Executes genuine benchmark validation on the active candidate adapter checkpoint"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url
        cid = props.candidate_model_name
        if not cid or cid.lower() in ("none", "null", ""):
            self.report({'WARNING'}, "No candidate model adapter available to evaluate.")
            return {'CANCELLED'}

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/learning/evaluate",
                data=json.dumps({"candidate_name": cid}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            score = float(data.get("score_percent", 0.0))
            passed = bool(data.get("passed_evaluation", False))
            props.model_eval_score = score
            props.model_eval_passed = passed
            st = "PASSED" if passed else "FAILED"
            self.report({'INFO'}, f"Candidate Evaluation {st}: Score {score:.1f}%")

        def _error(err):
            self.report({'ERROR'}, f"Candidate evaluation error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_OT_ToggleLearningConsent(bpy.types.Operator):
    bl_idname = "localmesh.toggle_learning_consent"
    bl_label = "Update Privacy Preferences"
    bl_description = "Updates strictly local opt-in consent for learning data collection"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url
        opted = props.learning_consent_opted_in

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/learning/consent",
                data=json.dumps({"opted_in": opted}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            st = "Opted IN (Strictly Local)" if data.get("opted_in") else "Opted OUT"
            self.report({'INFO'}, f"Consent updated: {st}")

        def _error(err):
            self.report({'ERROR'}, f"Consent error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}



class LOCALMESH_OT_PromoteCandidateModel(bpy.types.Operator):
    bl_idname = "localmesh.promote_candidate_model"
    bl_label = "Promote Candidate to Active"
    bl_description = "Promotes the evaluated candidate model adapter to active status"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url
        cid = props.candidate_model_name
        if not cid or cid.lower() in ("none", "null", ""):
            self.report({'WARNING'}, "No candidate model adapter available to promote.")
            return {'CANCELLED'}

        if not props.model_eval_passed or props.model_eval_score < 80.0:
            self.report({'ERROR'}, f"Cannot promote candidate '{cid}': must pass evaluation with score >= 80% (got {props.model_eval_score:.1f}%).")
            return {'CANCELLED'}

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/learning/models/promote",
                data=json.dumps({"model_id": cid, "approved": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            promoted_name = data.get("model_name") or data.get("model_id") or cid
            props.active_model_name = promoted_name
            props.candidate_model_name = "None"
            props.model_eval_passed = False
            self.report({'INFO'}, f"Successfully promoted candidate to active model: {promoted_name}")

        def _error(err):
            self.report({'ERROR'}, f"Model promotion failed: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}

class LOCALMESH_OT_RollbackModelLifecycle(bpy.types.Operator):
    bl_idname = "localmesh.rollback_model_lifecycle"
    bl_label = "Rollback Model Adapter"
    bl_description = "Atomically rolls back active AI model to previously verified version"

    def execute(self, context):
        props = context.scene.local_mesh_props
        server_url = props.server_url

        def _task():
            req = urllib.request.Request(
                f"{server_url}/api/v1/learning/models/rollback",
                data=json.dumps({"reason": "User triggered rollback from Blender"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _success(data):
            restored = data.get("restored_active_id", "base")
            props.active_model_name = restored
            self.report({'INFO'}, f"Model rolled back to: {restored}")

        def _error(err):
            self.report({'ERROR'}, f"Rollback error: {err}")

        run_in_background(_task, _success, _error)
        return {'FINISHED'}


class LOCALMESH_PT_SystemHealthLearning(bpy.types.Panel):
    bl_label = "Panel 10 — Health, Healing & Model Lifecycle"
    bl_idname = "LOCALMESH_PT_SystemHealthLearning"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LOCAL MESH AI'

    def draw(self, context):
        layout = self.layout
        props = context.scene.local_mesh_props

        # Section 1: System Health & Diagnostics
        box_health = layout.box()
        box_health.label(text="1. Runtime Health & Hardware", icon='SYSTEM')
        col = box_health.column(align=True)
        col.operator("localmesh.run_system_diagnostics", icon='VIEWZOOM')
        row = col.row(align=True)
        row.label(text=f"Status: {props.health_status_text}", icon='INFO')
        if props.health_issue_count > 0:
            row.label(text=f"Issues: {props.health_issue_count}", icon='ERROR')
        
        # Two-stage remediation
        col.separator()
        col.operator("localmesh.preview_recovery", icon='FILE_REFRESH', text="1. Preview Safe Recovery (Dry Run)")
        if props.recovery_preview_ready:
            prev_box = col.box()
            prev_box.label(text=props.recovery_preview_text, icon='INFO')
            if props.recovery_preview_id:
                prev_box.label(text=f"Preview ID: {props.recovery_preview_id[:16]}...", icon='KEY_HLT')
            if props.recovery_affected_files:
                for f_name in props.recovery_affected_files.split("; "):
                    if f_name:
                        prev_box.label(text=f_name, icon='FILE')
            prev_box.operator("localmesh.execute_approved_recovery", icon='CHECKMARK', text="2. Execute Approved Recovery")

        # Section 2: Privacy & Local Learning
        box_learn = layout.box()
        box_learn.label(text="2. Privacy & Local Learning (Opt-In)", icon='LOCKED')
        col = box_learn.column(align=True)
        col.prop(props, "learning_consent_opted_in", text="Allow Local Data Collection")
        col.operator("localmesh.toggle_learning_consent", icon='FILE_REFRESH')
        col.operator("localmesh.clear_learning_data", icon='TRASH', text="Purge All Learning Data")
        
        # Preflight Hardware Feasibility
        col.separator()
        box_pref = col.box()
        if props.is_server_online:
            box_pref.label(text=f"Free VRAM: {props.preflight_free_vram:.0f} MB", icon='PREFERENCES')
            box_pref.label(text=f"Training Feasibility: {props.preflight_feasibility}", icon='INFO')
        else:
            box_pref.label(text="Telemetry Offline (Start Backend)", icon='INFO')

        # Section 3: Model Version & Rollback
        box_model = layout.box()
        box_model.label(text="3. Model Lifecycle & Rollback", icon='FILE_PARENT')
        col = box_model.column(align=True)
        col.label(text=f"Active: {props.active_model_name}")
        col.label(text=f"Candidate: {props.candidate_model_name}")
        col.operator("localmesh.evaluate_candidate_model", icon='CHECKMARK')
        if props.model_eval_score > 0.0:
            st_icon = 'CHECKMARK' if props.model_eval_passed else 'ERROR'
            st_text = 'PASSED' if props.model_eval_passed else 'FAILED'
            col.label(text=f"Eval Score: {props.model_eval_score:.1f}% ({st_text})", icon=st_icon)
        
        # Explicit promotion button only after passing evaluation
        if props.candidate_model_name != "None" and props.model_eval_passed:
            col.operator("localmesh.promote_candidate_model", icon='FORWARD', text="Promote Candidate to Active")
        col.separator()
        col.operator("localmesh.rollback_model_lifecycle", icon='RECOVER_LAST')

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
    LOCALMESH_OT_StageActiveAsset,
    LOCALMESH_OT_AnalyzeActiveAsset,
    LOCALMESH_OT_GenerateArtistPlan,
    LOCALMESH_OT_ConfirmTargetComponent,
    LOCALMESH_OT_ApplyArtistPlan,
    LOCALMESH_OT_RollbackArtistEdit,
    LOCALMESH_PT_AITechnicalArtist,
    LOCALMESH_OT_InspectMeshHealth,
    LOCALMESH_OT_RepairActiveMesh,
    LOCALMESH_OT_ExecuteRetopology,
    LOCALMESH_OT_UnwrapActiveMesh,
    LOCALMESH_OT_ApplyPBRMaterials,
    LOCALMESH_OT_BakeTexturesCycles,
    LOCALMESH_OT_GenerateLODSet,
    LOCALMESH_OT_ExportFinishedAsset,
    LOCALMESH_PT_FinishingStudio,
    LOCALMESH_OT_RunSystemDiagnostics,
    LOCALMESH_OT_PreviewRecovery,
    LOCALMESH_OT_ExecuteApprovedRecovery,
    LOCALMESH_OT_AutoSelfHeal,
    LOCALMESH_OT_ClearLearningData,
    LOCALMESH_OT_EvaluateCandidateModel,
    LOCALMESH_OT_PromoteCandidateModel,
    LOCALMESH_OT_ToggleLearningConsent,
    LOCALMESH_OT_RollbackModelLifecycle,
    LOCALMESH_PT_SystemHealthLearning,
    LOCALMESH_PT_MeshControls
)

def _auto_lifecycle_check():
    """
    Automatic Backend Lifecycle:
    When Blender starts and the addon registers:
    1. Checks if a healthy backend already exists on server_url (default 8000).
    2. If healthy, reuses it without spawning duplicate processes.
    3. If offline, automatically spawns an application-owned backend console passing --parent-pid.
    4. Only this Blender session's spawned backend process is tracked and terminated on shutdown.
    """
    try:
        url = DEFAULT_SERVER_URL
        if hasattr(bpy.context, "scene"):
            props = getattr(bpy.context.scene, "local_mesh_props", None) or getattr(bpy.context.scene, "triposr_props", None)
            if props and props.server_url:
                url = props.server_url

        online, latency, _ = check_server_health(url, timeout=0.6)
        if online:
            if hasattr(bpy.context, "scene"):
                props = getattr(bpy.context.scene, "local_mesh_props", None) or getattr(bpy.context.scene, "triposr_props", None)
                if props:
                    props.is_server_online = True
                    props.server_latency_ms = latency
            return None

        root = find_project_root()
        pref = get_addon_preferences()
        if pref and pref.project_dir and os.path.exists(pref.project_dir):
            root = pref.project_dir

        success, msg = launch_backend_console(root)
        print(f"[Local Mesh AI Lifecycle] Auto-starting backend: {msg}")
    except Exception as e:
        print(f"[Local Mesh AI Lifecycle] Auto-start check encountered: {e}")
    return None


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

    if not bpy.app.timers.is_registered(_auto_lifecycle_check):
        bpy.app.timers.register(_auto_lifecycle_check, first_interval=0.5)
    if not bpy.app.timers.is_registered(_periodic_server_check):
        bpy.app.timers.register(_periodic_server_check, first_interval=1.5)
    if not bpy.app.timers.is_registered(_process_addon_result_queue):
        bpy.app.timers.register(_process_addon_result_queue, persistent=True)


def unregister():
    try:
        if bpy.app.timers.is_registered(_process_addon_result_queue):
            bpy.app.timers.unregister(_process_addon_result_queue)
    except Exception:
        pass
    try:
        stop_backend_console()
    except Exception:
        pass

    if bpy.app.timers.is_registered(_auto_lifecycle_check):
        bpy.app.timers.unregister(_auto_lifecycle_check)
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
