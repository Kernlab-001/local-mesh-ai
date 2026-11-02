"""
End-to-end acceptance test: the real Blender Generate button.

This is the acceptance test required by the recovery directive. It does NOT
mock anything: it runs inside a real Blender 5.2.3 process, presses the actual
``localmesh.generate_asset`` operator, waits for the real backend job to
finish and verifies that the produced asset was imported into the active
scene.

Run it with a real (non-background) Blender so that the add-on's timers fire::

    blender.exe --python tests/e2e_blender_generate.py

Optional environment variables:

    LMA_E2E_IMAGE      image to generate from (default: examples/chair.png)
    LMA_E2E_TIMEOUT    seconds to wait for the job (default: 420)
    LMA_E2E_OUTPUT     evidence directory (default: <project>/output/e2e)

The script writes ``evidence.json`` plus a saved ``.blend`` and a viewport
screenshot, and prints ``E2E_*`` markers that the test runner greps for.
"""

import json
import os
import sys
import time

import bpy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_DIR = os.environ.get("LMA_E2E_OUTPUT") or os.path.join(PROJECT_ROOT, "output", "e2e")
TIMEOUT = float(os.environ.get("LMA_E2E_TIMEOUT") or 420)
IMAGE = os.environ.get("LMA_E2E_IMAGE") or os.path.join(PROJECT_ROOT, "examples", "chair.png")

EVIDENCE = {
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "blender_version": bpy.app.version_string,
    "image": IMAGE,
    "steps": [],
}


def step(name, **data):
    entry = {"step": name, "at": time.strftime("%H:%M:%S")}
    entry.update(data)
    EVIDENCE["steps"].append(entry)
    print(f"E2E_STEP {name}: {json.dumps(data, default=str)}")
    sys.stdout.flush()


def finish(status):
    EVIDENCE["status"] = status
    EVIDENCE["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(os.path.join(EVIDENCE_DIR, "evidence.json"), "w", encoding="utf-8") as handle:
            json.dump(EVIDENCE, handle, indent=2, default=str)
    except Exception as exc:
        print(f"E2E_ERROR could not write evidence: {exc}")
    print(f"E2E_STATUS {status}")
    sys.stdout.flush()
    bpy.ops.wm.quit_blender()


def enable_addon():
    import addon_utils
    if "local_mesh_ai" in bpy.context.preferences.addons:
        step("addon_already_enabled")
        return
    module = addon_utils.enable("local_mesh_ai", default_set=False, persistent=True)
    step("addon_enabled", module=getattr(module, "__name__", None))
    return module


def configure_props(props):
    props.image_path = IMAGE
    props.input_mode = 'image'
    props.quality = 'draft'
    props.preset = 'furniture'
    props.hardware_profile = 'low_memory'
    props.generation_route = 'image_to_3d_triposr'
    props.target_height = 0.9
    props.remove_bg = False
    props.sanitize_mesh = False
    props.sharp_overlay = False
    props.adaptive_weld = False
    props.pbr_material = False
    props.accurate_colors = False
    props.retopology_and_bake = False
    props.generate_lods = False
    props.auto_rig = False
    step("props_configured", image=props.image_path, quality=props.quality,
         route=props.generation_route)


def scene_meshes():
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']


def poll(props, state):
    """Timer callback: waits for the job and finishes the test."""
    deadline = state["deadline"]
    if time.time() > deadline:
        step("timeout", status=props.job_status, message=props.status_message)
        finish("TIMEOUT")
        return None

    job_id = props.active_job_id
    if job_id and not state["logged_submit"]:
        state["logged_submit"] = True
        step("job_created", job_id=job_id)

    if job_id and props.job_status and props.job_status != state.get("last_status"):
        state["last_status"] = props.job_status
        step("job_status", status=props.job_status, stage=props.stage_name,
             progress=props.progress_pct, message=props.status_message)

    finished = (
        job_id
        and not props.is_generating
        and (state.get("saw_active") or bool(props.last_completed_mesh))
    )
    if finished:
        step("job_finished", status=props.job_status, message=props.status_message,
             mesh=props.last_completed_mesh)
        collect_evidence(props)
        finish("PASS" if len(scene_meshes()) > 0 else "FAIL_NO_MESH")
        return None

    if props.is_generating:
        state["saw_active"] = True
    return 1.0


def collect_evidence(props):
    meshes = scene_meshes()
    step("scene_contains_meshes", count=len(meshes))
    for obj in meshes:
        try:
            step("mesh_object", object_name=obj.name, vertices=len(obj.data.vertices),
                 polygons=len(obj.data.polygons),
                 collections=[c.name for c in obj.users_collection])
        except Exception as exc:
            step("mesh_object_error", object_name=obj.name, error=str(exc))

    try:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        blend_path = os.path.join(EVIDENCE_DIR, "e2e_generated_asset.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        step("blend_saved", path=blend_path)
    except Exception as exc:
        step("blend_save_failed", error=str(exc))

    try:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        shot = os.path.join(EVIDENCE_DIR, "e2e_viewport.png")
        bpy.ops.screen.screenshot(filepath=shot)
        step("screenshot_saved", path=shot)
    except Exception as exc:
        step("screenshot_failed", error=str(exc))


def main():
    step("start", image=IMAGE, timeout=TIMEOUT)
    if not os.path.exists(IMAGE):
        step("missing_image", path=IMAGE)
        finish("FAIL_NO_IMAGE")
        return

    try:
        enable_addon()
    except Exception as exc:
        step("addon_enable_failed", error=str(exc))
        finish("FAIL_ADDON")
        return

    props = getattr(bpy.context.scene, "local_mesh_props", None)
    if props is None:
        step("no_scene_props")
        finish("FAIL_NO_PROPS")
        return

    configure_props(props)

    # Press the REAL Generate button.
    try:
        result = bpy.ops.localmesh.generate_asset()
        step("generate_pressed", result=list(result))
    except Exception as exc:
        step("generate_raised", error=str(exc))
        finish("FAIL_OPERATOR")
        return

    state = {"deadline": time.time() + TIMEOUT, "saw_active": False,
             "logged_submit": False, "last_status": None}
    bpy.app.timers.register(lambda: poll(props, state), first_interval=1.0)


main()
