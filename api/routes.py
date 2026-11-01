"""
FastAPI Routes for Local Neural 3D Studio.
Hardened with input validation, path safety, and secure error boundaries.
"""

import os
import shutil
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import uuid
import time
from pathlib import Path
from api.schemas import (
    StageImageResponse,
    SceneSyncRequest,
    SceneSyncResponse,
    GenerateTextRequest,
    GenerateImageRequest,
    JobProgressResponse,
    RouteInfoResponse,
    AssetResponse,
    CreateJobRequest,
    JobResponse,
    EditJobRequest,
    ExportJobRequest,
    AnalyzeGeometryRequest,
    DecomposeComponentsRequest,
    AlignAndGroundRequest,
    BoundedCorrectionRequest,
    ExecuteToolRequest,
    StageAssetRequest,
    StageAssetResponse,
    AnalyzeAssetRequest,
    AnalyzeAssetResponse,
    GeneratePlanRequest,
    GeneratePlanResponse,
    ExecutePlanRequest,
    ExecutePlanResponse,
    RollbackRequest,
    RollbackResponse,
    SessionStateResponse,
    ComponentSummary
)
from services.job_manager import job_manager
from services.export_service import export_service
from services.editing_session_manager import editing_session_manager
from core.gpu_broker import gpu_broker
from core.qwen_planner import qwen_planner
from core.hardware_detector import hardware_detector
from core.engines.registry import engine_registry
from core.model_manager import model_manager
from core.geometry_validator import geometry_validator
from core.geometry_analyzer import geometry_analyzer
from core.component_decomposer import component_decomposer
from core.precision_aligner import precision_aligner
from core.tool_registry import tool_registry
from core.bounded_correction_loop import bounded_correction_loop
from core.generation_orchestrator import generation_orchestrator, RouteUnavailableError
from core.model_paths import model_path_resolver

router = APIRouter()

MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

def validate_job_id(job_id: str):
    if not job_id or ".." in job_id or "/" in job_id or "\\" in job_id or len(job_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid job ID format.")

@router.get("/health")
def health_check():
    gpu_status = gpu_broker.get_status()
    blender_path = hardware_detector.find_blender() or "blender"
    blender_installed = bool(blender_path and (os.path.exists(blender_path) or shutil.which(blender_path)))

    qwen_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen2.5-coder-7b-instruct-q4_k_m.gguf")

    return {
        "status": "healthy",
        "gpu": gpu_status,
        "blender": {
            "installed": blender_installed,
            "version": "Detected" if blender_installed else "Not Installed",
            "path": blender_path
        },
        "qwen_model": {
            "present": os.path.exists(qwen_path),
            "path": qwen_path
        }
    }

@router.get("/engines")
def list_available_engines():
    """Returns all registered 3D neural generation engines and live status."""
    return {
        "engines": engine_registry.list_engines(),
        "hardware": engine_registry.get_hardware_telemetry()
    }

@router.get("/hardware")
def get_hardware_telemetry():
    """Returns detailed hardware profile and memory allocation metrics."""
    return engine_registry.get_hardware_telemetry()

@router.get("/models")
def list_model_catalog():
    """Returns local model catalog and storage capacity report."""
    catalog = [m.__dict__ for m in model_manager.get_model_catalog()]
    return {
        "models": catalog,
        "free_disk_space_gb": model_manager.get_free_disk_space_gb()
    }

@router.post("/jobs", response_model=JobResponse)
def submit_job(req: CreateJobRequest):
    if not req.image_path and not req.prompt:
        raise HTTPException(status_code=400, detail="Either image_path or prompt must be provided.")

    if req.image_path:
        resolved_img = os.path.abspath(req.image_path)
        if not os.path.exists(resolved_img):
            raise HTTPException(status_code=400, detail=f"Image file does not exist: {req.image_path}")
        if os.path.getsize(resolved_img) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"Image size exceeds 25 MB limit.")

    job = job_manager.create_job(req.model_dump())
    return job.to_dict()

@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()

@router.get("/jobs/{job_id}/mesh")
def get_job_mesh(job_id: str):
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not job.output_mesh or not os.path.exists(job.output_mesh) or os.path.getsize(job.output_mesh) == 0:
        raise HTTPException(status_code=404, detail=f"3D mesh asset not available or job incomplete.")

    media_type = "model/gltf-binary"
    if job.output_mesh.lower().endswith(".obj"):
        media_type = "model/obj"

    return FileResponse(
        path=job.output_mesh,
        media_type=media_type,
        filename=os.path.basename(job.output_mesh)
    )

@router.get("/jobs/{job_id}/preview")
def get_job_preview(job_id: str):
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job or not job.preview_png or not os.path.exists(job.preview_png) or os.path.getsize(job.preview_png) == 0:
        raise HTTPException(status_code=404, detail=f"Preview render not available for job '{job_id}'.")
    return FileResponse(path=job.preview_png, media_type="image/png", filename="preview.png")

@router.get("/jobs/{job_id}/blend")
def get_job_blend(job_id: str):
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job or not job.output_blend or not os.path.exists(job.output_blend) or os.path.getsize(job.output_blend) == 0:
        raise HTTPException(status_code=404, detail=f"Blender project not available for job '{job_id}'.")
    return FileResponse(path=job.output_blend, media_type="application/x-blender", filename=f"{job_id}_project.blend")

@router.get("/jobs/{job_id}/concept")
def get_job_concept(job_id: str):
    validate_job_id(job_id)
    job_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jobs", job_id)
    for name in ("concept.png", "concept_art.png", "source_image.png"):
        p = os.path.join(job_dir, "input", name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return FileResponse(path=p, media_type="image/png", filename=name)
    raise HTTPException(status_code=404, detail=f"Concept image not available for job '{job_id}'.")

@router.get("/jobs/{job_id}/multiview")
def get_job_multiview(job_id: str):
    validate_job_id(job_id)
    job_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jobs", job_id)
    for name in ("multiview_grid.png", "composite_grid.png"):
        p = os.path.join(job_dir, "multiview", name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return FileResponse(path=p, media_type="image/png", filename=name)
    raise HTTPException(status_code=404, detail=f"Multi-view grid not available for job '{job_id}'.")

@router.get("/viewer", response_class=HTMLResponse)
def serve_3d_viewer():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    viewer_path = os.path.join(project_root, "ui", "static", "viewer.html")
    if not os.path.exists(viewer_path):
        raise HTTPException(status_code=404, detail="Viewer template not found.")
    with open(viewer_path, "r", encoding="utf-8") as f:
        return f.read()

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    validate_job_id(job_id)
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job '{job_id}'. Either not found or already finished.")
    return {"status": "cancelled", "job_id": job_id}

@router.post("/jobs/{job_id}/edit")
def edit_job_with_qwen(job_id: str, req: EditJobRequest):
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job or not job.output_blend or not os.path.exists(job.output_blend) or os.path.getsize(job.output_blend) == 0:
        raise HTTPException(status_code=400, detail="Job blend project not available for editing.")

    # 1. Ask Qwen for validated CommandPlan
    try:
        plan = qwen_planner.generate_plan(user_prompt=req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen planning failed: {str(e)}")

    # 2. Execute plan deterministically in Blender
    blend_in = job.output_blend
    blend_out = os.path.join(os.path.dirname(blend_in), f"project_edited_{job_id}.blend")
    temp_plan = os.path.join(os.path.dirname(blend_in), f"plan_{job_id}.json")

    with open(temp_plan, "w") as f:
        f.write(plan.model_dump_json())

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    executor_script = os.path.join(project_root, "core", "blender_executor.py")

    blender_exe = hardware_detector.find_blender() or "blender"
    cmd = [
        blender_exe, "--background",
        "--python", executor_script,
        "--", temp_plan, blend_in, blend_out
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180.0)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Blender script execution timed out after 180 seconds.")

    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Blender executor failed:\n{proc.stderr}")

    return {
        "status": "success",
        "plan": plan.model_dump(),
        "edited_blend": blend_out
    }

@router.post("/jobs/{job_id}/export")
def export_job_asset(job_id: str, req: ExportJobRequest):
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job or not job.output_blend or not os.path.exists(job.output_blend) or os.path.getsize(job.output_blend) == 0:
        raise HTTPException(status_code=400, detail="Job blend project not available for export.")

    clean_name = export_service.sanitize_filename(req.output_name)
    try:
        res = export_service.export_asset(
            blend_path=job.output_blend,
            format_type=req.format,
            output_name=f"{clean_name}_{job_id}"
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# ==============================================================================
# Phase 2 Geometry Analysis, Component Decomposition & Precision Alignment
# ==============================================================================

@router.post("/api/geometry/analyze")
def analyze_mesh_geometry(req: AnalyzeGeometryRequest):
    """Inspects mesh data and returns structured metrics and topology diagnostics."""
    if not os.path.exists(req.mesh_path):
        raise HTTPException(status_code=404, detail=f"Mesh file not found: {req.mesh_path}")
    try:
        report = geometry_analyzer.analyze_mesh(req.mesh_path)
        return report.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geometry analysis failed: {str(e)}")

@router.post("/api/geometry/decompose")
def decompose_mesh_components(req: DecomposeComponentsRequest):
    """Decomposes mesh into candidate geometric components and constructs a hierarchy."""
    if not os.path.exists(req.mesh_path):
        raise HTTPException(status_code=404, detail=f"Mesh file not found: {req.mesh_path}")
    try:
        hierarchy = component_decomposer.decompose(
            mesh_or_path=req.mesh_path,
            asset_id=req.asset_id,
            method=req.method,
            min_face_threshold=req.min_faces,
            output_dir=req.output_dir
        )
        return hierarchy.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Component decomposition failed: {str(e)}")

@router.post("/api/geometry/align")
def align_and_calibrate_mesh(req: AlignAndGroundRequest):
    """Applies precision grounding, orientation normalization, and metric scale calibration."""
    if not os.path.exists(req.mesh_path):
        raise HTTPException(status_code=404, detail=f"Mesh file not found: {req.mesh_path}")
    try:
        mesh, result = precision_aligner.execute_alignment(
            mesh_or_path=req.mesh_path,
            output_path=req.output_path,
            ground_z=req.ground_z,
            center_xy=req.center_xy,
            normalize_orientation=req.normalize_orientation,
            target_height_m=req.target_height_m,
            enforce_symmetry_plane=req.enforce_symmetry_plane
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Precision alignment failed: {str(e)}")

@router.post("/api/geometry/correct")
def run_bounded_correction(req: BoundedCorrectionRequest):
    """Executes bounded iterative feedback loop: Plan -> Validate -> Execute -> Measure -> Evaluate."""
    if not os.path.exists(req.mesh_path):
        raise HTTPException(status_code=404, detail=f"Mesh file not found: {req.mesh_path}")
    try:
        report = bounded_correction_loop.run_loop(
            mesh_path=req.mesh_path,
            target_height_m=req.target_height_m,
            target_upright=req.target_upright,
            target_grounded=req.target_grounded,
            max_attempts=req.max_attempts
        )
        return report.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bounded correction failed: {str(e)}")

@router.get("/api/tools")
def list_allowlisted_tools():
    """Returns all safe, allowlisted geometry tools."""
    return [t.model_dump() for t in tool_registry.list_tools()]

@router.post("/api/tools/execute")
def execute_allowlisted_tool(req: ExecuteToolRequest):
    """Executes a strictly allowlisted tool under schema validation."""
    try:
        res = tool_registry.execute_tool(req.tool_name, req.arguments)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e)}")



# ==============================================================================
# Phase 4 Unified Generation Pipeline Endpoints
# ==============================================================================

@router.get("/api/v1/routes")
def get_supported_routes():
    """Returns supported 3D generation routes and hardware profiles with live readiness."""
    return {
        "routes": generation_orchestrator.discover_routes(),
        "hardware_profiles": generation_orchestrator.get_hardware_profiles()
    }

@router.get("/api/v1/capabilities")
def get_generation_capabilities():
    """Returns truthful local AI model availability and hardware orchestration limits."""
    return {
        "models": model_path_resolver.get_model_inventory(),
        "routes": generation_orchestrator.discover_routes(),
        "hardware": engine_registry.get_hardware_telemetry()
    }

@router.post("/api/v1/generate/text", response_model=JobResponse)
def generate_from_text(req: GenerateTextRequest):
    """Submits a text-conditioned 3D asset generation request."""
    params = req.model_dump()
    params["input_mode"] = "prompt"

    valid, err, clean_params = generation_orchestrator.validate_request("text-to-3d", params)
    if not valid:
        raise HTTPException(status_code=400, detail=err)

    try:
        generation_orchestrator.resolve_route("text-to-3d", clean_params)
    except RouteUnavailableError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = job_manager.create_job(clean_params)
    return job.to_dict()

@router.post("/api/v1/generate/image", response_model=JobResponse)
def generate_from_image(req: GenerateImageRequest):
    """Submits an image-to-3D volumetric reconstruction request."""
    params = req.model_dump()
    params["input_mode"] = "image"

    valid, err, clean_params = generation_orchestrator.validate_request("image-to-3d", params)
    if not valid:
        raise HTTPException(status_code=400, detail=err)

    try:
        generation_orchestrator.resolve_route("image-to-3d", clean_params)
    except RouteUnavailableError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = job_manager.create_job(clean_params)
    return job.to_dict()

@router.get("/api/v1/jobs/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(job_id: str):
    """Retrieves real-time execution progress, stage, and recent logs for a job."""
    validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress_pct": job.progress_pct,
        "elapsed_seconds": job.elapsed_seconds if job.elapsed_seconds else round(time.time() - job.start_time, 2),
        "error": job.error,
        "logs": job.logs[-10:] if job.logs else []
    }

@router.get("/api/v1/assets")
def list_generated_assets(limit: int = 50):
    """Returns catalog of registered generated 3D assets across all jobs."""
    return {
        "assets": generation_orchestrator.list_assets(limit=limit),
        "total": len(generation_orchestrator.list_assets(limit=1000))
    }

@router.get("/api/v1/assets/{asset_id}")
def get_asset_record(asset_id: str):
    """Returns detailed metadata for a specific generated 3D asset."""
    record = generation_orchestrator.get_asset(asset_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found.")
    return record


# ==============================================================================
# Phase 5 Blender Studio Integration Endpoints
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = PROJECT_ROOT / "output" / "staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

_latest_blender_scene_state = {
    "status": "idle",
    "timestamp": None,
    "scene_id": "default_scene",
    "active_object": None,
    "selected_objects": [],
    "scene_objects_summary": [],
    "camera_info": None
}

@router.post("/api/v1/stage-image", response_model=StageImageResponse)
async def stage_image(file: UploadFile = File(...)):
    """Safely accepts, validates, and stages an uploaded 2D concept or reference image from Blender."""
    filename = file.filename or "uploaded_image.png"
    ext = Path(filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported image extension '{ext}'. Expected .png, .jpg, .jpeg, or .webp.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty (0 bytes).")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded image exceeds 25 MB size limit.")

    safe_name = f"{uuid.uuid4().hex[:12]}_{Path(filename).name}"
    staged_path = STAGING_DIR / safe_name
    with open(staged_path, "wb") as f:
        f.write(content)

    try:
        from PIL import Image
        with Image.open(staged_path) as img:
            w, h = img.size
            img_format = img.format or ext.replace(".", "").upper()
    except Exception as img_err:
        if staged_path.exists():
            staged_path.unlink()
        raise HTTPException(status_code=400, detail=f"Corrupt or invalid image file: {img_err}")

    return {
        "staged_path": str(staged_path),
        "filename": safe_name,
        "width": w,
        "height": h,
        "format": img_format,
        "file_size_bytes": len(content)
    }

@router.post("/api/v1/blender/sync-scene", response_model=SceneSyncResponse)
def sync_blender_scene(req: SceneSyncRequest):
    """Receives live scene state, active object, and selected meshes from Blender."""
    global _latest_blender_scene_state
    name = req.scene_name or req.scene_id or "default_scene"
    objs = req.objects or req.selected_objects or req.scene_objects_summary or []
    count = len(objs)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cur_ver = (_latest_blender_scene_state.get("scene_version", 0) if isinstance(_latest_blender_scene_state, dict) else 0) + 1
    _latest_blender_scene_state = {
        "status": "synchronized",
        "timestamp": now_str,
        "last_sync": now_str,
        "scene_name": name,
        "scene_id": name,
        "scene_version": cur_ver,
        "object_count": count,
        "objects": objs,
        "units": req.units or "METRIC",
        "blender_version": req.blender_version or "4.2.0"
    }
    return {
        "status": "synchronized",
        "timestamp": now_str,
        "scene_name": name,
        "synced_objects": count,
        "objects_synced_count": count,
        "scene_version": cur_ver,
        "message": f"Successfully synchronized {count} scene object(s) with Local Mesh AI backend."
    }

@router.get("/api/v1/blender/scene-status")
def get_blender_scene_status():
    """Returns the most recent synchronized scene metadata from Blender."""
    return _latest_blender_scene_state


# ==============================================================================
# Phase 6 Advanced AI Technical Artist Endpoints
# ==============================================================================

@router.post("/api/v1/blender/stage-asset", response_model=StageAssetResponse)
def stage_blender_asset(req: StageAssetRequest):
    """Stages an active mesh object from Blender into an isolated editing session."""
    import base64
    temp_mesh_file = None

    if req.mesh_base64:
        safe_name = "".join(c for c in req.asset_name if c.isalnum() or c in ("_", "-")) or "staged_mesh"
        staging_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "jobs" / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        temp_mesh_file = str(staging_dir / f"{safe_name}_{uuid.uuid4().hex[:6]}.{req.file_format}")
        with open(temp_mesh_file, "wb") as f:
            f.write(base64.b64decode(req.mesh_base64))
        mesh_to_load = temp_mesh_file
    elif req.mesh_path:
        mesh_to_load = os.path.abspath(req.mesh_path)
        if not os.path.exists(mesh_to_load):
            raise HTTPException(status_code=400, detail=f"Staged mesh file does not exist: {req.mesh_path}")
    else:
        raise HTTPException(status_code=400, detail="Either mesh_path or mesh_base64 must be provided.")

    try:
        session = editing_session_manager.create_session(
            asset_name=req.asset_name,
            source_mesh_path=mesh_to_load,
            object_metadata=req.object_metadata,
            fingerprint=req.fingerprint
        )
        rep = geometry_analyzer.analyze_mesh(session.working_mesh_path)
    finally:
        if temp_mesh_file and os.path.exists(temp_mesh_file):
            try:
                os.remove(temp_mesh_file)
            except Exception:
                pass

    return {
        "session_id": session.session_id,
        "asset_id": session.asset_id,
        "asset_name": session.asset_name,
        "mesh_path": session.working_mesh_path,
        "mesh_url": f"/api/v1/artist/session/{session.session_id}/mesh",
        "vertex_count": rep.counts.vertex_count,
        "face_count": rep.counts.face_count,
        "dimensions": [float(d) for d in rep.bounds.dimensions],
        "created_at": session.created_at,
        "message": f"Asset '{session.asset_name}' staged successfully ({rep.counts.face_count} faces)."
    }


@router.post("/api/v1/artist/analyze", response_model=AnalyzeAssetResponse)
def analyze_asset_components(req: AnalyzeAssetRequest):
    """Executes component decomposition, multi-view rendering, and visual understanding for session."""
    try:
        session = editing_session_manager.analyze_session(
            session_id=req.session_id,
            decomposition_method=req.decomposition_method,
            num_clusters=req.num_clusters,
            include_vlm=req.include_vlm,
            user_description=req.user_description
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Editing session '{req.session_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Asset analysis failed: {str(e)}")

    components = []
    if session.hierarchy:
        for cid, node in session.hierarchy.components.items():
            if "root" in cid.lower() or "assembly" in cid.lower():
                continue
            components.append(ComponentSummary(
                component_id=cid,
                semantic_label=node.semantic_label,
                confidence=round(node.confidence, 2),
                is_uncertain=node.is_uncertain,
                dimensions=[float(d) for d in node.dimensions],
                centroid=[float(c) for c in node.centroid],
                face_count=node.face_count,
                vertex_count=node.vertex_count,
                relative_z_position=round(node.relative_z_position, 2),
                color_hex=session.color_legend.get(cid),
                submesh_url=f"/api/v1/artist/session/{session.session_id}/submesh/{cid}" if node.submesh_path else None
            ))

    render_urls = {
        view: f"/api/v1/artist/session/{session.session_id}/renders/{view}"
        for view in session.render_paths.keys()
    }
    mask_url = f"/api/v1/artist/session/{session.session_id}/renders/mask" if session.mask_render_path else None

    return {
        "session_id": session.session_id,
        "asset_id": session.asset_id,
        "asset_name": session.asset_name,
        "total_components": len(components),
        "components": components,
        "decomposition_provenance": session.decomposition_provenance,
        "vlm_provenance": session.vlm_provenance,
        "render_paths": session.render_paths,
        "render_urls": render_urls,
        "mask_render_url": mask_url,
        "color_legend": session.color_legend,
        "summary": session.summary
    }


@router.get("/api/v1/artist/components/{session_id}")
def get_session_components(session_id: str):
    """Retrieves component hierarchy and color legend for session."""
    session = editing_session_manager.get_session(session_id)
    if not session or not session.hierarchy:
        raise HTTPException(status_code=404, detail=f"Hierarchy not available for session '{session_id}'. Run analyze first.")

    comps = [
        {
            "component_id": cid,
            "semantic_label": node.semantic_label,
            "confidence": round(node.confidence, 2),
            "is_uncertain": node.is_uncertain,
            "dimensions": [float(d) for d in node.dimensions],
            "centroid": [float(c) for c in node.centroid],
            "face_count": node.face_count,
            "vertex_count": node.vertex_count,
            "relative_z_position": round(node.relative_z_position, 2),
            "color_hex": session.color_legend.get(cid)
        }
        for cid, node in session.hierarchy.components.items()
        if "root" not in cid.lower() and "assembly" not in cid.lower()
    ]
    return {
        "session_id": session_id,
        "total_components": len(comps),
        "components": comps,
        "color_legend": session.color_legend,
        "decomposition_provenance": session.decomposition_provenance,
        "vlm_provenance": session.vlm_provenance
    }


@router.post("/api/v1/artist/plan", response_model=GeneratePlanResponse)
def generate_artist_plan(req: GeneratePlanRequest):
    """Produces structured ComponentEditPlan from natural-language instructions."""
    try:
        plan = editing_session_manager.generate_plan(
            session_id=req.session_id,
            user_instruction=req.user_instruction,
            target_component_id=req.target_component_id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Editing session '{req.session_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")

    return {
        "session_id": req.session_id,
        "plan_id": plan.plan_id,
        "plan": plan.model_dump(),
        "status": plan.status,
        "requires_user_confirmation": plan.requires_user_confirmation,
        "rationale": plan.rationale
    }


@router.post("/api/v1/artist/execute", response_model=ExecutePlanResponse)
def execute_artist_plan(req: ExecutePlanRequest):
    """Executes the approved plan deterministically, verifies invariants, and returns report."""
    try:
        report = editing_session_manager.execute_plan(
            session_id=req.session_id,
            plan_id=req.plan_id,
            confirmed_target_id=req.confirmed_target_id,
            plan_override=req.plan_override
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Editing session '{req.session_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan execution failed: {str(e)}")

    return {
        "session_id": req.session_id,
        "status": report.status,
        "report": report.model_dump(),
        "output_mesh_url": f"/api/v1/artist/session/{req.session_id}/mesh",
        "message": report.summary
    }


@router.post("/api/v1/artist/rollback", response_model=RollbackResponse)
def rollback_artist_edit(req: RollbackRequest):
    """Restores the working mesh to the pre-edit snapshot."""
    success, msg = editing_session_manager.rollback_session(req.session_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {
        "session_id": req.session_id,
        "status": "ROLLED_BACK",
        "message": msg,
        "output_mesh_url": f"/api/v1/artist/session/{req.session_id}/mesh"
    }


@router.get("/api/v1/artist/session/{session_id}", response_model=SessionStateResponse)
def get_artist_session(session_id: str):
    """Returns complete state, components, and history for an editing session."""
    session = editing_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Editing session '{session_id}' not found.")

    comps = []
    if session.hierarchy:
        for cid, node in session.hierarchy.components.items():
            if "root" in cid.lower() or "assembly" in cid.lower():
                continue
            comps.append(ComponentSummary(
                component_id=cid,
                semantic_label=node.semantic_label,
                confidence=round(node.confidence, 2),
                is_uncertain=node.is_uncertain,
                dimensions=[float(d) for d in node.dimensions],
                centroid=[float(c) for c in node.centroid],
                face_count=node.face_count,
                vertex_count=node.vertex_count,
                relative_z_position=round(node.relative_z_position, 2),
                color_hex=session.color_legend.get(cid),
                submesh_url=f"/api/v1/artist/session/{session.session_id}/submesh/{cid}" if node.submesh_path else None
            ))

    return {
        "session_id": session.session_id,
        "asset_id": session.asset_id,
        "asset_name": session.asset_name,
        "version": session.version,
        "status": session.status,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "total_components": len(comps),
        "components": comps,
        "history_count": len(session.history),
        "has_pre_edit_snapshot": bool(session.pre_edit_snapshot_path and os.path.exists(session.pre_edit_snapshot_path)),
        "mesh_url": f"/api/v1/artist/session/{session.session_id}/mesh",
        "history": session.history
    }


@router.get("/api/v1/artist/session/{session_id}/mesh")
def get_session_mesh(session_id: str):
    """Serves the working GLB/OBJ mesh asset for the session."""
    session = editing_session_manager.get_session(session_id)
    if not session or not os.path.exists(session.working_mesh_path):
        raise HTTPException(status_code=404, detail=f"Mesh asset not found for session '{session_id}'.")

    media_type = "model/gltf-binary"
    if session.working_mesh_path.lower().endswith(".obj"):
        media_type = "model/obj"

    return FileResponse(
        path=session.working_mesh_path,
        media_type=media_type,
        filename=f"{session.asset_name}_v{session.version}{Path(session.working_mesh_path).suffix}"
    )


@router.get("/api/v1/artist/session/{session_id}/renders/{view_name}")
def get_session_render(session_id: str, view_name: str):
    """Serves rendered multi-view image or color component mask."""
    session = editing_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if view_name.lower() in ("mask", "masks", "component_masks"):
        if not session.mask_render_path or not os.path.exists(session.mask_render_path):
            raise HTTPException(status_code=404, detail="Mask render not available.")
        return FileResponse(path=session.mask_render_path, media_type="image/png", filename="mask.png")

    p = session.render_paths.get(view_name.lower())
    if not p or not os.path.exists(p):
        raise HTTPException(status_code=404, detail=f"Render view '{view_name}' not available.")
    return FileResponse(path=p, media_type="image/png", filename=f"{view_name}.png")
