"""
FastAPI Routes for Local Neural 3D Studio.
Hardened with input validation, path safety, and secure error boundaries.
"""

import os
import shutil
import subprocess
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from api.schemas import (
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
    ExecuteToolRequest
)
from services.job_manager import job_manager
from services.export_service import export_service
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
