"""
FastAPI Routes for Local Neural 3D Studio.
Hardened with input validation, path safety, and secure error boundaries.
"""

import os
import shutil
import subprocess
from fastapi import APIRouter, HTTPException
from api.schemas import CreateJobRequest, JobResponse, EditJobRequest, ExportJobRequest
from services.job_manager import job_manager
from services.export_service import export_service
from core.gpu_broker import gpu_broker
from core.qwen_planner import qwen_planner
from core.hardware_detector import hardware_detector

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
