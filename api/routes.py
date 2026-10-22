"""
FastAPI Routes for Local Neural 3D Studio.
"""

import os
from fastapi import APIRouter, HTTPException
from api.schemas import CreateJobRequest, JobResponse, EditJobRequest, ExportJobRequest
from services.job_manager import job_manager
from services.export_service import export_service
from core.gpu_broker import gpu_broker
from core.qwen_planner import qwen_planner
import subprocess

router = APIRouter()

from core.hardware_detector import hardware_detector

@router.get("/health")
def health_check():
    gpu_status = gpu_broker.get_status()
    blender_path = hardware_detector.find_blender() or "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
    blender_installed = os.path.exists(blender_path)

    qwen_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen2.5-coder-7b-instruct-q4_k_m.gguf")

    return {
        "status": "healthy",
        "gpu": gpu_status,
        "blender": {
            "installed": blender_installed,
            "version": "5.2.2 LTS",
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
    if req.image_path and not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image file does not exist: {req.image_path}")

    job = job_manager.create_job(req.model_dump())
    return job.to_dict()

@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job '{job_id}'. Either not found or already finished.")
    return {"status": "cancelled", "job_id": job_id}

@router.post("/jobs/{job_id}/edit")
def edit_job_with_qwen(job_id: str, req: EditJobRequest):
    job = job_manager.get_job(job_id)
    if not job or not job.output_blend or not os.path.exists(job.output_blend):
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

    executor_script = "E:/GPUPLAT/core/blender_executor.py"
    blender_exe = "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
    cmd = [
        blender_exe, "--background",
        "--python", executor_script,
        "--", temp_plan, blend_in, blend_out
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Blender executor failed:\n{proc.stderr}")

    return {
        "status": "success",
        "plan": plan.model_dump(),
        "edited_blend": blend_out
    }

@router.post("/jobs/{job_id}/export")
def export_job_asset(job_id: str, req: ExportJobRequest):
    job = job_manager.get_job(job_id)
    if not job or not job.output_blend or not os.path.exists(job.output_blend):
        raise HTTPException(status_code=400, detail="Job blend project not available for export.")

    try:
        res = export_service.export_asset(
            blend_path=job.output_blend,
            format_type=req.format,
            output_name=f"{req.output_name}_{job_id}"
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
