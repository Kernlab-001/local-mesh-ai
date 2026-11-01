"""
Asynchronous Job Orchestrator & Production State Machine for Local Mesh AI.
Supports dual-mode entry points that converge on a shared 3D production pipeline:
- Mode A (Text-to-3D): Validates text prompt, synthesizes concept image via SD-Turbo, and forwards to 3D pipeline.
- Mode B (Image-to-3D): Validates uploaded photograph, stages it safely, and bypasses image generation completely.
- Shared Pipeline: Perspective alignment, volumetric reconstruction, color calibration,
  retopology decimation & quad conversion, high-to-low ray-cast texture baking, and machine-readable manifest generation.
"""

import os
import time
import json
import uuid
import threading
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

from core.hardware_detector import hardware_detector
from core.triposr_engine import triposr_engine
from core.gpu_broker import gpu_broker
from core.prompt_to_image import prompt_engine
from core.color_calibrator import color_calibrator
from core.multiview_synthesizer import multiview_synthesizer
from core.model_lifecycle import model_lifecycle, ModelSlot
from core.input_validator import input_validator, ValidationError
from core.engines.registry import engine_registry
from core.geometry_validator import geometry_validator
from core.generation_orchestrator import generation_orchestrator, PipelineStage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

class JobStatus:
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    WAITING_FOR_GPU = "WAITING_FOR_GPU"
    LOADING_MODEL = "LOADING_MODEL"
    GENERATING_CONCEPT = "GENERATING_CONCEPT"
    RUNNING_RECONSTRUCTION = "RUNNING_RECONSTRUCTION"
    PROCESSING_MESH = "PROCESSING_MESH"
    SAVING_ASSET = "SAVING_ASSET"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    # Backward compatibility aliases
    PREPROCESSING = "VALIDATING"
    RECONSTRUCTING = "RUNNING_RECONSTRUCTION"
    POSTPROCESSING = "PROCESSING_MESH"

class Job:
    def __init__(self, job_id: str, params: Dict[str, Any]):
        self.job_id = job_id
        self.params = params
        self._lock = threading.Lock()

        # Explicitly determine input mode
        explicit_mode = params.get("input_mode")
        if explicit_mode:
            self.input_mode = explicit_mode.lower()
        elif params.get("prompt") and not params.get("image_path"):
            self.input_mode = "prompt"
        else:
            self.input_mode = "image"

        self.status = JobStatus.QUEUED
        self.stage = "Queued"
        self.progress_pct = 0
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.elapsed_seconds = 0.0
        self.error: Optional[str] = None
        self.logs: List[str] = []

        # Asset output paths (only populated when verified to exist on disk)
        self.output_mesh: Optional[str] = None
        self.high_poly_mesh: Optional[str] = None
        self.low_poly_mesh: Optional[str] = None
        self.output_blend: Optional[str] = None
        self.preview_png: Optional[str] = None
        self.turntable_mp4: Optional[str] = None
        self.baked_normal: Optional[str] = None
        self.baked_diffuse: Optional[str] = None
        self.game_ready_glb: Optional[str] = None
        self.manifest: Optional[Dict[str, Any]] = None
        self.appearance_profile: Optional[Dict[str, Any]] = None

        self._cancel_requested = False

    def log(self, message: str):
        t = time.strftime("%H:%M:%S")
        entry = f"[{t}] {message}"
        with self._lock:
            self.logs.append(entry)
        self.save()

    def set_stage(self, stage: str, pct: int):
        with self._lock:
            self.stage = stage
            self.progress_pct = pct
        self.log(f"Stage: {stage} ({pct}%)")

    def cancel(self):
        with self._lock:
            self._cancel_requested = True
            if self.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                self.status = JobStatus.CANCELLED
                self.stage = "Cancelled by User"
        self.log("Cancellation requested by user.")
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = round(time.time() - self.start_time if not self.end_time else self.end_time - self.start_time, 2)
            return {
                "job_id": self.job_id,
                "input_mode": self.input_mode,
                "status": self.status,
                "stage": self.stage,
                "progress_pct": self.progress_pct,
                "elapsed_seconds": elapsed,
                "error": self.error,
                "output_mesh": self.output_mesh,
                "high_poly_mesh": self.high_poly_mesh,
                "low_poly_mesh": self.low_poly_mesh,
                "output_blend": self.output_blend,
                "preview_png": self.preview_png,
                "turntable_mp4": self.turntable_mp4,
                "baked_normal": self.baked_normal,
                "baked_diffuse": self.baked_diffuse,
                "game_ready_glb": self.game_ready_glb,
                "mesh_url": f"/jobs/{self.job_id}/mesh" if self.output_mesh and os.path.exists(self.output_mesh) and os.path.getsize(self.output_mesh) > 0 else None,
                "preview_url": f"/jobs/{self.job_id}/preview" if self.preview_png and os.path.exists(self.preview_png) and os.path.getsize(self.preview_png) > 0 else None,
                "blend_url": f"/jobs/{self.job_id}/blend" if self.output_blend and os.path.exists(self.output_blend) and os.path.getsize(self.output_blend) > 0 else None,
                "concept_url": f"/jobs/{self.job_id}/concept" if os.path.exists(os.path.join(JOBS_DIR, self.job_id, "input", "concept.png")) or os.path.exists(os.path.join(JOBS_DIR, self.job_id, "input", "source_image.png")) else None,
                "multiview_url": f"/jobs/{self.job_id}/multiview" if os.path.exists(os.path.join(JOBS_DIR, self.job_id, "multiview", "multiview_grid.png")) or os.path.exists(os.path.join(JOBS_DIR, self.job_id, "multiview", "composite_grid.png")) else None,
                "appearance_profile": self.appearance_profile,
                "manifest": self.manifest,
                "logs": self.logs[-25:],
                "params": self.params
            }

    def save(self):
        job_file = os.path.join(JOBS_DIR, f"{self.job_id}.json")
        try:
            temp_file = f"{job_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(temp_file, job_file)
        except Exception:
            pass

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)  # Strictly serialized GPU workers
        self._lock = threading.Lock()

    def create_job(self, params: Dict[str, Any]) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id, params)
        with self._lock:
            self.jobs[job_id] = job
        job.save()
        self.executor.submit(self._run_job, job)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            if job_id in self.jobs:
                return self.jobs[job_id]
        # Check disk cache
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            try:
                with open(job_file, "r") as f:
                    data = json.load(f)
                job = Job(job_id, data.get("params", {}))
                job.input_mode = data.get("input_mode", "image")
                job.status = data.get("status")
                job.stage = data.get("stage")
                job.progress_pct = data.get("progress_pct", 0)
                job.output_mesh = data.get("output_mesh")
                job.high_poly_mesh = data.get("high_poly_mesh")
                job.low_poly_mesh = data.get("low_poly_mesh")
                job.output_blend = data.get("output_blend")
                job.preview_png = data.get("preview_png")
                job.turntable_mp4 = data.get("turntable_mp4")
                job.baked_normal = data.get("baked_normal")
                job.baked_diffuse = data.get("baked_diffuse")
                job.game_ready_glb = data.get("game_ready_glb")
                job.manifest = data.get("manifest")
                job.logs = data.get("logs", [])
                job.error = data.get("error")
                return job
            except Exception:
                pass
        return None

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job and job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.cancel()
            return True
        return False

    def _run_job(self, job: Job):
        try:
            generation_orchestrator.execute_pipeline(job)
        except Exception as e:
            # Failure states and logs are captured and persisted inside execute_pipeline
            pass

job_manager = JobManager()
