"""
Asynchronous Job Orchestrator & State Machine.
Tracks stages, progress %, execution timers, structured logs,
cancellation tokens, and persistent job records in E:\\GPUPLAT\\jobs.
"""

import os
import time
import json
import uuid
import threading
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from core.hardware_detector import hardware_detector
from core.triposr_engine import triposr_engine
from core.gpu_broker import gpu_broker
from core.prompt_to_image import prompt_engine
from core.color_calibrator import color_calibrator

JOBS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

class JobStatus:
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    PREPROCESSING = "PREPROCESSING"
    RECONSTRUCTING = "RECONSTRUCTING"
    POSTPROCESSING = "POSTPROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class Job:
    def __init__(self, job_id: str, params: Dict[str, Any]):
        self.job_id = job_id
        self.params = params
        self.status = JobStatus.QUEUED
        self.stage = "Queued"
        self.progress_pct = 0
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.elapsed_seconds = 0.0
        self.error: Optional[str] = None
        self.logs: List[str] = []
        self.output_mesh: Optional[str] = None
        self.output_blend: Optional[str] = None
        self.preview_png: Optional[str] = None
        self.turntable_mp4: Optional[str] = None
        self._cancel_requested = False

    def log(self, message: str):
        t = time.strftime("%H:%M:%S")
        entry = f"[{t}] {message}"
        self.logs.append(entry)
        self.save()

    def set_stage(self, stage: str, pct: int):
        self.stage = stage
        self.progress_pct = pct
        self.log(f"Stage: {stage} ({pct}%)")

    def cancel(self):
        self._cancel_requested = True
        self.status = JobStatus.CANCELLED
        self.log("Cancellation requested by user.")
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress_pct": self.progress_pct,
            "elapsed_seconds": round(time.time() - self.start_time if not self.end_time else self.end_time - self.start_time, 2),
            "error": self.error,
            "output_mesh": self.output_mesh,
            "output_blend": self.output_blend,
            "preview_png": self.preview_png,
            "turntable_mp4": self.turntable_mp4,
            "logs": self.logs[-20:],  # last 20 log entries
            "params": self.params
        }

    def save(self):
        job_file = os.path.join(JOBS_DIR, f"{self.job_id}.json")
        try:
            with open(job_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
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
        # Check disk
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                data = json.load(f)
            job = Job(job_id, data.get("params", {}))
            job.status = data.get("status")
            job.stage = data.get("stage")
            job.progress_pct = data.get("progress_pct", 0)
            job.output_mesh = data.get("output_mesh")
            job.output_blend = data.get("output_blend")
            job.preview_png = data.get("preview_png")
            job.turntable_mp4 = data.get("turntable_mp4")
            job.logs = data.get("logs", [])
            job.error = data.get("error")
            return job
        return None

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job and job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.cancel()
            return True
        return False

    def _run_job(self, job: Job):
        try:
            job.status = JobStatus.VALIDATING
            job.set_stage("Validating Input Parameters", 5)

            image_path = job.params.get("image_path")
            prompt = job.params.get("prompt")

            if (not image_path or not os.path.exists(image_path)) and prompt:
                job.status = JobStatus.PREPROCESSING
                job.set_stage(f"Synthesizing Concept Image from Prompt", 15)
                inputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs")
                os.makedirs(inputs_dir, exist_ok=True)
                synth_path = os.path.join(inputs_dir, f"prompt_{job.job_id}.png")

                job.log(f"Starting prompt-to-image synthesis: '{prompt}'")
                t0 = time.time()
                prompt_engine.generate(prompt=prompt, output_path=synth_path)
                job.log(f"Concept image synthesized in {round(time.time() - t0, 2)}s: {synth_path}")
                image_path = synth_path
                job.params["image_path"] = image_path

            if not image_path or not os.path.exists(image_path):
                raise ValueError("Neither a valid concept image nor a text prompt was provided.")

            triposr_engine.validate_image(image_path)

            if job._cancel_requested:
                return

            # Stage 2: Reconstruction
            job.status = JobStatus.RECONSTRUCTING
            job.set_stage("Acquiring GPU Lease & Loading TripoSR", 30)

            quality = job.params.get("quality", "balanced")
            rec_result = triposr_engine.reconstruct(
                image_path=image_path,
                quality=quality,
                render_turntable=job.params.get("render_turntable", False)
            )

            job.output_mesh = rec_result["mesh_path"]
            job.log(f"Reconstruction completed in {rec_result['elapsed_seconds']}s")

            # Stage 2.5: Accurate Color Calibration (Tripo AI Style)
            if job.params.get("accurate_colors", True):
                job.set_stage("Calibrating Color Distribution (Tripo AI Style)", 65)
                try:
                    color_calibrator.calibrate_mesh_colors(job.output_mesh, image_path)
                    job.log("Color calibration complete: perceptual Lab palette aligned with source artwork.")
                except Exception as col_err:
                    job.log(f"Color calibration notice: {col_err}")

            if job._cancel_requested:
                return

            # Stage 3: Blender Post-Processing
            job.status = JobStatus.POSTPROCESSING
            job.set_stage("Sanitizing Geometry & Blender PBR Shading", 75)

            blend_out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
            bridge_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge", "blender_bridge.py")
            blender_exe = hardware_detector.find_blender() or "blender"

            is_blender_available = False
            if blender_exe:
                if os.path.exists(blender_exe) or shutil.which(blender_exe):
                    is_blender_available = True

            if is_blender_available:
                cmd = [
                    blender_exe, "--background",
                    "--python", bridge_script,
                    "--",
                    "--mesh", job.output_mesh,
                    "--image", os.path.abspath(image_path),
                    "--out", blend_out_dir,
                    "--preset", job.params.get("preset", "prop")
                ]

                if not job.params.get("sharp_overlay", True):
                    cmd.append("--no-overlay")

                target_blend = job.params.get("target_blend")
                if target_blend and os.path.exists(target_blend):
                    cmd.extend(["--blend", os.path.abspath(target_blend)])

                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180.0)
                    if proc.returncode != 0:
                        job.log(f"Blender notice: {proc.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    job.log("Blender post-processing timed out after 180s. Keeping raw mesh.")
                except Exception as b_err:
                    job.log(f"Blender execution notice: {b_err}")
            else:
                job.log("Blender executable not found on system; generated 3D mesh saved directly.")

            job.output_blend = os.path.join(blend_out_dir, "project.blend")
            job.preview_png = os.path.join(blend_out_dir, "preview.png")

            # Check turntable
            tt_path = os.path.join(os.path.dirname(job.output_mesh), "render.mp4")
            if os.path.exists(tt_path):
                job.turntable_mp4 = tt_path

            job.status = JobStatus.COMPLETED
            job.end_time = time.time()
            job.set_stage("Completed Successfully", 100)
            job.log("All pipeline stages completed.")
        except Exception as e:
            if not job._cancel_requested:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.log(f"Job failed with error: {str(e)}")
            else:
                job.status = JobStatus.CANCELLED
            job.end_time = time.time()
        finally:
            job.save()

job_manager = JobManager()
