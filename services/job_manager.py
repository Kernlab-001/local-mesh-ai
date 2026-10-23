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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
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
        # Setup per-job structured workspace
        job_dir = os.path.join(JOBS_DIR, job.job_id)
        input_dir = os.path.join(job_dir, "input")
        high_poly_dir = os.path.join(job_dir, "high_poly_reference")
        low_poly_dir = os.path.join(job_dir, "low_poly_asset")
        textures_dir = os.path.join(job_dir, "textures")
        reports_dir = os.path.join(job_dir, "reports")
        multiview_dir = os.path.join(job_dir, "multiview")

        for d in (job_dir, input_dir, high_poly_dir, low_poly_dir, textures_dir, reports_dir, multiview_dir):
            os.makedirs(d, exist_ok=True)

        timings = {}
        t_start_total = time.time()

        try:
            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                job.end_time = time.time()
                return

            job.status = JobStatus.VALIDATING
            job.set_stage("Validating Input Parameters", 5)

            staged_image_path = None
            input_metadata = {}

            # =========================================================================
            # DUAL ENTRY PATH HANDLING
            # =========================================================================
            if job.input_mode == "prompt":
                # ---------------------------------------------------------------------
                # MODE A: TEXT-TO-3D
                # ---------------------------------------------------------------------
                job.status = JobStatus.PREPROCESSING
                prompt = job.params.get("prompt")
                clean_prompt = input_validator.validate_prompt(prompt)
                job.log(f"[Mode A: Text-to-3D] Validated text prompt: '{clean_prompt}'")

                job.set_stage("Synthesizing Concept Image from Prompt (SD-Turbo)", 15)
                synth_target = os.path.join(input_dir, "concept.png")

                t_prompt_start = time.time()
                prompt_engine.generate(prompt=clean_prompt, output_path=synth_target)
                timings["concept_synthesis"] = round(time.time() - t_prompt_start, 2)

                # Validate generated concept output
                input_metadata = input_validator.validate_generated_concept(synth_target)
                staged_image_path = synth_target
                job.log(f"Concept image synthesized in {timings['concept_synthesis']}s: {synth_target}")

            else:
                # ---------------------------------------------------------------------
                # MODE B: IMAGE-TO-3D
                # ---------------------------------------------------------------------
                job.set_stage("Validating Uploaded Reference Image", 10)
                source_image = job.params.get("image_path")
                if not source_image:
                    raise ValidationError("Field 'image_path' is required for Image-to-3D mode.")

                input_metadata = input_validator.validate_uploaded_image(
                    source_path=source_image,
                    destination_dir=input_dir
                )
                staged_image_path = input_metadata["staged_path"]
                job.log(f"[Mode B: Image-to-3D] Uploaded image validated & staged: {input_metadata['width']}x{input_metadata['height']} ({input_metadata['format']})")

            if not staged_image_path or not os.path.exists(staged_image_path):
                raise ValidationError("Failed to prepare a valid input image for 3D reconstruction.")

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                job.end_time = time.time()
                return

            # =========================================================================
            # SHARED DOWNSTREAM 3D PRODUCTION PIPELINE
            # =========================================================================

            # Stage 1.5: Multi-Angle Reference Perspectives
            t_mv_start = time.time()
            try:
                job.set_stage("Generating 360 Multi-View Reference Grid", 25)
                mv_res = multiview_synthesizer.generate_canonical_views(staged_image_path, multiview_dir)
                timings["multiview_reference"] = round(time.time() - t_mv_start, 2)
                job.log(f"Multi-view reference generation complete in {timings['multiview_reference']}s (Front, Right, Back, Left views).")
            except Exception as mv_err:
                job.log(f"Multi-view reference notice: {mv_err}. Proceeding with primary input.")

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                job.end_time = time.time()
                return

            # Stage 2: Volumetric 3D Reconstruction
            job.status = JobStatus.RECONSTRUCTING
            job.set_stage("Reconstructing Volumetric Geometry (TripoSR)", 45)

            t_rec_start = time.time()
            quality = job.params.get("quality", "balanced")
            rec_result = triposr_engine.reconstruct(
                image_path=staged_image_path,
                quality=quality,
                render_turntable=job.params.get("render_turntable", False)
            )
            timings["reconstruction"] = rec_result.get("elapsed_seconds", round(time.time() - t_rec_start, 2))

            # Stage raw high-poly reference mesh
            raw_mesh = rec_result["mesh_path"]
            high_poly_target = os.path.join(high_poly_dir, "mesh.glb")
            if os.path.exists(raw_mesh) and os.path.getsize(raw_mesh) > 0:
                shutil.copy2(raw_mesh, high_poly_target)
                job.high_poly_mesh = high_poly_target
                job.output_mesh = high_poly_target
            else:
                raise RuntimeError("Reconstruction completed but output mesh file is missing or empty.")

            job.log(f"Volumetric reconstruction completed in {timings['reconstruction']}s on device: {rec_result.get('device', 'auto')}.")

            # Stage 2.5: Accurate Color Calibration (Reinhard Lab Space Transfer)
            if job.params.get("accurate_colors", True):
                job.set_stage("Calibrating Color Distribution (Reinhard Lab)", 65)
                try:
                    color_calibrator.calibrate_mesh_colors(job.output_mesh, staged_image_path)
                    job.log("Color calibration complete: perceptual Lab palette aligned with source artwork.")
                except Exception as col_err:
                    job.log(f"Color calibration notice: {col_err}")

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                job.end_time = time.time()
                return

            # Stage 3: Blender Post-Processing, Sanitation & Optional Retopology/Baking
            job.status = JobStatus.POSTPROCESSING
            job.set_stage("Blender Post-Processing & Geometry Sanitation", 75)

            t_blender_start = time.time()
            bridge_script = os.path.join(PROJECT_ROOT, "bridge", "blender_bridge.py")
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
                    "--image", os.path.abspath(staged_image_path),
                    "--out", job_dir,
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
                        job.log(f"Blender bridge notice: {proc.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    job.log("Blender post-processing timed out after 180s. Keeping raw mesh.")
                except Exception as b_err:
                    job.log(f"Blender execution notice: {b_err}")

                # Optional Game-Asset Retopology & High-to-Low Texture Baking
                if job.params.get("retopology_and_bake", False):
                    job.set_stage("Quad-Dominant Retopology & Texture Baking", 85)
                    baker_script = os.path.join(PROJECT_ROOT, "bridge", "retopology_and_baker.py")
                    target_f = job.params.get("target_faces", 8000)
                    bake_cmd = [
                        blender_exe, "--background",
                        "--python", baker_script,
                        "--",
                        "--mesh", job.output_mesh,
                        "--out", job_dir,
                        "--faces", str(target_f),
                        "--res", "2048"
                    ]
                    try:
                        proc_b = subprocess.run(bake_cmd, capture_output=True, text=True, timeout=240.0)
                        if proc_b.returncode == 0:
                            norm_p = os.path.join(job_dir, "baked_normal.png")
                            diff_p = os.path.join(job_dir, "baked_diffuse.png")
                            gr_glb = os.path.join(job_dir, "model_game_ready.glb")

                            if os.path.exists(norm_p) and os.path.getsize(norm_p) > 0:
                                norm_dest = os.path.join(textures_dir, "baked_normal.png")
                                shutil.copy2(norm_p, norm_dest)
                                job.baked_normal = norm_dest
                            if os.path.exists(diff_p) and os.path.getsize(diff_p) > 0:
                                diff_dest = os.path.join(textures_dir, "baked_diffuse.png")
                                shutil.copy2(diff_p, diff_dest)
                                job.baked_diffuse = diff_dest
                            if os.path.exists(gr_glb) and os.path.getsize(gr_glb) > 0:
                                gr_dest = os.path.join(low_poly_dir, "model_game_ready.glb")
                                shutil.copy2(gr_glb, gr_dest)
                                job.game_ready_glb = gr_dest
                                job.low_poly_mesh = gr_dest

                            job.log("Quad retopology decimation & normal/diffuse texture baking completed.")
                        else:
                            job.log(f"Baking notice: {proc_b.stderr[:200]}")
                    except subprocess.TimeoutExpired:
                        job.log("Retopology & baking timed out after 240s; raw mesh preserved.")
                    except Exception as bake_err:
                        job.log(f"Retopology notice: {bake_err}")

                timings["blender_processing"] = round(time.time() - t_blender_start, 2)
            else:
                job.log("Blender executable not found on system; generated 3D mesh saved directly.")

            # Only assign output files if they actually exist on disk and have non-zero size
            blend_candidate = os.path.join(job_dir, "project.blend")
            if os.path.exists(blend_candidate) and os.path.getsize(blend_candidate) > 0:
                job.output_blend = blend_candidate
            else:
                job.output_blend = None

            preview_candidate = os.path.join(job_dir, "preview.png")
            if os.path.exists(preview_candidate) and os.path.getsize(preview_candidate) > 0:
                job.preview_png = preview_candidate
            else:
                job.preview_png = None

            tt_path = os.path.join(os.path.dirname(job.output_mesh), "render.mp4")
            if os.path.exists(tt_path) and os.path.getsize(tt_path) > 0:
                job.turntable_mp4 = tt_path

            timings["total_duration"] = round(time.time() - t_start_total, 2)

            # Stage 4: Production Manifest Generation
            manifest_data = {
                "job_id": job.job_id,
                "input_mode": job.input_mode,
                "status": "completed",
                "settings": job.params,
                "source_input": staged_image_path,
                "input_metadata": input_metadata,
                "timings": timings,
                "assets": {
                    "high_poly_reference": job.high_poly_mesh,
                    "low_poly_game_ready": job.game_ready_glb,
                    "baked_normal": job.baked_normal,
                    "baked_diffuse": job.baked_diffuse,
                    "blender_project": job.output_blend,
                    "preview_render": job.preview_png
                }
            }

            manifest_file = os.path.join(reports_dir, "manifest.json")
            try:
                with open(manifest_file, "w") as f:
                    json.dump(manifest_data, f, indent=2)
                job.manifest = manifest_data
            except Exception:
                pass

            # Final cancellation check before setting completed
            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                job.stage = "Cancelled by User"
                job.end_time = time.time()
                job.log("Job was cancelled prior to final completion.")
            else:
                job.status = JobStatus.COMPLETED
                job.end_time = time.time()
                job.set_stage("Completed Successfully", 100)
                job.log(f"All pipeline stages completed successfully in {timings['total_duration']}s.")

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
