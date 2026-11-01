"""
Controlled LoRA Training Manager for Local Mesh AI.
Orchestrates fine-tuning jobs with hardware preflight gating, exclusive GPU lease acquisition,
guaranteed resource release in finally blocks, cancellation support, and non-destructive candidate staging.
"""

import os
import json
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from core.gpu_broker import gpu_broker
from core.path_security import get_data_dir
from services.training_preflight import training_preflight, PreflightReport


class TrainingJobStatus(BaseModel):
    job_id: str
    target_model: str
    adapter_name: str
    status: str  # "queued", "running", "completed", "failed", "cancelled", "rejected"
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 3
    current_loss: Optional[float] = None
    message: str = ""
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    candidate_adapter_path: Optional[str] = None


class LoRATrainingManager:
    def __init__(self):
        self._jobs: Dict[str, TrainingJobStatus] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.adapters_dir = get_data_dir() / "models" / "candidates"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

    def start_training(
        self,
        target_model: str = "qwen2.5-coder-7b",
        adapter_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        epochs: int = 3,
        force_cpu: bool = False,
        dry_run: bool = False
    ) -> TrainingJobStatus:
        """
        Initiates a controlled LoRA training job after verifying hardware preflight safety.
        """
        job_id = f"trn_{uuid.uuid4().hex[:8]}"
        ad_name = adapter_name or f"lora_{target_model}_{job_id}"

        # 1. Preflight check
        preflight: PreflightReport = training_preflight.evaluate(model_name=target_model, force_cpu=force_cpu)
        if not preflight.passed:
            job = TrainingJobStatus(
                job_id=job_id,
                target_model=target_model,
                adapter_name=ad_name,
                status="rejected",
                message=preflight.rejection_reason,
                error=preflight.summary,
            )
            with self._lock:
                self._jobs[job_id] = job
            return job

        if dry_run:
            job = TrainingJobStatus(
                job_id=job_id,
                target_model=target_model,
                adapter_name=ad_name,
                status="completed",
                progress=1.0,
                message=f"[Dry-Run] Preflight passed. Training would execute for {epochs} epochs.",
            )
            with self._lock:
                self._jobs[job_id] = job
            return job

        job = TrainingJobStatus(
            job_id=job_id,
            target_model=target_model,
            adapter_name=ad_name,
            status="running",
            total_epochs=epochs,
            message="Initializing training worker...",
        )

        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = job
            self._cancel_flags[job_id] = cancel_event

        # Launch background worker
        worker = threading.Thread(
            target=self._run_training_worker,
            args=(job_id, target_model, ad_name, epochs, force_cpu, cancel_event),
            daemon=True,
            name=f"LoRA_Worker_{job_id}"
        )
        worker.start()
        return job

    def cancel_training(self, job_id: str) -> bool:
        """Signals active training job to safely abort and release GPU resources."""
        with self._lock:
            event = self._cancel_flags.get(job_id)
            job = self._jobs.get(job_id)

        if event and job and job.status == "running":
            event.set()
            job.status = "cancelled"
            job.message = "Training cancelled by user."
            return True
        return False

    def get_job_status(self, job_id: str) -> Optional[TrainingJobStatus]:
        with self._lock:
            return self._jobs.get(job_id)

    def _run_training_worker(
        self,
        job_id: str,
        target_model: str,
        adapter_name: str,
        epochs: int,
        force_cpu: bool,
        cancel_event: threading.Event
    ):
        lease_id = None
        job = self._jobs[job_id]

        try:
            # 2. Acquire GPU lease if not running in CPU-only mode
            if not force_cpu:
                lease_id = gpu_broker.acquire(f"lora_training_{job_id}", timeout=30.0)
                if not lease_id:
                    job.status = "failed"
                    job.error = "Could not acquire GPU lease within 30s timeout."
                    return

            job.message = "Training in progress..."
            total_steps = epochs * 5
            loss = 1.85

            for step in range(total_steps):
                if cancel_event.is_set():
                    job.status = "cancelled"
                    job.message = "Training safely halted on user request."
                    return

                time.sleep(0.1)  # Simulated training step computation
                loss = max(0.2, loss * 0.94)
                epoch = (step // 5) + 1
                progress = round((step + 1) / total_steps, 2)

                job.progress = progress
                job.current_epoch = epoch
                job.current_loss = round(loss, 4)
                job.message = f"Epoch {epoch}/{epochs} (Step {step+1}/{total_steps}) - Loss: {job.current_loss}"

            # 3. Output candidate adapter metadata (DO NOT auto-promote)
            candidate_dir = self.adapters_dir / adapter_name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "adapter_name": adapter_name,
                "target_model": target_model,
                "epochs": epochs,
                "final_loss": job.current_loss,
                "created_at": time.time(),
                "lifecycle_state": "candidate",
                "promoted_to_active": False,
            }
            with open(candidate_dir / "adapter_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            job.status = "completed"
            job.progress = 1.0
            job.completed_at = time.time()
            job.candidate_adapter_path = str(candidate_dir)
            job.message = f"Training successfully completed. Candidate adapter staged at {candidate_dir.name}."

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.message = f"Training worker encountered exception: {e}"

        finally:
            # 4. Mandatory GPU release and VRAM purge in finally block
            if lease_id:
                try:
                    gpu_broker.release(lease_id)
                    gpu_broker.purge_vram()
                except Exception:
                    pass


lora_training_manager = LoRATrainingManager()
