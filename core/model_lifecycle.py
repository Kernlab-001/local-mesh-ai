"""
Production-Grade Sequential Model Lifecycle & Memory Manager.
Implements a strict "Zero-VRAM Overlap" state machine to run heavy neural
3D pipelines (SD-Turbo, Multi-View Synthesizer, TripoSR, and Qwen LLM) on
consumer GPUs with 6GB–8GB VRAM without Out-Of-Memory (OOM) collisions.
Supports both CUDA GPU execution and CPU fallback seamlessly.
"""

import os
import sys
import gc
import time
import logging
import threading
from enum import Enum
from typing import Dict, Any, Optional, Generator, Callable
from contextlib import contextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.gpu_broker import gpu_broker

logger = logging.getLogger("ModelLifecycle")

class ModelSlot(str, Enum):
    CONCEPT_DIFFUSION = "concept_diffusion"         # SD-Turbo (1-step Latent Diffusion)
    MULTIVIEW_SYNTH = "multiview_synth"             # Zero123++ / Orthogonal Multi-View Generator
    NEURAL_RECONSTRUCTION = "neural_reconstruction" # TripoSR NeRF / Marching Cubes
    LLM_PLANNER = "llm_planner"                     # Qwen 2.5 Coder 7B (GGUF Quantized)
    SEGMENTATION_3D = "segmentation_3d"             # PartField / 3D Part Decomposition
    VISION_LANGUAGE = "vision_language"             # Qwen3-VL-4B Multimodal Semantic Interpreter
    TEXTURE_SYNTHESIS = "texture_synthesis"         # PBR Texture Synthesizer

class ModelLocation(str, Enum):
    UNLOADED = "unloaded"       # Resides purely on disk
    CPU_RAM = "cpu_ram"         # Stored in Host RAM (fast swap back to VRAM)
    GPU_VRAM = "gpu_vram"       # Resident in GPU VRAM ready for tensor execution

class SequentialModelManager:
    """
    Coordinates single-resident GPU execution. Enforces that only ONE neural
    model occupies GPU VRAM at any point during pipeline execution.
    Thread-safe and supports CPU/GPU device selection.
    """

    def __init__(self, vram_budget_gb: float = 6.0):
        self.vram_budget_gb = vram_budget_gb
        self.active_slot: Optional[ModelSlot] = None
        self._cpu_cache: Dict[ModelSlot, Any] = {}
        self._gpu_resident: Dict[ModelSlot, Any] = {}
        self._locations: Dict[ModelSlot, ModelLocation] = {slot: ModelLocation.UNLOADED for slot in ModelSlot}
        self._lock = threading.RLock()

    def get_vram_usage(self) -> Dict[str, float]:
        """Returns allocated, reserved, and free GPU VRAM in Gigabytes."""
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                res = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return {
                    "allocated_gb": round(alloc, 2),
                    "reserved_gb": round(res, 2),
                    "total_gb": round(total, 2),
                    "free_gb": round(total - res, 2)
                }
        except Exception:
            pass
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0}

    def purge_cuda_memory(self):
        """Forces hard PyTorch CUDA memory release and garbage collection."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()

    def offload_active_model(self, keep_in_cpu_ram: bool = False):
        """
        Evicts currently active GPU model from VRAM.
        If keep_in_cpu_ram is True, moves tensors to Host RAM for rapid recall.
        Otherwise deletes the instance entirely to minimize system memory footprint.
        """
        with self._lock:
            if self.active_slot is None:
                self.purge_cuda_memory()
                return

            evicting_slot = self.active_slot
            logger.info(f"[Zero-VRAM] Evicting active slot '{evicting_slot.value}' from GPU VRAM...")

            model_obj = self._gpu_resident.pop(evicting_slot, None)
            if model_obj is not None:
                if keep_in_cpu_ram and hasattr(model_obj, "to"):
                    try:
                        model_obj.to("cpu")
                        self._cpu_cache[evicting_slot] = model_obj
                        self._locations[evicting_slot] = ModelLocation.CPU_RAM
                        logger.info(f"[Zero-VRAM] Offloaded '{evicting_slot.value}' to CPU RAM.")
                    except Exception as e:
                        logger.warning(f"Failed to offload to CPU: {e}. Purging.")
                        del model_obj
                        self._locations[evicting_slot] = ModelLocation.UNLOADED
                else:
                    del model_obj
                    self._locations[evicting_slot] = ModelLocation.UNLOADED

            self.active_slot = None
            self.purge_cuda_memory()
            vram = self.get_vram_usage()
            logger.info(f"[Zero-VRAM] GPU VRAM cleared. Free: {vram['free_gb']} GB")

    def prepare_slot(
        self,
        slot: ModelSlot,
        model_loader: Callable[[], Any],
        precision: str = "fp16",
        keep_previous_in_ram: bool = False
    ) -> Any:
        """
        Ensures target slot is ready in GPU VRAM (or CPU) while strictly offloading any previous resident.
        """
        with self._lock:
            # If already active, return immediately
            if self.active_slot == slot and slot in self._gpu_resident:
                return self._gpu_resident[slot]

            # 1. Evict whatever was previously holding the GPU
            if self.active_slot is not None and self.active_slot != slot:
                self.offload_active_model(keep_in_cpu_ram=keep_previous_in_ram)

            import torch
            has_cuda = torch.cuda.is_available()
            target_device = "cuda" if has_cuda else "cpu"
            target_dtype = torch.float16 if (has_cuda and precision == "fp16") else torch.float32

            # 2. Check if cached in CPU RAM
            if slot in self._cpu_cache:
                logger.info(f"[Zero-VRAM] Swapping '{slot.value}' from CPU RAM to {target_device}...")
                model_obj = self._cpu_cache.pop(slot)
                try:
                    if hasattr(model_obj, "to"):
                        model_obj.to(target_device, dtype=target_dtype)
                    self._gpu_resident[slot] = model_obj
                    self.active_slot = slot
                    self._locations[slot] = ModelLocation.GPU_VRAM if has_cuda else ModelLocation.CPU_RAM
                    return model_obj
                except Exception as e:
                    logger.warning(f"Failed swap: {e}. Reloading from scratch.")
                    del model_obj
                    self.purge_cuda_memory()

            # 3. Load fresh model via loader callback
            logger.info(f"[Zero-VRAM] Loading fresh '{slot.value}' into {target_device} (Precision: {precision})...")
            model_obj = model_loader()
            self._gpu_resident[slot] = model_obj
            self.active_slot = slot
            self._locations[slot] = ModelLocation.GPU_VRAM if has_cuda else ModelLocation.CPU_RAM
            return model_obj

    @contextmanager
    def execution_scope(
        self,
        slot: ModelSlot,
        model_loader: Callable[[], Any],
        timeout: float = 180.0,
        precision: str = "fp16",
        offload_on_exit: bool = True
    ) -> Generator[Any, None, None]:
        """
        Guarded execution context:
        1. Acquires atomic cross-process GPU lease from gpu_broker.
        2. Evicts all other models from VRAM.
        3. Loads target model.
        4. Yields model for inference.
        5. Automatically purges VRAM upon exit.
        """
        with gpu_broker.lease(f"Slot_{slot.value}", timeout=timeout):
            try:
                model = self.prepare_slot(slot, model_loader, precision=precision)
                yield model
            finally:
                if offload_on_exit:
                    self.offload_active_model(keep_in_cpu_ram=False)
                else:
                    self.purge_cuda_memory()

# Global Singleton Manager
model_lifecycle = SequentialModelManager()
