"""
Hybrid Text-to-3D Adapter (Strategy B: Text-to-Image-to-3D).
Combines local SD-Turbo 1-step concept synthesis, canonical 4-view projections,
and downstream volumetric 3D reconstruction (TripoSR or secondary engine).
Exposes intermediate concept artwork for user inspection and editing.
"""

import os
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset
from core.prompt_to_image import prompt_engine
from core.multiview_synthesizer import multiview_synthesizer
from core.foreground_segmenter import foreground_segmenter, SegmentationError
from core.input_validator import input_validator
from core.gpu_broker import gpu_broker


class HybridTextTo3DAdapter(Base3DEngine):
    def __init__(self, secondary_engine: Optional[Base3DEngine] = None):
        super().__init__(
            engine_id="hybrid_sd_triposr",
            name="Hybrid Text-to-3D (SD-Turbo + TripoSR)",
            description="Multi-stage text-driven pipeline: Synthesizes high-fidelity 2D concept art first, projects canonical views, then reconstructs 3D geometry."
        )
        self.secondary_engine = secondary_engine

    @property
    def supported_modalities(self) -> List[str]:
        return ["text-to-3d"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "pbr_textures": True,
            "vertex_colors": True,
            "uv_unwrapping": True,
            "watertight": True,
            "direct_text_conditioned": False,   # Multi-stage with visual concept preview
            "exposes_concept_art": True,
            "typical_generation_seconds": 35.0
        }

    @property
    def hardware_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 4.0,
            "recommended_vram_gb": 6.0,
            "requires_cuda": True,
            "supports_cpu": True,
            "recommended_precision": "fp16"
        }

    def get_status(self) -> EngineStatus:
        """Determines if both SD-Turbo and downstream reconstruction engine are ready."""
        from core.engines.triposr_adapter import TripoSRAdapter
        downstream = self.secondary_engine or TripoSRAdapter()
        down_status = downstream.get_status()
        if down_status != EngineStatus.READY:
            return down_status

        from core.model_manager import model_manager, ModelMetadata
        sd_meta = ModelMetadata(
            model_id="sd_turbo",
            name="SD-Turbo",
            engine_id="hybrid_sd_triposr",
            description="",
            approx_size_gb=2.5,
            hf_repo_id="stabilityai/sd-turbo"
        )
        if not model_manager.is_model_installed(sd_meta):
            return EngineStatus.NEEDS_DOWNLOAD

        return EngineStatus.READY

    def get_status_details(self) -> Dict[str, Any]:
        status = self.get_status()
        from core.engines.triposr_adapter import TripoSRAdapter
        downstream = self.secondary_engine or TripoSRAdapter()
        from core.model_manager import model_manager, ModelMetadata
        sd_meta = ModelMetadata(
            model_id="sd_turbo",
            name="SD-Turbo",
            engine_id="hybrid_sd_triposr",
            description="",
            approx_size_gb=2.5,
            hf_repo_id="stabilityai/sd-turbo"
        )
        sd_installed = model_manager.is_model_installed(sd_meta)
        downstream_ready = downstream.get_status() == EngineStatus.READY

        return {
            "status": status.value,
            "engine_id": self.engine_id,
            "name": self.name,
            "ready": status == EngineStatus.READY,
            "sd_turbo_installed": sd_installed,
            "downstream_ready": downstream_ready,
            "multi_stage": True,
            "stages": ["1. Concept Art Diffusion (SD-Turbo)", "2. 4-View Canonical Projection", "3. Volumetric Reconstruction (TripoSR)"],
            "requirements": self.hardware_requirements,
            "license": "Apache 2.0 (Code) / Stability AI Non-Commercial Research (SD-Turbo Weights)"
        }

    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        if modality != "text-to-3d":
            return False, f"Modality '{modality}' not supported by Hybrid Text-to-3D."
        try:
            input_validator.validate_prompt(str(input_data))
            return True, None
        except Exception as e:
            return False, str(e)

    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        return 3.5  # Sequential execution keeps peak VRAM below 3.5 GB

    def generate(
        self,
        input_data: Any = None,
        output_dir: Optional[str] = None,
        modality: Optional[str] = "text-to-3d",
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        **kwargs
    ) -> StandardizedAsset:
        params = params or {}
        prompt_text = str(input_data or params.get("prompt", "")).strip()
        out_dir = output_dir or params.get("output_dir") or str(Path(__file__).resolve().parent.parent.parent / "output")

        is_valid, err = self.validate_input(prompt_text, modality or "text-to-3d")
        if not is_valid:
            raise ValueError(f"Input validation error: {err}")

        os.makedirs(out_dir, exist_ok=True)
        start_time = time.time()

        # Stage 1: Synthesize concept artwork via SD-Turbo
        if progress_callback:
            progress_callback("Synthesizing 2D concept artwork (SD-Turbo)", 15)

        concept_path = os.path.join(out_dir, "concept.png")
        try:
            prompt_engine.generate(
                prompt=prompt_text,
                output_path=concept_path,
                seed=params.get("seed", -1)
            )
        except Exception as synth_err:
            raise RuntimeError(f"Stage 1 [Concept Art Synthesis] failed: {synth_err}")

        # Stage 2: Offline Foreground Segmentation & Alpha Matting
        if progress_callback:
            progress_callback("Isolating foreground subject and generating alpha matte", 30)

        segmented_path = os.path.join(out_dir, "concept_segmented.png")
        seg_result = None
        try:
            seg_result = foreground_segmenter.segment(
                image_path=concept_path,
                output_path=segmented_path
            )
            reconstruction_input = segmented_path
        except SegmentationError as seg_err:
            raise RuntimeError(f"Stage 2 [Foreground Segmentation] failed: {seg_err}")
        except Exception as seg_err:
            raise RuntimeError(f"Stage 2 [Foreground Matting Error]: {seg_err}")

        # Stage 3: Canonical 4-view projections
        if progress_callback:
            progress_callback("Generating 4-view canonical perspective grid", 45)

        mv_dir = os.path.join(out_dir, "multiview")
        os.makedirs(mv_dir, exist_ok=True)
        try:
            multiview_synthesizer.generate_canonical_views(reconstruction_input, mv_dir)
        except Exception as mv_err:
            print(f"[Notice] Multi-view reference notice: {mv_err}. Proceeding with primary concept.")

        # Stage 4: Reconstruct 3D geometry using TripoSR
        if progress_callback:
            progress_callback("Reconstructing 3D neural geometry from isolated subject", 60)

        from core.engines.triposr_adapter import TripoSRAdapter
        downstream = self.secondary_engine or TripoSRAdapter()

        try:
            asset = downstream.generate(
                input_data=reconstruction_input,
                output_dir=out_dir,
                modality="image-to-3d",
                params=params,
                progress_callback=lambda msg, pct: progress_callback(msg, 60 + int(pct * 0.35)) if progress_callback else None
            )
        except Exception as rec_err:
            raise RuntimeError(f"Stage 4 [Volumetric Reconstruction] failed: {rec_err}")

        elapsed = time.time() - start_time
        asset.engine_id = self.engine_id
        asset.engine_name = self.name
        asset.generation_time_seconds = elapsed
        asset.preview_image_path = reconstruction_input
        asset.diagnostics["multi_stage"] = True
        asset.diagnostics["prompt"] = prompt_text
        asset.diagnostics["concept_image"] = concept_path
        asset.diagnostics["segmented_concept_image"] = reconstruction_input
        if seg_result:
            asset.diagnostics["segmentation"] = seg_result.to_dict()

        if progress_callback:
            progress_callback("Hybrid Text-to-3D asset ready", 100)

        return asset
