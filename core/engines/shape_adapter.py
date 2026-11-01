"""
Shap-E Engine Adapter for Local Mesh AI.
Provides Direct Text-to-3D (Strategy A) and Image-to-3D using OpenAI's Shap-E
via Hugging Face Diffusers, running locally on CUDA or CPU with MIT licensing.
"""

import os
import time
import shutil
import torch
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
from PIL import Image

from core.engines.base import Base3DEngine, EngineStatus, StandardizedAsset
from core.gpu_broker import gpu_broker

try:
    import trimesh
except ImportError:
    trimesh = None


class ShapEAdapter(Base3DEngine):
    def __init__(self):
        super().__init__(
            engine_id="shap_e",
            name="OpenAI Shap-E (Direct Latent 3D)",
            description="Direct text-conditioned and image-conditioned 3D implicit surface generator by OpenAI."
        )
        self.text_model_id = "openai/shap-e"
        self.image_model_id = "openai/shap-e-img2img"

    @property
    def supported_modalities(self) -> List[str]:
        return ["text-to-3d", "image-to-3d"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "pbr_textures": False,             # Implicit color fields, not PBR maps
            "vertex_colors": True,
            "uv_unwrapping": False,
            "watertight": True,
            "direct_text_conditioned": True,   # Direct Text-to-3D without intermediate 2D diffusion
            "typical_generation_seconds": 25.0
        }

    @property
    def hardware_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 4.0,
            "recommended_vram_gb": 6.0,
            "requires_cuda": False,           # Supports CPU fallback
            "supports_cpu": True,
            "recommended_precision": "fp16"
        }

    def _is_cached(self, model_id: str) -> bool:
        """Checks if model weights exist in local Hugging Face cache without network calls."""
        try:
            from pathlib import Path
            hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
            cache_folder = "models--" + model_id.replace("/", "--")
            snapshots_dir = hf_cache / cache_folder / "snapshots"
            if snapshots_dir.exists():
                for s in snapshots_dir.iterdir():
                    if s.is_dir() and any(s.iterdir()):
                        return True
        except Exception:
            pass
        return False

    def get_status(self) -> EngineStatus:
        """Determines if Shap-E is ready or needs download."""
        import importlib.util
        if not importlib.util.find_spec("diffusers"):
            return EngineStatus.NOT_INSTALLED

        # Check CUDA or CPU capability
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram < 3.5:
                return EngineStatus.UNSUPPORTED_HARDWARE

        if self._is_cached(self.text_model_id):
            return EngineStatus.READY
        return EngineStatus.NEEDS_DOWNLOAD

    def get_status_details(self) -> Dict[str, Any]:
        status = self.get_status()
        return {
            "status": status.value,
            "engine_id": self.engine_id,
            "name": self.name,
            "ready": status == EngineStatus.READY,
            "cached": status == EngineStatus.READY,
            "approximate_download_size": "~1.7 GB",
            "supported_modalities": self.supported_modalities,
            "requirements": self.hardware_requirements,
            "license": "MIT License (Permissive)"
        }

    def validate_input(self, input_data: Any, modality: str) -> Tuple[bool, Optional[str]]:
        if modality not in self.supported_modalities:
            return False, f"Modality '{modality}' not supported by Shap-E."

        if modality == "text-to-3d":
            if not input_data or not isinstance(input_data, str) or len(input_data.strip()) < 2:
                return False, "Prompt must be a non-empty string of at least 2 characters."
            return True, None

        elif modality == "image-to-3d":
            p = str(input_data)
            if not os.path.exists(p):
                return False, f"Image file not found: {p}"
            return True, None

        return False, "Unknown modality."

    def estimate_vram_gb(self, modality: str, params: Optional[Dict[str, Any]] = None) -> float:
        return 4.2  # Typical peak VRAM under fp16

    def generate(
        self,
        input_data: Any = None,
        output_dir: Optional[str] = None,
        modality: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        **kwargs
    ) -> StandardizedAsset:
        params = params or {}
        mod = modality or ("text-to-3d" if "prompt" in params else "image-to-3d")
        inp = input_data or params.get("prompt") or params.get("image_path")
        out_dir = output_dir or params.get("output_dir") or str(Path(__file__).resolve().parent.parent.parent / "output")

        is_valid, err = self.validate_input(inp, mod)
        if not is_valid:
            raise ValueError(f"Input validation error: {err}")

        os.makedirs(out_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        num_steps = params.get("num_inference_steps", 64)
        guidance = params.get("guidance_scale", 15.0)
        frame_size = params.get("frame_size", 256)

        start_time = time.time()
        pipe = None
        target_glb = os.path.join(out_dir, "mesh.glb")
        temp_ply = os.path.join(out_dir, "temp_shape.ply")

        with gpu_broker.lease("ShapE_Inference", timeout=180):
            try:
                from diffusers.utils import export_to_ply

                if modality == "text-to-3d":
                    if progress_callback:
                        progress_callback("Loading Shap-E Text-to-3D model weights", 15)

                    from diffusers import ShapEPipeline
                    pipe = ShapEPipeline.from_pretrained(
                        self.text_model_id,
                        torch_dtype=dtype
                    ).to(device)

                    if progress_callback:
                        progress_callback("Executing direct latent 3D inference", 45)

                    outputs = pipe(
                        prompt=str(input_data),
                        guidance_scale=guidance,
                        num_inference_steps=num_steps,
                        frame_size=frame_size,
                        output_type="mesh"
                    )

                else:  # image-to-3d
                    if progress_callback:
                        progress_callback("Loading Shap-E Image-to-3D model weights", 15)

                    from diffusers import ShapEImg2ImgPipeline
                    pipe = ShapEImg2ImgPipeline.from_pretrained(
                        self.image_model_id,
                        torch_dtype=dtype
                    ).to(device)

                    input_img = Image.open(str(input_data)).convert("RGB").resize((frame_size, frame_size))

                    if progress_callback:
                        progress_callback("Executing image-conditioned 3D inference", 45)

                    outputs = pipe(
                        image=input_img,
                        guidance_scale=guidance,
                        num_inference_steps=num_steps,
                        frame_size=frame_size,
                        output_type="mesh"
                    )

                if progress_callback:
                    progress_callback("Exporting and converting implicit surface to GLB", 75)

                mesh_obj = outputs.images[0]
                export_to_ply(mesh_obj, temp_ply)

                # Convert PLY to GLB via trimesh
                if trimesh is not None and os.path.exists(temp_ply):
                    loaded_m = trimesh.load(temp_ply)
                    loaded_m.export(target_glb, file_type="glb")
                    try:
                        os.remove(temp_ply)
                    except OSError:
                        pass
                else:
                    target_glb = temp_ply

            finally:
                if pipe is not None:
                    del pipe
                gpu_broker.purge_vram()

        elapsed = time.time() - start_time

        # Extract mesh properties
        v_count, f_count = 0, 0
        dims = (1.0, 1.0, 1.0)
        is_wt = False
        if trimesh is not None and os.path.exists(target_glb):
            try:
                tm = trimesh.load(target_glb, force="mesh")
                v_count = len(tm.vertices)
                f_count = len(tm.faces)
                ext = tm.extents
                dims = (float(ext[0]), float(ext[1]), float(ext[2]))
                is_wt = getattr(tm, "is_watertight", False)
            except Exception:
                pass

        if progress_callback:
            progress_callback("Shap-E generation complete", 100)

        return StandardizedAsset(
            mesh_path=target_glb,
            format="glb",
            vertex_count=v_count,
            face_count=f_count,
            dimensions=dims,
            has_uvs=False,
            has_vertex_colors=True,
            is_watertight=is_wt,
            engine_id=self.engine_id,
            engine_name=self.name,
            generation_time_seconds=elapsed,
            peak_vram_gb=4.2,
            diagnostics={
                "steps": num_steps,
                "guidance_scale": guidance,
                "device": device
            }
        )
