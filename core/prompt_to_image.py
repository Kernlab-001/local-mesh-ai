"""
Prompt-to-Image Concept Generator for Local Mesh AI.
Uses local StabilityAI SD-Turbo (1-step real-time inference) on local CUDA GPU.
Enforces strict GPU lease acquisition, prompt styling for 3D reconstruction,
and total VRAM purge before TripoSR / Blender stages.
"""

import os
import gc
import time
import torch
from typing import Optional
from PIL import Image
from core.gpu_broker import gpu_broker
from core.input_validator import input_validator, ValidationError

class PromptToImageEngine:
    def __init__(self, model_id: str = "stabilityai/sd-turbo"):
        self.model_id = model_id
        self._pipe = None

    def _prepare_prompt(self, user_prompt: str) -> str:
        """
        Enriches the prompt to generate isolated, clean, orthographic 3D game asset concepts
        with solid neutral/white background suitable for rembg and TripoSR reconstruction.
        """
        clean = input_validator.validate_prompt(user_prompt)
        enhanced = (
            f"3d asset of {clean}, centered, single object, solid white background, "
            f"studio lighting, photorealistic prop, game ready asset, high quality, "
            f"sharp silhouette, isometric perspective"
        )
        return enhanced

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        seed: int = -1
    ) -> str:
        """
        Acquires GPU lease, loads SD-Turbo, generates concept image in 1 step,
        saves image, purges VRAM, and returns file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        final_prompt = self._prepare_prompt(prompt)
        neg_prompt = negative_prompt or (
            "blurry, cropped, multiple views, multiple objects, human hands, person, "
            "dark background, complex shadows, text, watermark, low quality, noisy"
        )

        # Acquire GPU lease to avoid VRAM collision with TripoSR or Qwen
        with gpu_broker.lease("sd_turbo", timeout=120):
            pipe = None
            try:
                from diffusers import AutoPipelineForText2Image

                use_cuda = torch.cuda.is_available()
                target_device = "cuda" if use_cuda else "cpu"
                target_dtype = torch.float16 if use_cuda else torch.float32

                # Setup generator for deterministic seed if specified
                generator = None
                if seed >= 0:
                    generator = torch.Generator(device=target_device).manual_seed(seed)

                # Load pipeline (float16 on CUDA, float32 on CPU)
                load_kwargs = {"torch_dtype": target_dtype}
                if use_cuda:
                    load_kwargs["variant"] = "fp16"

                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id,
                    **load_kwargs
                )
                pipe.to(target_device)

                # SD-Turbo is trained for 1 step with guidance_scale=0.0
                image = pipe(
                    prompt=final_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    generator=generator,
                    height=height,
                    width=width
                ).images[0]

                image.save(output_path)
                input_validator.validate_generated_concept(output_path)
                return output_path

            except torch.cuda.OutOfMemoryError as oom:
                raise RuntimeError(f"CUDA Out-of-Memory during concept generation: {oom}. Free GPU VRAM before retrying.")
            finally:
                # Complete VRAM cleanup
                if pipe is not None:
                    del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

prompt_engine = PromptToImageEngine()
