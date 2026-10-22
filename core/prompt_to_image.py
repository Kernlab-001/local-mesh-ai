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

class PromptToImageEngine:
    def __init__(self, model_id: str = "stabilityai/sd-turbo"):
        self.model_id = model_id
        self._pipe = None

    def _prepare_prompt(self, user_prompt: str) -> str:
        """
        Enriches the prompt to generate isolated, clean, orthographic 3D game asset concepts
        with solid neutral/white background suitable for rembg and TripoSR reconstruction.
        """
        clean = user_prompt.strip()
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
        lease = gpu_broker.acquire("sd_turbo", timeout=120)

        pipe = None
        try:
            from diffusers import AutoPipelineForText2Image

            # Setup generator for deterministic seed if specified
            generator = None
            if seed >= 0:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            # Load pipeline directly into float16 on CUDA
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.to("cuda")

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
            return output_path

        finally:
            # Complete VRAM cleanup
            if pipe is not None:
                del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gpu_broker.release(lease)

prompt_engine = PromptToImageEngine()
