"""
Prompt-to-Image Concept Generator for Local Mesh AI.
Uses local StabilityAI SD-Turbo (1-step real-time inference) on local CUDA GPU.
Enforces strict GPU lease acquisition, intelligent CLIP 77-token prompt optimization,
priority retention of core material and color terms, and total VRAM purge before TripoSR / Blender stages.
"""

import os
import gc
import re
import time
import torch
from typing import Optional, List, Tuple
from PIL import Image
from core.gpu_broker import gpu_broker
from core.input_validator import input_validator, ValidationError

# Core semantic keywords to prioritize in token budget
COLOR_KEYWORDS = {
    'walnut', 'dark walnut', 'brown', 'ivory', 'cream', 'ivory-cream', 'white',
    'black', 'red', 'crimson', 'green', 'emerald', 'blue', 'navy', 'gold',
    'golden', 'silver', 'gray', 'grey', 'yellow', 'orange', 'purple', 'cyan',
    'bronze', 'copper'
}

MATERIAL_KEYWORDS = {
    'wood', 'wooden', 'timber', 'fabric', 'upholstered', 'upholstery', 'cushion',
    'backrest', 'tufted', 'leather', 'metal', 'steel', 'brass', 'iron',
    'ceramic', 'glaze', 'rubber', 'glass', 'plastic', 'carved', 'buttons',
    'legs', 'frame', 'chassis', 'armor', 'seat'
}

FLUFF_KEYWORDS = {
    'photorealistic', 'realistic', 'high quality', 'game ready', 'sharp silhouette',
    'isometric perspective', 'prop', 'studio lighting', '4k', '8k', 'hyperrealistic',
    'unreal engine', 'trending on artstation', 'masterpiece', 'subtle stitching',
    'detailed trim', 'ultra high res'
}

def optimize_prompt_for_clip(
    user_prompt: str,
    tokenizer=None,
    max_content_tokens: int = 70
) -> str:
    """
    Adapts user prompt to fit strictly within CLIP's 77-token limit (75 content tokens max).
    Prioritizes subject, material, and color specifications over generic quality adjectives.
    Eliminates tokenizer truncation warnings and preserves critical visual features.
    """
    clean = input_validator.validate_prompt(user_prompt)
    suffix = ", single object, white background"

    def estimate_tokens(text: str) -> int:
        if tokenizer is not None:
            try:
                enc = tokenizer.encode(text)
                return max(0, len(enc) - 2)
            except Exception:
                pass
        # Conservative word-to-token ratio approximation (1 word ~ 1.3 tokens)
        words = re.findall(r'\b\w+\b', text)
        return int(len(words) * 1.3) + 2

    suffix_tokens = estimate_tokens(suffix)
    budget = max(20, max_content_tokens - suffix_tokens)

    # If raw prompt already fits within budget, return cleanly
    if estimate_tokens(clean) <= budget:
        return f"{clean.rstrip('.,; ')}{suffix}"

    # Split into constituent clauses by punctuation and conjunctions
    raw_phrases = [p.strip() for p in re.split(r'[,;]+|\band\b', clean) if p.strip()]

    scored_phrases: List[Tuple[float, int, str]] = []
    for idx, phrase in enumerate(raw_phrases):
        p_lower = phrase.lower()
        score = 0.0

        # Core subject clause gets highest anchor priority
        if idx == 0:
            score += 15.0

        # High priority for material and color definitions
        has_color = any(c in p_lower for c in COLOR_KEYWORDS)
        has_mat = any(m in p_lower for m in MATERIAL_KEYWORDS)
        if has_color:
            score += 8.0
        if has_mat:
            score += 7.0

        # Demote generic boilerplate / fluff words
        for f in FLUFF_KEYWORDS:
            if f in p_lower:
                score -= 4.0

        scored_phrases.append((score, idx, phrase))

    # Sort descending by priority score
    scored_phrases.sort(key=lambda x: (-x[0], x[1]))

    selected: List[Tuple[int, str]] = []
    current_tokens = 0

    for score, orig_idx, phrase in scored_phrases:
        p_toks = estimate_tokens(phrase)
        if current_tokens + p_toks <= budget:
            selected.append((orig_idx, phrase))
            current_tokens += p_toks
        else:
            # Check if trimming fluff allows the clause to fit
            words = phrase.split()
            pruned_words = [w for w in words if w.lower() not in FLUFF_KEYWORDS]
            pruned_phrase = " ".join(pruned_words)
            pruned_toks = estimate_tokens(pruned_phrase)
            if pruned_phrase and (current_tokens + pruned_toks <= budget):
                selected.append((orig_idx, pruned_phrase))
                current_tokens += pruned_toks

    # Restore natural sentence order
    selected.sort(key=lambda x: x[0])
    combined_body = ", ".join(p for _, p in selected)

    # Fallback to truncated original if selector resulted in empty string
    if not combined_body:
        words = clean.split()
        combined_body = " ".join(words[:budget])

    final_prompt = f"{combined_body.rstrip('.,; ')}{suffix}"
    return final_prompt


class PromptToImageEngine:
    def __init__(self, model_id: str = "stabilityai/sd-turbo"):
        self.model_id = model_id
        self._pipe = None
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    subfolder="tokenizer"
                )
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def _prepare_prompt(self, user_prompt: str) -> str:
        """
        Enriches and adapts the prompt to fit strictly within CLIP's 77-token limit
        while preserving all material, color, and subject attributes.
        """
        tok = self._get_tokenizer()
        optimized = optimize_prompt_for_clip(
            user_prompt=user_prompt,
            tokenizer=tok,
            max_content_tokens=70
        )
        return optimized

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
            "dark background, complex shadows, text, watermark, low quality, noisy, distorted"
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
                self._tokenizer = pipe.tokenizer

                # Verify token count of effective prompt to guarantee zero truncation
                token_count = len(pipe.tokenizer.encode(final_prompt))
                if token_count > 77:
                    # Hard truncate strictly at token boundary before inference
                    tokens = pipe.tokenizer.encode(final_prompt)[:76]
                    final_prompt = pipe.tokenizer.decode(tokens, skip_special_tokens=True)

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
