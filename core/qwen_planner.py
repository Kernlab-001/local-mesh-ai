"""
Qwen 2.5 Coder 7B Integration & Natural-Language Command Planner.
Converts user prompts into strict, versioned JSON command plans
using an allowlist of deterministic Blender operations.
Features Windows CUDA DLL pre-loading for llama-cpp-python.
"""

import os
import sys
import json
import re
import ctypes
from typing import Dict, Any, Optional
from core.gpu_broker import gpu_broker
from core.command_schemas import CommandPlan

def _preload_cuda_dlls():
    """Ensures CUDA 12.4 runtime DLLs are loaded before llama_cpp on Windows."""
    if sys.platform == "win32":
        venv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        site_packages = os.path.join(venv_base, ".venv", "Lib", "site-packages")
        torch_lib = os.path.join(site_packages, "torch", "lib")
        llama_lib = os.path.join(site_packages, "llama_cpp", "lib")

        for dll in ['cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll']:
            p = os.path.join(torch_lib, dll)
            if os.path.exists(p):
                try:
                    ctypes.windll.kernel32.LoadLibraryW(p)
                except Exception:
                    pass

        for dll in ['ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll', 'llama.dll']:
            p = os.path.join(llama_lib, dll)
            if os.path.exists(p):
                try:
                    ctypes.windll.kernel32.LoadLibraryW(p)
                except Exception:
                    pass

SYSTEM_PROMPT = """You are a 3D graphics assistant for Blender.
You convert the user's natural language request into a strict JSON command plan to modify 3D objects.
You MUST output ONLY a valid JSON object matching the CommandPlan schema.
DO NOT output Python code, shell commands, or conversational text.

Available operations in 'commands':
1. set_material: {"op": "set_material", "target_object": "...", "preset": "custom"|"metal"|"gold"|"wood"|"plastic"|"stone", "base_color": [r,g,b,a], "metallic": 0.0-1.0, "roughness": 0.0-1.0}
2. bevel: {"op": "bevel", "target_object": "...", "width": 0.001-0.5, "segments": 1-8}
3. set_height: {"op": "set_height", "target_object": "...", "height_meters": 0.1-50.0}
4. set_location: {"op": "set_location", "target_object": "...", "x": float, "y": float, "z": float}
5. set_rotation: {"op": "set_rotation", "target_object": "...", "rx_deg": float, "ry_deg": float, "rz_deg": float}
6. smooth_shading: {"op": "smooth_shading", "target_object": "...", "enable": true|false, "angle_degrees": 30.0}
7. add_subdivision: {"op": "add_subdivision", "target_object": "...", "levels": 1-3}
8. decimate: {"op": "decimate", "target_object": "...", "ratio": 0.1-0.9}
9. duplicate: {"op": "duplicate", "target_object": "...", "new_name": "...", "offset_x": float, "offset_y": float, "offset_z": float}

Output format:
{
  "version": "1.0",
  "description": "Brief description of changes",
  "commands": [ ... ]
}
"""

class QwenPlanner:
    def __init__(self, model_path: Optional[str] = None):
        if not model_path:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
        self.model_path = model_path
        self._llm = None

    def _load_model(self, n_gpu_layers: int = 24):
        """Loads Qwen GGUF model under GPU lease."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Qwen GGUF model not found at {self.model_path}")
        
        _preload_cuda_dlls()
        from llama_cpp import Llama
        
        # n_gpu_layers is dynamically calibrated by hardware_detector based on available VRAM
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False
        )

    def _unload_model(self):
        """Releases Qwen model references and frees memory."""
        self._llm = None
        gpu_broker.purge_vram()

    def generate_plan(self, user_prompt: str, scene_inventory: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> CommandPlan:
        """
        Acquires GPU lease, prompts Qwen, validates JSON response against schema,
        and unloads model.
        """
        scene_desc = ""
        if scene_inventory:
            scene_desc = f"\nCurrent Scene Inventory: {json.dumps(scene_inventory)}"

        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nUser Request: {user_prompt}{scene_desc}\nOutput the JSON CommandPlan:<|im_end|>\n<|im_start|>assistant\n"

        raw_response = ""
        with gpu_broker.lease("Qwen_Planner", timeout=timeout):
            try:
                self._load_model()
                output = self._llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.2,
                    stop=["<|im_end|>", "```\n\n"]
                )
                raw_response = output["choices"][0]["text"].strip()
            finally:
                self._unload_model()

        return self.parse_and_validate(raw_response)

    def parse_and_validate(self, text: str) -> CommandPlan:
        """Extracts JSON substring and validates with Pydantic schema."""
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract valid JSON from Qwen response: {text[:200]}")
            
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON from Qwen: {e}")

        return CommandPlan.model_validate(data)

# Global planner instance
qwen_planner = QwenPlanner()
