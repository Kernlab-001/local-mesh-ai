"""
Qwen 2.5 Coder 7B Integration & Natural-Language Technical Artist Planner.
Converts user prompts into strict, versioned JSON command plans and
comprehensive TechnicalArtistPlan specifications using an allowlist of
deterministic Blender operations, component hierarchies, and PBR definitions.
Features Windows CUDA DLL pre-loading for llama-cpp-python and exclusive GPU leasing.
"""

import os
import sys
import json
import re
import ctypes
import logging
from typing import Dict, Any, Optional
from core.gpu_broker import gpu_broker
from core.command_schemas import CommandPlan, TechnicalArtistPlan

logger = logging.getLogger("QwenPlanner")

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
1. set_material: {"op": "set_material", "target_object": "...", "preset": "custom"|"metal"|"gold"|"wood"|"plastic"|"stone"|"steel"|"brass"|"leather", "base_color": [r,g,b,a], "metallic": 0.0-1.0, "roughness": 0.0-1.0}
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

TECHNICAL_ARTIST_SYSTEM_PROMPT = """You are an expert AI 3D Technical Artist and Production Planner.
You convert natural language 3D asset requests into structured, validated TechnicalArtistPlan JSON.
You MUST output ONLY a valid JSON object matching the TechnicalArtistPlan schema.
DO NOT output Python code, Markdown wrappers, shell commands, or conversational preamble.

Schema Fields Required in JSON:
{
  "version": "2.0",
  "asset_name": "CamelCaseName",
  "category": "weapon" | "prop" | "furniture" | "vehicle" | "architecture" | "organic",
  "description": "Comprehensive technical description",
  "components": [
    {
      "name": "component_name",
      "role": "primary_structural" | "detail" | "functional" | "accessory" | "decorative",
      "parent_component": null or "parent_name",
      "material_preset": "preset_name"
    }
  ],
  "dimensions": {
    "target_height_m": 1.2,
    "target_width_m": 0.3,
    "target_depth_m": 0.1,
    "proportions": [0.25, 0.08, 1.0],
    "symmetry_axis": "none" | "x" | "y" | "z" | "radial"
  },
  "materials": [
    {
      "material_id": "mat_name",
      "target_component": "component_name",
      "preset": "steel" | "metal" | "gold" | "wood" | "plastic" | "stone" | "leather" | "brass" | "custom",
      "base_color_rgba": [r, g, b, 1.0],
      "roughness": 0.3,
      "metallic": 0.9,
      "notes": "visual description"
    }
  ],
  "textures": {
    "requires_pbr_maps": true,
    "map_types": ["diffuse", "normal", "roughness"],
    "resolution": 1024 | 2048,
    "uv_unwrapping_strategy": "smart_project",
    "bake_vertex_colors": true
  },
  "topology": {
    "target_triangles": 15000,
    "max_triangles": 30000,
    "mesh_style": "game_ready",
    "watertight_required": true,
    "manifold_geometry_required": true
  },
  "geometry_correction": {
    "decimate_ratio": null or float (e.g. 0.5),
    "smooth_shading": true,
    "auto_smooth_angle_deg": 30.0,
    "bevel_edges": false,
    "bevel_width_m": 0.005
  },
  "blender_commands": [
    {"op": "smooth_shading", "target_object": "CamelCaseName", "enable": true, "angle_degrees": 30.0}
  ],
  "validation": {
    "check_watertight": true,
    "check_non_manifold_edges": true,
    "min_face_count": 500
  },
  "improvement_notes": [
    {
      "aspect": "geometry" | "silhouette" | "materials" | "textures" | "scale" | "topology",
      "issue_description": "Weakness to prevent",
      "corrective_action": "Specific engineering remedy",
      "priority": "medium" | "high" | "critical"
    }
  ]
}
"""


class QwenPlanner:
    def __init__(self, model_path: Optional[str] = None):
        self._explicit_model_path = model_path
        self._llm = None

    def resolve_model_path(self) -> str:
        """Resolves local Qwen GGUF model path from explicit arg, config, or download manager."""
        if self._explicit_model_path and os.path.exists(self._explicit_model_path):
            return self._explicit_model_path

        from core.download_manager import download_manager
        stat = download_manager.verify_qwen_status()
        if stat.get("state") == "fully_installed" and stat.get("model_path"):
            return stat["model_path"]

        # Default fallback candidate
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        )
        if os.path.exists(default_path):
            return default_path

        raise FileNotFoundError(
            "Qwen 2.5 Coder GGUF weights not found. "
            "Please run: python scripts/download_models.py --qwen "
            "or set 'qwen_model_path' in config.json."
        )

    def _load_model(self, n_gpu_layers: int = 20):
        """Loads Qwen GGUF model under GPU lease."""
        model_path = self.resolve_model_path()
        _preload_cuda_dlls()
        from llama_cpp import Llama

        logger.info(f"[QwenPlanner] Loading Qwen GGUF from: {model_path} (n_gpu_layers={n_gpu_layers})")
        self._llm = Llama(
            model_path=model_path,
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
        Acquires GPU lease, prompts Qwen for CommandPlan, validates JSON response against schema,
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

    def generate_technical_artist_plan(
        self,
        user_prompt: str,
        scene_inventory: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0
    ) -> TechnicalArtistPlan:
        """
        Generates a comprehensive TechnicalArtistPlan specifying component decomposition,
        dimensions, PBR materials, topology targets, and Blender commands.
        """
        scene_desc = ""
        if scene_inventory:
            scene_desc = f"\nCurrent Scene Inventory: {json.dumps(scene_inventory)}"

        prompt = (
            f"<|im_start|>system\n{TECHNICAL_ARTIST_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nCreate a TechnicalArtistPlan for 3D asset request:\n\"{user_prompt}\"{scene_desc}\n"
            f"Output the JSON TechnicalArtistPlan:<|im_end|>\n<|im_start|>assistant\n"
        )

        raw_response = ""
        with gpu_broker.lease("Qwen_Technical_Artist_Planner", timeout=timeout):
            try:
                self._load_model()
                output = self._llm(
                    prompt,
                    max_tokens=1200,
                    temperature=0.1,
                    stop=["<|im_end|>", "<|endoftext|>"]
                )
                raw_response = output["choices"][0]["text"].strip()
            finally:
                self._unload_model()

        return self.parse_and_validate_technical_artist_plan(raw_response)

    def parse_and_validate(self, text: str) -> CommandPlan:
        """Extracts JSON substring and validates with CommandPlan schema."""
        # Strip markdown wrappers if present
        clean_text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"```$", "", clean_text.strip(), flags=re.MULTILINE)

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract valid JSON from Qwen response: {text[:200]}")

        json_str = json_match.group(0)
        # Strip illegal trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON from Qwen: {e}")

        return CommandPlan.model_validate(data)

    def parse_and_validate_technical_artist_plan(self, text: str) -> TechnicalArtistPlan:
        """Extracts JSON substring and validates with TechnicalArtistPlan schema."""
        clean_text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"```$", "", clean_text.strip(), flags=re.MULTILINE)

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract valid JSON from Qwen response: {text[:200]}")

        json_str = json_match.group(0)
        # Strip illegal trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON from Qwen: {e}")

        return TechnicalArtistPlan.model_validate(data)


# Global planner instance
qwen_planner = QwenPlanner()
