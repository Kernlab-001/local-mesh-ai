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
from typing import Dict, Any, Optional, List
from core.gpu_broker import gpu_broker
from core.command_schemas import (
    ComponentEditPlan,
    ComponentInvariant,
    CommandPlan,
    TechnicalArtistPlan,
    ComponentSemanticProposal,
    ComponentSemanticAssignment,
    PrecisionCorrectionPlan,
    PrecisionToolCall
)

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


SEMANTIC_INTERPRETATION_SYSTEM_PROMPT = """You are an expert AI 3D Technical Artist and Anatomical Segmenter.
Given a 3D asset description and a list of geometrically detected candidate components (with bounding dimensions and relative vertical heights),
you propose semantic labels (e.g. blade, crossguard, grip, pommel for a sword; or seat, backrest, legs for a chair).
If an assignment cannot be determined with high certainty based on geometric evidence, set confidence < 0.70 and is_uncertain = true.

Output strictly valid JSON matching this schema:
{
  "version": "2.0",
  "asset_id": "asset_id",
  "category": "weapon" | "furniture" | "prop" | "vehicle" | "architecture",
  "assignments": [
    {
      "component_id": "comp_id",
      "assigned_semantic_label": "label_name",
      "confidence": 0.0-1.0,
      "is_uncertain": true|false,
      "parent_component_id": null or "parent_id",
      "reasoning": "Geometric rationale based on relative position and aspect ratio"
    }
  ],
  "overall_confidence": 0.85,
  "has_uncertain_components": false,
  "summary": "Brief summary of component breakdown"
}
"""

PRECISION_CORRECTION_SYSTEM_PROMPT = """You are an expert AI 3D Technical Artist for Precision Alignment and Calibration.
Given a user request, current geometry measurements, and target dimensional constraints,
you emit a sequence of allowlisted tool calls to correct orientation, grounding, symmetry, and scale.
Available Tools in 'tool_calls':
- align_and_ground: {"tool_name": "align_and_ground", "arguments": {"mesh_path": "...", "normalize_orientation": true, "ground_z": true, "center_xy": true, "target_height_m": 1.2}}
- calibrate_scale: {"tool_name": "calibrate_scale", "arguments": {"mesh_path": "...", "target_height_m": 1.2}}
- enforce_symmetry: {"tool_name": "enforce_symmetry", "arguments": {"mesh_path": "...", "symmetry_plane": "yz_plane"|"xz_plane"}}
- decimate_mesh: {"tool_name": "decimate_mesh", "arguments": {"mesh_path": "...", "ratio": 0.5}}

Output strictly valid JSON matching:
{
  "version": "2.0",
  "plan_id": "corr_plan_01",
  "asset_name": "asset_name",
  "target_height_m": 1.2,
  "target_upright": true,
  "target_grounded": true,
  "tool_calls": [ ... ],
  "rationale": "Engineering justification"
}
"""


class QwenPlanner:
    def __init__(self, model_path: Optional[str] = None):
        self._explicit_model_path = model_path
        self._llm = None

    def resolve_model_path(self) -> str:
        """Resolves local Qwen GGUF model path from central model_path_resolver."""
        if self._explicit_model_path:
            if os.path.exists(self._explicit_model_path):
                return self._explicit_model_path
            raise FileNotFoundError(f"Qwen GGUF model not found at {self._explicit_model_path}")

        from core.model_paths import model_path_resolver
        p = model_path_resolver.resolve_qwen_coder_path()
        if p and p.exists():
            return str(p)

        raise FileNotFoundError(
            "Qwen 2.5 Coder GGUF weights not found. "
            "Please verify path in .local_models_config.json or QWEN_CODER_PATH."
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

    def set_active_adapter(self, adapter_path: Optional[str] = None):
        """Sets active adapter path for inference."""
        self._active_adapter_path = adapter_path

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

    def generate_semantic_interpretation(
        self,
        user_prompt: str,
        asset_id: str,
        candidate_components: List[Dict[str, Any]],
        timeout: float = 120.0
    ) -> ComponentSemanticProposal:
        """
        Receives candidate geometric components and prompts Qwen to assign semantic labels,
        parent-child hierarchies, and confidence ratings.
        """
        candidates_summary = [
            {
                "component_id": c.get("component_id"),
                "relative_z_position": c.get("relative_z_position"),
                "dimensions": c.get("dimensions"),
                "volume_ratio": c.get("volume_ratio"),
                "face_count": c.get("face_count")
            }
            for c in candidate_components
        ]

        prompt = (
            f"<|im_start|>system\n{SEMANTIC_INTERPRETATION_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nAsset Request: \"{user_prompt}\"\nAsset ID: \"{asset_id}\"\n"
            f"Detected Geometric Candidate Components:\n{json.dumps(candidates_summary, indent=2)}\n"
            f"Output the JSON ComponentSemanticProposal:<|im_end|>\n<|im_start|>assistant\n"
        )

        raw_response = ""
        with gpu_broker.lease("Qwen_Semantic_Interpreter", timeout=timeout):
            try:
                self._load_model()
                output = self._llm(
                    prompt,
                    max_tokens=800,
                    temperature=0.1,
                    stop=["<|im_end|>", "<|endoftext|>"]
                )
                raw_response = output["choices"][0]["text"].strip()
            finally:
                self._unload_model()

        return self.parse_and_validate_semantic_proposal(raw_response)

    def parse_and_validate_semantic_proposal(self, text: str) -> ComponentSemanticProposal:
        clean_text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"```$", "", clean_text.strip(), flags=re.MULTILINE)

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract valid JSON from Qwen response: {text[:200]}")

        json_str = json_match.group(0)
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON from Qwen: {e}")

        return ComponentSemanticProposal.model_validate(data)

    def generate_precision_correction_plan(
        self,
        user_prompt: str,
        asset_name: str,
        geometry_report: Dict[str, Any],
        target_height_m: Optional[float] = None,
        timeout: float = 120.0
    ) -> PrecisionCorrectionPlan:
        """
        Generates a sequence of allowlisted tool calls to correct geometry, orientation, and dimensions.
        """
        geom_summary = {
            "dimensions": geometry_report.get("bounds", {}).get("dimensions"),
            "is_upright": geometry_report.get("orientation", {}).get("is_upright"),
            "estimated_up_axis": geometry_report.get("orientation", {}).get("estimated_up_axis"),
            "is_watertight": geometry_report.get("topology", {}).get("is_watertight"),
            "symmetry_plane": geometry_report.get("symmetry", {}).get("primary_plane"),
            "symmetry_score": geometry_report.get("symmetry", {}).get("deviation_score")
        }

        prompt = (
            f"<|im_start|>system\n{PRECISION_CORRECTION_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nAsset Request: \"{user_prompt}\"\nAsset Name: \"{asset_name}\"\n"
            f"Target Height (m): {target_height_m}\n"
            f"Measured Geometry Summary:\n{json.dumps(geom_summary, indent=2)}\n"
            f"Output the JSON PrecisionCorrectionPlan:<|im_end|>\n<|im_start|>assistant\n"
        )

        raw_response = ""
        with gpu_broker.lease("Qwen_Precision_Planner", timeout=timeout):
            try:
                self._load_model()
                output = self._llm(
                    prompt,
                    max_tokens=650,
                    temperature=0.1,
                    stop=["<|im_end|>", "<|endoftext|>"]
                )
                raw_response = output["choices"][0]["text"].strip()
            finally:
                self._unload_model()

        return self.parse_and_validate_correction_plan(raw_response)

    def parse_and_validate_correction_plan(self, text: str) -> PrecisionCorrectionPlan:
        clean_text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"```$", "", clean_text.strip(), flags=re.MULTILINE)

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract valid JSON from Qwen response: {text[:200]}")

        json_str = json_match.group(0)
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON from Qwen: {e}")

        return PrecisionCorrectionPlan.model_validate(data)



# Global planner instance

    def generate_component_edit_plan(
        self,
        user_instruction: str,
        hierarchy: Any,
        geometry_report: Optional[Any] = None,
        semantic_proposal: Optional[Any] = None,
        max_tokens: int = 512,
        require_neural: bool = False,
    ) -> ComponentEditPlan:
        """
        Generates a structured ComponentEditPlan using Qwen Coder or dynamic regex fallback.
        Supports arbitrary scale percentages (e.g. 23%, 12%), translation deltas,
        and preservation of non-target component invariants.
        If target component resolution is ambiguous, returns NEEDS_COMPONENT_CONFIRMATION.
        """
        candidate_nodes = {}
        for cid, node in hierarchy.components.items():
            if "root" not in cid.lower() and "assembly" not in cid.lower():
                candidate_nodes[cid] = node

        if not candidate_nodes:
            return ComponentEditPlan(
                asset_id=getattr(hierarchy, "root_asset_id", "asset_unknown"),
                user_request=user_instruction,
                status="REJECTED",
                rationale="Hierarchy contains no non-root candidate components."
            )

        # Build candidate summaries
        comp_summaries = []
        label_to_cid = {}
        if semantic_proposal and hasattr(semantic_proposal, "assignments"):
            for a in semantic_proposal.assignments:
                label_to_cid[a.assigned_semantic_label.lower()] = a.component_id

        for cid, node in candidate_nodes.items():
            assigned_label = getattr(node, "semantic_label", cid)
            for lbl, c_match in label_to_cid.items():
                if c_match == cid:
                    assigned_label = lbl
                    break
            dims = getattr(node, "dimensions", (0, 0, 0))
            comp_summaries.append(
                f"- ID '{cid}': Label '{assigned_label}', Dimensions [W={dims[0]:.2f}, D={dims[1]:.2f}, H={dims[2]:.2f}], Rel_Z={getattr(node, 'relative_z_position', 0.5):.2f}"
            )

        # Prompt Qwen Coder if model path exists
        from core.model_paths import model_path_resolver
        coder_path = model_path_resolver.resolve_qwen_coder_path()

        if coder_path and coder_path.exists():
            prompt = (
                f"<|im_start|>system\n"
                f"You are an expert AI 3D Technical Artist.\n"
                f"Analyze the user's natural language edit request and candidate components.\n"
                f"Extract target component IDs, operation, scale factor, axis, and invariants to preserve.\n"
                f"If the requested component cannot be identified with high confidence, set status='NEEDS_COMPONENT_CONFIRMATION'.\n"
                f"Output strictly valid JSON conforming to ComponentEditPlan schema.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"User Request: \"{user_instruction}\"\n"
                f"Candidate Components:\n" + "\n".join(comp_summaries) + "\n"
                f"Return JSON matching:\n"
                f"{{\n"
                f"  \"asset_id\": \"{getattr(hierarchy, 'root_asset_id', 'asset')}\",\n"
                f"  \"user_request\": \"{user_instruction}\",\n"
                f"  \"target_component_ids\": [\"...\"],\n"
                f"  \"target_semantic_labels\": [\"...\"],\n"
                f"  \"operation_type\": \"scale\",\n"
                f"  \"axis\": \"z\",\n"
                f"  \"scale_factor\": 1.2,\n"
                f"  \"anchor\": \"bottom\",\n"
                f"  \"invariants\": [{{\"component_id\": \"...\", \"preserve_dimensions\": true, \"preserve_position\": true}}],\n"
                f"  \"status\": \"READY\",\n"
                f"  \"rationale\": \"...\"\n"
                f"}}\n"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )

            try:
                with gpu_broker.lease("Qwen_Component_Planner", timeout=120.0):
                    try:
                        self._load_model()
                        if self._llm:
                            output = self._llm(
                                prompt,
                                max_tokens=max_tokens,
                                temperature=0.1,
                                stop=["<|im_end|>", "<|endoftext|>"]
                            )
                            raw_text = output["choices"][0]["text"].strip()
                            plan = self._parse_component_edit_plan(raw_text, hierarchy, user_instruction)
                            if plan:
                                plan.execution_provenance = "qwen_coder_neural"
                                return plan
                    finally:
                        self._unload_model()
            except Exception as e:
                if require_neural:
                    raise RuntimeError(f"Neural planning requested (require_neural=True) but failed: {e}")
                logger.warning(f"Qwen Coder inference failed ({e}), using dynamic semantic planner fallback.")
        elif require_neural:
            raise RuntimeError(f"Neural planning requested (require_neural=True) but Qwen Coder weights not found at {coder_path}")

        # Dynamic semantic planner fallback (supports arbitrary percentages & fuzzy label matching)
        plan = self._dynamic_rule_edit_plan(user_instruction, hierarchy, candidate_nodes, label_to_cid)
        plan.execution_provenance = "dynamic_rule_fallback"
        return plan

    def _parse_component_edit_plan(self, text: str, hierarchy: Any, instruction: str) -> Optional[ComponentEditPlan]:
        """Parses model JSON and validates component IDs exist in hierarchy."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            # Validate target IDs exist
            target_ids = data.get("target_component_ids", [])
            valid_targets = [cid for cid in target_ids if cid in hierarchy.components]
            
            candidates_list = [
                {"component_id": cid, "label": getattr(n, "semantic_label", cid), "dimensions": getattr(n, "dimensions", (0, 0, 0))}
                for cid, n in hierarchy.components.items() if "root" not in cid.lower()
            ]

            if not valid_targets and data.get("status") != "NEEDS_COMPONENT_CONFIRMATION":
                data["status"] = "NEEDS_COMPONENT_CONFIRMATION"
                data["requires_user_confirmation"] = True
                data["candidate_components"] = candidates_list
            else:
                data["target_component_ids"] = valid_targets

            # Reconcile operation_type and exact parameters from instruction
            inst_lower = instruction.lower()
            if any(w in inst_lower for w in ("separate", "split", "detach", "isolate", "individual objects")):
                data["operation_type"] = "separate_component"
                data["parameters"] = {}
                data["scale_factor"] = 1.0
            elif any(w in inst_lower for w in ("symmetrical", "symmetry", "mirror")):
                data["operation_type"] = "mirror"
                data["parameters"] = {"plane": "x", "duplicate": True}
                data["scale_factor"] = 1.0
            elif any(w in inst_lower for w in ("move", "translate", "backward", "forward", "upward", "downward")):
                data["operation_type"] = "translate"
                data["scale_factor"] = 1.0
                if "backward" in inst_lower:
                    data["translation_delta"] = [0.0, -0.1, 0.0]
                elif "forward" in inst_lower:
                    data["translation_delta"] = [0.0, 0.1, 0.0]
                elif "up" in inst_lower or "upward" in inst_lower:
                    data["translation_delta"] = [0.0, 0.0, 0.1]
                elif "down" in inst_lower or "downward" in inst_lower:
                    data["translation_delta"] = [0.0, 0.0, -0.1]
                data["parameters"] = {"translation": data.get("translation_delta", [0.0, 0.0, 0.0])}
            else:
                pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", instruction)
                if pct_match:
                    pct = float(pct_match.group(1))
                    is_decrease = any(w in inst_lower for w in ("shorter", "decrease", "smaller", "reduce", "lower", "less", "narrower", "thinner"))
                    data["scale_factor"] = round(1.0 - (pct / 100.0), 3) if is_decrease else round(1.0 + (pct / 100.0), 3)
                    if "width" in inst_lower or "wider" in inst_lower or "narrower" in inst_lower:
                        data["axis"] = "x"
                    elif "depth" in inst_lower or "thicker" in inst_lower or "thinner" in inst_lower:
                        data["axis"] = "y"
                    elif "taller" in inst_lower or "shorter" in inst_lower or "height" in inst_lower:
                        data["axis"] = "z"
                    if "parameters" not in data or not isinstance(data["parameters"], dict):
                        data["parameters"] = {}
                    data["parameters"]["factor"] = data["scale_factor"]
                    data["parameters"]["axis"] = data.get("axis", "z")

            # Semantic match verification: ensure user-specified noun matches target label
            action_clause = instruction.lower()
            for split_word in [" without ", " preserving ", " keeping ", " while preserving "]:
                if split_word in action_clause:
                    action_clause = action_clause.split(split_word)[0]
                    break

            known_nouns = [
                "armrest", "headrest", "backrest", "back", "seat", "cushion", "leg", "legs",
                "blade", "guard", "crossguard", "hilt", "pommel", "handle", "grip",
                "wheel", "tire", "wing", "spoilers", "propeller", "door", "drawer", "shelf"
            ]
            mentioned = [w for w in known_nouns if w in action_clause]
            if mentioned and valid_targets and data.get("status") != "NEEDS_COMPONENT_CONFIRMATION":
                target_matched = False
                for tid in valid_targets:
                    node = hierarchy.components.get(tid)
                    lbl = getattr(node, "semantic_label", "").lower() if node else ""
                    if any(w in lbl or lbl in w or w in tid.lower() for w in mentioned):
                        target_matched = True
                        break
                if not target_matched:
                    # User requested a specific component that doesn't match the selected component's label
                    data["status"] = "NEEDS_COMPONENT_CONFIRMATION"
                    data["requires_user_confirmation"] = True
                    data["target_component_ids"] = []
                    data["candidate_components"] = candidates_list
                    data["rationale"] = (
                        f"Requested component '{mentioned[0]}' does not match identified target components. "
                        f"Available candidates: {[c['label'] for c in candidates_list]}."
                    )

            if data.get("status") == "NEEDS_COMPONENT_CONFIRMATION" and not data.get("candidate_components"):
                data["candidate_components"] = candidates_list

            return ComponentEditPlan.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to parse ComponentEditPlan: {e}")
            return None

    def _dynamic_rule_edit_plan(
        self,
        user_instruction: str,
        hierarchy: Any,
        candidate_nodes: Dict[str, Any],
        label_to_cid: Dict[str, str]
    ) -> ComponentEditPlan:
        """Dynamic parser supporting arbitrary percentages (e.g. 23%, 12%), axes, and invariants."""
        inst_lower = user_instruction.lower()

        # 1. Parse arbitrary percentage: e.g. "23%", "12.5%", "50%"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", user_instruction)
        scale_pct = float(pct_match.group(1)) if pct_match else 20.0

        # Check direction: taller, increase, larger vs shorter, decrease, smaller
        is_decrease = any(w in inst_lower for w in ("shorter", "decrease", "smaller", "reduce", "lower", "less"))
        scale_factor = round(1.0 - (scale_pct / 100.0), 3) if is_decrease else round(1.0 + (scale_pct / 100.0), 3)

        # 2. Parse axis
        axis = "z"
        if "width" in inst_lower or "wider" in inst_lower or "narrower" in inst_lower or "x-axis" in inst_lower:
            axis = "x"
        elif "depth" in inst_lower or "thicker" in inst_lower or "thinner" in inst_lower or "y-axis" in inst_lower:
            axis = "y"
        elif "all" in inst_lower or "overall" in inst_lower or "uniform" in inst_lower:
            axis = "all"

        # 3. Match target component from action clause (ignore invariant clause)
        action_text = inst_lower
        for split_word in [" without ", " preserving ", " keeping ", " while preserving "]:
            if split_word in action_text:
                action_text = action_text.split(split_word)[0]
                break

        matched_cid = None
        matched_label = None

        # Check semantic labels from VLM
        for lbl, cid in label_to_cid.items():
            if lbl in action_text:
                matched_cid = cid
                matched_label = lbl
                break

        # Check candidate node labels
        if not matched_cid:
            for cid, node in candidate_nodes.items():
                lbl = getattr(node, "semantic_label", "").lower()
                if lbl and lbl in action_text:
                    matched_cid = cid
                    matched_label = lbl
                    break

        # Check common component keywords in instruction
        if not matched_cid:
            keywords = ["backrest", "back", "seat", "cushion", "leg", "legs", "blade", "grip", "handle", "crossguard", "pommel", "armrest", "frame", "wheel", "wheels", "tire", "tires"]
            for kw in keywords:
                if kw in action_text:
                    # Look for candidate containing this keyword
                    for cid, node in candidate_nodes.items():
                        lbl = getattr(node, "semantic_label", "").lower()
                        if kw in lbl or kw in cid.lower():
                            matched_cid = cid
                            matched_label = kw
                            break
                    if matched_cid:
                        break

        # Spatial relative elevation matching (e.g. "upper part", "top section", "lower legs")
        if not matched_cid:
            if any(w in action_text for w in ("upper", "top", "roof", "head")):
                sorted_by_z = sorted(candidate_nodes.items(), key=lambda kv: getattr(kv[1], "relative_z_position", 0.5), reverse=True)
                if sorted_by_z:
                    matched_cid = sorted_by_z[0][0]
                    matched_label = getattr(sorted_by_z[0][1], "semantic_label", "upper_structure")
            elif any(w in action_text for w in ("lower", "bottom", "base")):
                sorted_by_z = sorted(candidate_nodes.items(), key=lambda kv: getattr(kv[1], "relative_z_position", 0.5))
                if sorted_by_z:
                    matched_cid = sorted_by_z[0][0]
                    matched_label = getattr(sorted_by_z[0][1], "semantic_label", "lower_structure")

        # CRITICAL SAFETY RULE: If target cannot be resolved, DO NOT guess or pick first component!
        if not matched_cid:
            candidates = [
                {"component_id": cid, "label": getattr(n, "semantic_label", cid), "dimensions": getattr(n, "dimensions", (0, 0, 0))}
                for cid, n in candidate_nodes.items()
            ]
            return ComponentEditPlan(
                asset_id=getattr(hierarchy, "root_asset_id", "asset"),
                user_request=user_instruction,
                target_component_ids=[],
                target_semantic_labels=[],
                operation_type="scale",
                status="NEEDS_COMPONENT_CONFIRMATION",
                requires_user_confirmation=True,
                candidate_components=candidates,
                rationale=f"Cannot uniquely identify target component for instruction '{user_instruction}'. Awaiting user confirmation."
            )

        # 4. Invariants: All OTHER non-target candidate components must remain preserved!
        invariants = []
        for cid in candidate_nodes.keys():
            if cid != matched_cid:
                invariants.append(ComponentInvariant(
                    component_id=cid,
                    preserve_dimensions=True,
                    preserve_position=True,
                    tolerance=0.05
                ))

        # Operation detection
        op_type = "scale"
        params = {"axis": axis, "factor": scale_factor, "anchor": "bottom"}
        trans_delta = None

        if any(w in inst_lower for w in ("separate", "split", "detach", "isolate", "individual objects")):
            op_type = "separate_component"
            params = {}
        elif any(w in inst_lower for w in ("symmetrical", "symmetry", "mirror")):
            op_type = "mirror"
            params = {"plane": "x", "duplicate": True}
        elif any(w in inst_lower for w in ("move", "translate", "backward", "forward", "upward", "downward")):
            op_type = "translate"
            if "backward" in inst_lower:
                trans_delta = [0.0, -0.1, 0.0]
            elif "forward" in inst_lower:
                trans_delta = [0.0, 0.1, 0.0]
            elif "up" in inst_lower or "upward" in inst_lower:
                trans_delta = [0.0, 0.0, 0.1]
            elif "down" in inst_lower or "downward" in inst_lower:
                trans_delta = [0.0, 0.0, -0.1]
            else:
                trans_delta = [0.0, 0.0, 0.0]
            params = {"translation": trans_delta}
        elif any(w in inst_lower for w in ("bevel", "chamfer")):
            op_type = "bevel"
            params = {"width": 0.01}
        elif any(w in inst_lower for w in ("solidify", "thicken")):
            op_type = "solidify"
            params = {"thickness": 0.02}
        elif any(w in inst_lower for w in ("cleanup", "clean geometry")):
            op_type = "cleanup"
            params = {}

        return ComponentEditPlan(
            asset_id=getattr(hierarchy, "root_asset_id", "asset"),
            user_request=user_instruction,
            target_component_ids=[matched_cid],
            target_semantic_labels=[matched_label or "component"],
            operation_type=op_type,
            parameters=params,
            axis=axis,
            scale_factor=scale_factor if op_type == "scale" else 1.0,
            anchor="bottom",
            translation_delta=trans_delta,
            invariants=invariants,
            status="READY",
            requires_user_confirmation=False,
            rationale=f"Execute {op_type} on {matched_label} ({matched_cid}) while preserving {len(invariants)} non-target component invariants."
        )


qwen_planner = QwenPlanner()
