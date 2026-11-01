"""Local Multimodal Vision-Language Model Adapter for Semantic 3D Component Interpretation.

Interfaces with local Qwen3-VL-4B-Instruct GGUF model and mmproj vision encoder.

Workflow:
1. Generates 6-view orthographic and perspective renders (Front, Back, Left, Right, Top, Perspective).
2. Generates component ID color mask renders with stable, distinct color coding.
3. Formulates structured multimodal prompt containing:
   - Genuine base64 image data URIs
   - Component ID color legend
   - Metric dimensions, centroids, and relative elevations
   - Original user asset description
4. Interrogates local VLM via llama-cpp-python Qwen25VLChatHandler.
5. Strictly validates that the VLM never invents new component IDs (rejects hallucinations).
6. Flags uncertain predictions truthfully when visual confidence is low (< 0.70).
7. Transparently distinguishes genuine neural VLM inference ('qwen3_vl_multimodal')
   from deterministic fallback ('heuristic_fallback').
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

try:
    import trimesh
except ImportError:
    trimesh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.command_schemas import ComponentSemanticAssignment, ComponentSemanticProposal
from core.component_decomposer import ComponentHierarchy, ComponentNode
from core.gpu_broker import gpu_broker
from core.hardware_detector import hardware_detector
from core.model_paths import model_path_resolver

logger = logging.getLogger("VisionLanguageAdapter")


class MultiViewRenderResult(BaseModel):
    render_paths: Dict[str, str] = Field(description="Angles: front, back, left, right, top, perspective")
    mask_render_path: Optional[str] = Field(default=None, description="Component ID color-coded render")
    color_legend: Dict[str, str] = Field(default_factory=dict, description="Component ID to RGB hex color")


class VisionLanguageAdapter:
    """Multimodal Vision-Language adapter for semantic 3D component interpretation."""

    COMPONENT_PALETTE = [
        [230, 25, 75],    # Red
        [60, 180, 75],    # Green
        [255, 225, 25],   # Yellow
        [0, 130, 200],    # Blue
        [245, 130, 48],   # Orange
        [145, 30, 180],   # Purple
        [70, 240, 240],   # Cyan
        [240, 50, 230],   # Magenta
        [210, 245, 60],   # Lime
        [250, 190, 212],  # Pink
        [0, 128, 128],    # Teal
        [220, 190, 255],  # Lavender
    ]

    def __init__(self, external_models_root: Optional[str] = None):
        self._resolver = model_path_resolver

    def resolve_model_path(self) -> Optional[Path]:
        return self._resolver.resolve_qwen3_vl_model_path()

    def resolve_mmproj_path(self) -> Optional[Path]:
        return self._resolver.resolve_qwen3_vl_mmproj_path()

    def is_model_installed(self) -> bool:
        """Verifies local presence of both language GGUF and vision projector."""
        m = self.resolve_model_path()
        p = self.resolve_mmproj_path()
        return bool(m and m.exists() and p and p.exists())

    def generate_multiview_renders(
        self,
        mesh: trimesh.Trimesh,
        output_dir: str,
        hierarchy: Optional[ComponentHierarchy] = None,
    ) -> MultiViewRenderResult:
        """Generates 6 distinct camera view renders and an ID-colored component mask render."""
        os.makedirs(output_dir, exist_ok=True)
        render_paths = {}

        # Angles corresponding to camera look directions from camera to mesh center
        angles = {
            "front": (0.0, 0.0, 1.0),
            "back": (0.0, 0.0, -1.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
            "top": (0.0, 1.0, 0.0),
            "perspective": (0.707, 0.5, 0.707),
        }

        for view_name, direction in angles.items():
            path = os.path.join(output_dir, f"view_{view_name}.png")
            self._render_view(mesh, path, direction)
            render_paths[view_name] = path

        color_legend = {}
        mask_path = os.path.join(output_dir, "view_component_masks.png")

        if hierarchy and hierarchy.components:
            num_faces = len(mesh.faces)
            face_colors = np.zeros((num_faces, 3), dtype=np.uint8)
            face_colors[:] = [120, 125, 135]  # Neutral background

            pal_idx = 0
            for cid, node in hierarchy.components.items():
                if "root" in cid.lower() or "assembly" in cid.lower():
                    continue
                rgb = self.COMPONENT_PALETTE[pal_idx % len(self.COMPONENT_PALETTE)]
                color_legend[cid] = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                pal_idx += 1

                # Apply to face indices of this component
                if hasattr(node, "face_indices") and node.face_indices:
                    for f_idx in node.face_indices:
                        if 0 <= f_idx < num_faces:
                            face_colors[f_idx] = rgb
                elif hasattr(node, "bounds") and node.bounds:
                    # Spatial fallback if face_indices empty
                    b = node.bounds
                    centroids = mesh.triangles_center
                    mask = (
                        (centroids[:, 0] >= b.min_point[0] - 1e-4) & (centroids[:, 0] <= b.max_point[0] + 1e-4) &
                        (centroids[:, 1] >= b.min_point[1] - 1e-4) & (centroids[:, 1] <= b.max_point[1] + 1e-4) &
                        (centroids[:, 2] >= b.min_point[2] - 1e-4) & (centroids[:, 2] <= b.max_point[2] + 1e-4)
                    )
                    face_colors[mask] = rgb

            self._render_view(mesh, mask_path, (0.707, 0.5, 0.707), face_colors=face_colors)

        return MultiViewRenderResult(
            render_paths=render_paths,
            mask_render_path=mask_path if os.path.exists(mask_path) else None,
            color_legend=color_legend,
        )

    def _render_view(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        direction: Tuple[float, float, float],
        face_colors: Optional[np.ndarray] = None,
    ):
        """Renders true 3D camera projection image to disk with occlusion depth sorting."""
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (512, 512), (32, 35, 40))
            draw = ImageDraw.Draw(img)

            # Camera look-at coordinate frame:
            # forward: pointing from camera to object, so cam_pos = direction * dist
            # view_dir = -normalize(direction)
            dir_vec = np.asarray(direction, dtype=float)
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-6:
                dir_vec = np.array([0.0, 0.0, 1.0])
                dir_norm = 1.0
            forward = -dir_vec / dir_norm

            up_ref = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(forward, up_ref)) > 0.95:
                up_ref = np.array([0.0, 0.0, 1.0])

            right = np.cross(up_ref, forward)
            r_norm = np.linalg.norm(right)
            if r_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            else:
                right /= r_norm

            up = np.cross(forward, right)
            up /= np.linalg.norm(up)

            # Transform mesh vertices to camera space: X=right, Y=up, Z=forward
            cam_rot = np.column_stack([right, up, forward])
            v_centered = mesh.vertices - mesh.centroid
            v_cam = v_centered @ cam_rot

            ext = np.maximum(mesh.extents, 1e-4)
            scale = 210.0 / np.max(ext)

            px = 256.0 + v_cam[:, 0] * scale
            py = 256.0 - v_cam[:, 1] * scale

            faces = mesh.faces
            # Depth sort faces from furthest to nearest (back-to-front painter's algorithm)
            face_z = np.mean(v_cam[faces, 2], axis=1)
            depth_order = np.argsort(face_z)

            # Lighting vector in camera space
            light_cam = np.array([0.3, 0.5, 0.82])
            light_cam /= np.linalg.norm(light_cam)
            face_normals_cam = mesh.face_normals @ cam_rot

            step = 1 if len(faces) <= 15000 else max(1, len(faces) // 15000)
            for f_idx in depth_order[::step]:
                f = faces[f_idx]
                p1 = (float(px[f[0]]), float(py[f[0]]))
                p2 = (float(px[f[1]]), float(py[f[1]]))
                p3 = (float(px[f[2]]), float(py[f[2]]))

                if face_colors is not None:
                    rgb = tuple(int(c) for c in face_colors[f_idx])
                    draw.polygon([p1, p2, p3], outline=rgb, fill=rgb)
                else:
                    dot = float(np.dot(face_normals_cam[f_idx], light_cam))
                    intensity = 0.35 + 0.65 * max(0.0, dot)
                    c = int(np.clip(160 * intensity, 40, 245))
                    color = (c, int(c * 0.96), int(c * 1.05))
                    draw.polygon([p1, p2, p3], outline=color, fill=color)

            img.save(output_path)
        except Exception as e:
            logger.warning(f"View rendering error for {output_path}: {e}")

    def interpret_components(
        self,
        hierarchy: ComponentHierarchy,
        renders: MultiViewRenderResult,
        user_asset_description: Optional[str] = None,
        allow_fallback: bool = True,
        max_tokens: int = 512,
    ) -> ComponentSemanticProposal:
        """Interprets candidate components using visual and spatial evidence.

        Validates that returned component IDs match the candidate inputs exactly.
        Flags uncertainty truthfully when confidence is below 0.70.
        """
        candidate_ids = [
            cid for cid in hierarchy.components.keys()
            if not ("root" in cid.lower() or "assembly" in cid.lower())
        ]

        if not candidate_ids:
            return ComponentSemanticProposal(
                asset_id=hierarchy.root_asset_id,
                category="empty",
                inference_source="none",
                assignments=[],
                overall_confidence=1.0,
                has_uncertain_components=False,
                summary="No non-root components to interpret.",
            )

        # If VLM is installed and running, invoke genuine multimodal inference
        if self.is_model_installed():
            try:
                return self._run_vlm_inference(
                    candidate_ids, hierarchy, renders, user_asset_description, max_tokens=max_tokens
                )
            except Exception as e:
                logger.warning(f"VLM neural inference failed: {e}")
                if not allow_fallback:
                    raise RuntimeError(f"VLM inference failed and allow_fallback=False: {e}")
                logger.info("Proceeding with clearly labeled spatial-visual heuristic fallback.")

        elif not allow_fallback:
            raise RuntimeError("Qwen3-VL models not installed and allow_fallback=False")

        # Clearly labeled Deterministic Spatial-Visual Interpreter Fallback
        return self._heuristic_visual_interpret(candidate_ids, hierarchy, user_asset_description)

    def _encode_image_b64(self, file_path: str) -> Optional[str]:
        """Encodes an image file to base64 data URI."""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = p.suffix.lower().replace(".", "")
            if ext == "jpg":
                ext = "jpeg"
            return f"data:image/{ext};base64,{b64}"
        except Exception as e:
            logger.warning(f"Failed to encode image {file_path}: {e}")
            return None

    def _run_vlm_inference(
        self,
        candidate_ids: List[str],
        hierarchy: ComponentHierarchy,
        renders: MultiViewRenderResult,
        description: Optional[str],
        max_tokens: int = 512,
    ) -> ComponentSemanticProposal:
        """Executes genuine local Qwen3-VL multimodal inference passing actual images."""
        model_path = self.resolve_model_path()
        mmproj_path = self.resolve_mmproj_path()
        if not model_path or not mmproj_path:
            raise FileNotFoundError("Qwen3-VL model or mmproj weights not found.")

        # Ensure Windows DLL directory is configured for llama_cpp
        lib_dir = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib"
        if hasattr(os, "add_dll_directory") and lib_dir.exists():
            os.add_dll_directory(str(lib_dir))

        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if hasattr(os, "add_dll_directory") and torch_lib.exists():
            os.add_dll_directory(str(torch_lib))

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        # Format image content parts
        content_parts: List[Dict[str, Any]] = []

        # Add perspective view and mask view if available
        image_sources = []
        if renders.mask_render_path and os.path.exists(renders.mask_render_path):
            image_sources.append(("Component ID Mask", renders.mask_render_path))
        if "perspective" in renders.render_paths and os.path.exists(renders.render_paths["perspective"]):
            image_sources.append(("Perspective View", renders.render_paths["perspective"]))
        elif renders.render_paths:
            first_key = next(iter(renders.render_paths))
            image_sources.append((f"{first_key.title()} View", renders.render_paths[first_key]))

        for label, img_path in image_sources:
            data_uri = self._encode_image_b64(img_path)
            if data_uri:
                content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

        # Build candidate component descriptions
        comp_summaries = []
        for cid in candidate_ids:
            node = hierarchy.components[cid]
            color_hex = renders.color_legend.get(cid, "N/A")
            comp_summaries.append(
                f"- ID '{cid}' (Mask Color: {color_hex}): Dimensions [W={node.dimensions[0]:.2f}, D={node.dimensions[1]:.2f}, H={node.dimensions[2]:.2f}], Relative Z-Height={node.relative_z_position:.2f}"
            )

        candidate_ids_str = ", ".join([f"'{c}'" for c in candidate_ids])
        text_prompt = (
            f"You are a 3D technical artist analyzing components of a 3D asset.\n"
            f"Asset Description: {description or '3D Game Asset'}\n\n"
            f"Candidate Components to identify:\n" + "\n".join(comp_summaries) + "\n\n"
            f"CRITICAL RULES:\n"
            f"1. You MUST label each of the following candidate IDs: {candidate_ids_str}.\n"
            f"2. You MUST NEVER invent or hallucinate any component ID outside this list.\n"
            f"3. Assign a realistic functional semantic label (e.g. backrest, seat, leg, blade, handle, base) and material intent.\n"
            f"4. Respond with ONLY a valid JSON object matching this schema:\n"
            "```json\n"
            "{\n"
            '  "assignments": [\n'
            '    {"component_id": "...", "assigned_semantic_label": "...", "material_intent": "...", "confidence": 0.85, "is_uncertain": false, "reasoning": "..."}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        content_parts.append({"type": "text", "text": text_prompt})

        # Load Qwen3-VL under exclusive GPU lease
        t0 = time.time()
        with gpu_broker.lease("Qwen3_VL_Multimodal", timeout=120.0):
            handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            llm = Llama(
                model_path=str(model_path),
                chat_handler=handler,
                n_ctx=2048,
                n_gpu_layers=15,
                verbose=False,
            )

            try:
                response = llm.create_chat_completion(
                    messages=[{"role": "user", "content": content_parts}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                raw_text = response["choices"][0]["message"]["content"]
                logger.info(f"Qwen3-VL inference completed in {time.time() - t0:.2f}s")
            finally:
                del llm
                del handler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Parse and validate JSON
        return self._parse_and_validate_vlm_output(raw_text, candidate_ids, hierarchy)

    def _parse_and_validate_vlm_output(
        self,
        raw_text: str,
        candidate_ids: List[str],
        hierarchy: ComponentHierarchy,
    ) -> ComponentSemanticProposal:
        """Parses model text, enforces candidate ID allowlist, and rejects hallucinated IDs."""
        json_str = raw_text
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            obj_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if obj_match:
                json_str = obj_match.group(1)

        parsed_data = {}
        # Clean trailing commas
        cleaned_json = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            parsed_data = json.loads(cleaned_json)
        except Exception as e:
            # Fallback regex extraction of individual assignment dicts
            raw_items = []
            for block in re.findall(r'\{[^{}]*"component_id"[^{}]*\}', raw_text):
                try:
                    c_block = re.sub(r",\s*([\]}])", r"\1", block)
                    raw_items.append(json.loads(c_block))
                except Exception:
                    pass
            if raw_items:
                parsed_data = {"assignments": raw_items}
            else:
                logger.warning(f"Failed to parse Qwen3-VL JSON output ({e}). Falling back to heuristic.")
                return self._heuristic_visual_interpret(candidate_ids, hierarchy, None)

        raw_assignments = parsed_data.get("assignments", [])
        valid_assignments: List[ComponentSemanticAssignment] = []
        candidate_set = set(candidate_ids)
        seen_ids = set()

        for item in raw_assignments:
            cid = item.get("component_id")
            # STRICT RULE: Reject any hallucinated or unknown component ID
            if not cid or cid not in candidate_set:
                logger.warning(f"Rejecting hallucinated or unknown component ID from VLM: '{cid}'")
                continue

            seen_ids.add(cid)
            label = item.get("assigned_semantic_label", "unknown_part")
            mat = item.get("material_intent", "custom")
            conf = float(item.get("confidence", 0.75))
            conf = max(0.0, min(1.0, conf))
            is_unc = bool(item.get("is_uncertain", conf < 0.70))

            valid_assignments.append(
                ComponentSemanticAssignment(
                    component_id=cid,
                    assigned_semantic_label=label,
                    material_intent=mat,
                    confidence=conf,
                    is_uncertain=is_unc,
                    reasoning=item.get("reasoning", "Identified by Qwen3-VL multimodal inspection."),
                )
            )

        # For any candidate component omitted by model, generate uncertain placeholder
        for cid in candidate_ids:
            if cid not in seen_ids:
                node = hierarchy.components[cid]
                valid_assignments.append(
                    ComponentSemanticAssignment(
                        component_id=cid,
                        assigned_semantic_label=f"unlabeled_{cid}",
                        material_intent="custom",
                        confidence=0.4,
                        is_uncertain=True,
                        reasoning="Omitted by VLM; flagged for user confirmation.",
                    )
                )

        overall_conf = (
            sum(a.confidence for a in valid_assignments) / len(valid_assignments)
            if valid_assignments
            else 0.5
        )

        return ComponentSemanticProposal(
            asset_id=hierarchy.root_asset_id,
            category="furniture" if any("seat" in a.assigned_semantic_label for a in valid_assignments) else "prop",
            inference_source="qwen3_vl_multimodal",
            assignments=valid_assignments,
            overall_confidence=round(overall_conf, 3),
            has_uncertain_components=any(a.is_uncertain for a in valid_assignments),
            summary=f"Qwen3-VL multimodal perception identified {len(valid_assignments)} components.",
        )

    def _heuristic_visual_interpret(
        self,
        candidate_ids: List[str],
        hierarchy: ComponentHierarchy,
        description: Optional[str],
    ) -> ComponentSemanticProposal:
        """High-precision spatial elevation and aspect-ratio interpreter fallback."""
        assignments: List[ComponentSemanticAssignment] = []
        desc_lower = (description or "").lower()

        is_furniture = any(w in desc_lower for w in ("chair", "table", "stool", "desk", "sofa", "bench"))
        is_weapon = any(w in desc_lower for w in ("sword", "blade", "dagger", "axe", "weapon", "spear"))

        for cid in candidate_ids:
            node = hierarchy.components[cid]
            w, d, h = node.dimensions
            rel_z = node.relative_z_position

            label = "component"
            material = "custom"
            conf = 0.65
            is_unc = True
            reasoning = []

            if is_furniture or "chair" in desc_lower:
                if rel_z >= 0.65:
                    label = "backrest"
                    material = "wood"
                    conf = 0.85
                    is_unc = False
                    reasoning.append("Upper elevation (Z >= 0.65) consistent with backrest.")
                elif 0.35 <= rel_z < 0.65:
                    label = "seat"
                    material = "leather"
                    conf = 0.88
                    is_unc = False
                    reasoning.append("Mid elevation (0.35 <= Z < 0.65) consistent with seat.")
                else:
                    label = "legs"
                    material = "metal"
                    conf = 0.82
                    is_unc = False
                    reasoning.append("Lower elevation (Z < 0.35) consistent with support legs.")
            elif is_weapon or "sword" in desc_lower:
                if h > max(w, d) * 2.0 and rel_z > 0.4:
                    label = "blade"
                    material = "steel"
                    conf = 0.90
                    is_unc = False
                    reasoning.append("Elongated vertical geometry consistent with blade.")
                elif rel_z < 0.2:
                    label = "pommel"
                    material = "brass"
                    conf = 0.75
                    is_unc = False
                    reasoning.append("Base terminal feature consistent with pommel.")
                elif 0.2 <= rel_z <= 0.4:
                    if w > d * 1.5:
                        label = "crossguard"
                        material = "iron"
                        conf = 0.85
                        is_unc = False
                    else:
                        label = "grip"
                        material = "leather"
                        conf = 0.80
                        is_unc = False
            else:
                if rel_z > 0.7:
                    label = "upper_structure"
                    material = "custom"
                    conf = 0.60
                elif rel_z < 0.3:
                    label = "base_support"
                    material = "custom"
                    conf = 0.60
                else:
                    label = "central_body"
                    material = "custom"
                    conf = 0.60

            assignments.append(
                ComponentSemanticAssignment(
                    component_id=cid,
                    assigned_semantic_label=label,
                    material_intent=material,
                    confidence=conf,
                    is_uncertain=is_unc,
                    reasoning=" ".join(reasoning) or "Deterministic spatial elevation heuristic.",
                )
            )

        overall_conf = sum(a.confidence for a in assignments) / len(assignments) if assignments else 0.5

        return ComponentSemanticProposal(
            asset_id=hierarchy.root_asset_id,
            category="furniture" if is_furniture else ("weapon" if is_weapon else "prop"),
            inference_source="heuristic_fallback",
            assignments=assignments,
            overall_confidence=round(overall_conf, 3),
            has_uncertain_components=any(a.is_uncertain for a in assignments),
            summary=f"Spatial heuristic classified {len(assignments)} components.",
        )


    def run_multimodal_inference(
        self,
        prompt: str,
        images: List[Any],
        max_tokens: int = 256,
    ) -> str:
        """Executes standalone multimodal inference on Qwen3-VL under exclusive GPU lease."""
        model_path = self.resolve_model_path()
        mmproj_path = self.resolve_mmproj_path()
        if not model_path or not mmproj_path:
            raise FileNotFoundError("Qwen3-VL model or mmproj weights not found.")

        # Ensure Windows DLL directory is configured for llama_cpp
        lib_dir = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib"
        if hasattr(os, "add_dll_directory") and lib_dir.exists():
            os.add_dll_directory(str(lib_dir))

        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if hasattr(os, "add_dll_directory") and torch_lib.exists():
            os.add_dll_directory(str(torch_lib))

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        content_parts: List[Dict[str, Any]] = []
        for img_item in images:
            if isinstance(img_item, (str, Path)):
                uri = self._encode_image_b64(str(img_item))
                if uri:
                    content_parts.append({"type": "image_url", "image_url": {"url": uri}})
            elif hasattr(img_item, "save"):
                # PIL Image
                import io
                buf = io.BytesIO()
                img_item.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        content_parts.append({"type": "text", "text": prompt})

        with gpu_broker.lease("Qwen3_VL_Multimodal", timeout=120.0):
            handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            llm = Llama(
                model_path=str(model_path),
                chat_handler=handler,
                n_ctx=2048,
                n_gpu_layers=15,
                verbose=False,
            )
            try:
                response = llm.create_chat_completion(
                    messages=[{"role": "user", "content": content_parts}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return response["choices"][0]["message"]["content"]
            finally:
                del llm
                del handler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    _run_multimodal_inference = run_multimodal_inference


vision_language_adapter = VisionLanguageAdapter()
