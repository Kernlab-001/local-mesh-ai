"""
Export Service for GLB, FBX, and OBJ.
Executes conversion using Blender's native export operators
with strict path validation, argument safety, and export verification.
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

ALLOWED_FORMATS = {"glb", "fbx", "obj"}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ExportService:
    def __init__(self, blender_path: Optional[str] = None, export_dir: Optional[str] = None):
        if not blender_path:
            try:
                from core.hardware_detector import hardware_detector
                blender_path = hardware_detector.find_blender()
            except Exception:
                pass
        self.blender_path = blender_path or "blender"
        self.export_dir = os.path.abspath(export_dir or os.path.join(PROJECT_ROOT, "output", "exports"))
        os.makedirs(self.export_dir, exist_ok=True)

    def sanitize_filename(self, name: str) -> str:
        """Sanitizes user-provided output filename to prevent path traversal and shell injection."""
        base = os.path.basename(name)
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", base)
        return clean or "model_export"

    def export_asset(self, blend_path: str, format_type: str, output_name: str = "model") -> Dict[str, Any]:
        fmt = format_type.lower().strip(".")
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format '{format_type}'. Allowed: {ALLOWED_FORMATS}")

        resolved_blend = os.path.abspath(blend_path)
        if not os.path.exists(resolved_blend):
            raise FileNotFoundError(f"Source blend file not found: {resolved_blend}")

        clean_name = self.sanitize_filename(output_name)
        out_filename = f"{clean_name}.{fmt}"
        target_path = os.path.abspath(os.path.join(self.export_dir, out_filename))

        # Enforce path traversal boundary
        if not Path(target_path).resolve().is_relative_to(Path(self.export_dir).resolve()):
            raise PermissionError("Export target path violates security boundary.")

        # Safe script using sys.argv rather than string interpolation
        blender_script = """
import bpy
import sys

args = []
if "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]

if len(args) < 3:
    sys.exit(1)

blend_file = args[0]
out_file = args[1]
fmt = args[2].lower()

bpy.ops.wm.open_mainfile(filepath=blend_file)
mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
if not mesh_objs:
    sys.exit(2)

if fmt == "glb":
    bpy.ops.export_scene.gltf(filepath=out_file, export_format='GLB')
elif fmt == "fbx":
    bpy.ops.export_scene.fbx(filepath=out_file)
elif fmt == "obj":
    try:
        bpy.ops.wm.obj_export(filepath=out_file)
    except Exception:
        bpy.ops.export_scene.obj(filepath=out_file)
"""

        cmd = [
            self.blender_path,
            "--background",
            "--python-expr", blender_script,
            "--",
            resolved_blend,
            target_path,
            fmt
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Blender export timed out after 180 seconds for format '{fmt}'.") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"Blender export failed for format '{fmt}':\n{proc.stderr}")

        if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
            raise RuntimeError(f"Export file not created or empty: {target_path}")

        return {
            "status": "success",
            "format": fmt,
            "export_path": target_path,
            "size_bytes": os.path.getsize(target_path)
        }

export_service = ExportService()
