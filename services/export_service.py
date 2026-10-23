"""
Export Service for GLB, FBX, and OBJ.
Executes conversion using Blender's native export operators
with strict path validation.
"""

import os
import sys
import subprocess
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

    def export_asset(self, blend_path: str, format_type: str, output_name: str = "model") -> Dict[str, Any]:
        fmt = format_type.lower().strip(".")
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format '{format_type}'. Allowed: {ALLOWED_FORMATS}")

        if not os.path.exists(blend_path):
            raise FileNotFoundError(f"Source blend file not found: {blend_path}")

        out_filename = f"{output_name}.{fmt}"
        target_path = os.path.join(self.export_dir, out_filename)

        # Blender Python script for export
        blender_script = f"""
import bpy
import sys

bpy.ops.wm.open_mainfile(filepath=r"{blend_path}")
mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
if not mesh_objs:
    sys.exit(1)

out_file = r"{target_path}"
fmt = "{fmt}"

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
            "--python-expr", blender_script
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Blender export failed for format '{fmt}':\n{proc.stderr}")

        if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
            raise RuntimeError(f"Export file not created or empty: {target_path}")

        return {
            "status": "success",
            "format": fmt,
            "export_path": os.path.abspath(target_path),
            "size_bytes": os.path.getsize(target_path)
        }

export_service = ExportService()
