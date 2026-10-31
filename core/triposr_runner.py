"""
TripoSR Inference Controller with Dynamic Hardware Tuning.
Wraps the hardened TripoSREngine to provide backward-compatible API
with strict per-job output isolation, dynamic hardware tuning,
and immediate memory evacuation.
Supports both CUDA GPU and CPU fallback execution.
"""

import os
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.triposr_engine import triposr_engine


def run_triposr_inference(
    image_path: str,
    output_dir: Optional[str] = None,
    python_exe: Optional[str] = None,
    device: Optional[str] = None,
    mc_resolution: int = 256,
    chunk_size: int = 8192,
    render: bool = False
) -> str:
    """
    Executes neural 3D reconstruction using hardened TripoSREngine.
    Enforces isolated job output directory and returns the absolute path to the generated GLB mesh.
    """
    res = triposr_engine.reconstruct(
        image_path=image_path,
        mc_resolution=mc_resolution,
        chunk_size=chunk_size,
        render_turntable=render,
        device=device,
        output_dir=output_dir
    )
    return res["mesh_path"]
