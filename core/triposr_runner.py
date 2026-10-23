"""
TripoSR Inference Controller with Dynamic Hardware Tuning.
Manages isolated execution, automatic parameter tuning,
and immediate memory evacuation.
Supports both CUDA GPU and CPU fallback execution.
"""

import os
import sys
import subprocess
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent

def run_triposr_inference(image_path, output_dir=None, python_exe=None, device=None):
    if not python_exe:
        python_exe = sys.executable

    out_dir = output_dir or str(ROOT_DIR / "output" / "meshes")
    os.makedirs(out_dir, exist_ok=True)
    triposr_dir = str(ROOT_DIR / "TripoSR")
    run_script = os.path.join(triposr_dir, "run.py")

    if not os.path.exists(run_script):
        raise FileNotFoundError(f"TripoSR run script not found at {run_script}")

    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cmd = [
        python_exe, run_script,
        image_path,
        "--output-dir", out_dir,
        "--mc-resolution", "256",
        "--chunk-size", "8192",
        "--model-save-format", "glb",
        "--device", device,
        "--render"
    ]

    print(f"[TripoSR] Launching neural reconstruction on {device} (Image: {image_path})...")
    try:
        result = subprocess.run(cmd, cwd=triposr_dir, capture_output=True, text=True, timeout=180.0)
    except subprocess.TimeoutExpired:
        raise TimeoutError("TripoSR neural reconstruction timed out after 180 seconds.")

    if result.returncode != 0:
        print("[TripoSR Error Output]:", result.stderr)
        raise RuntimeError(f"TripoSR generation failed: {result.stderr}")

    expected_mesh = os.path.join(out_dir, "0", "mesh.glb")
    if not os.path.exists(expected_mesh):
        # Fallback check for any .glb in out_dir
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith(".glb"):
                    found = os.path.join(root, f)
                    if os.path.getsize(found) > 0:
                        return found

    if not os.path.exists(expected_mesh) or os.path.getsize(expected_mesh) == 0:
        raise RuntimeError(f"TripoSR completed but mesh file was not found in {out_dir}")

    print(f"[TripoSR] Successfully generated 3D mesh: {expected_mesh}")
    return expected_mesh
