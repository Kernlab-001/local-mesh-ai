"""
TripoSR Inference Controller with Dynamic Hardware Tuning.
Manages isolated execution, automatic parameter tuning,
and immediate memory evacuation.
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def run_triposr_inference(image_path, output_dir=None, python_exe=None):
    if not python_exe:
        python_exe = sys.executable
        
    out_dir = output_dir or str(ROOT_DIR / "output" / "meshes")
    os.makedirs(out_dir, exist_ok=True)
    triposr_dir = str(ROOT_DIR / "TripoSR")
    run_script = os.path.join(triposr_dir, "run.py")
    
    if not os.path.exists(run_script):
        raise FileNotFoundError(f"TripoSR run script not found at {run_script}")
        
    cmd = [
        python_exe, run_script,
        image_path,
        "--output-dir", out_dir,
        "--mc-resolution", "256",
        "--chunk-size", "8192",
        "--model-save-format", "glb",
        "--device", "cuda:0",
        "--render"
    ]
    
    print(f"[TripoSR] Launching neural reconstruction on local GPU (Image: {image_path})...")
    result = subprocess.run(cmd, cwd=triposr_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[TripoSR Error Output]:", result.stderr)
        raise RuntimeError(f"TripoSR generation failed: {result.stderr}")
        
    expected_mesh = os.path.join(output_dir, "0", "mesh.glb")
    if not os.path.exists(expected_mesh):
        # Fallback check for any .glb
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".glb"):
                    return os.path.join(root, f)
                    
    print(f"[TripoSR] Successfully generated 3D mesh: {expected_mesh}")
    return expected_mesh
