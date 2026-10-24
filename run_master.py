"""
Master Pipeline Runner for E:\\GPUPLAT.
Orchestrates:
1. TripoSR Neural 3D Generation (RTX 3050 CUDA)
2. Memory Purge
3. Blender 5.2 / 4.2 Automated Sanitation, PBR Shading, and Project Injection
"""

import os
import sys
import json
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    from core.hardware_detector import hardware_detector
    return {
        "blender_path": hardware_detector.find_blender() or "blender",
        "output_dir": os.path.join(SCRIPT_DIR, "output")
    }

def main():
    parser = argparse.ArgumentParser(description="TripoSR to Blender Master Pipeline")
    parser.add_argument("--image", default=None, help="Path to input 2D image")
    parser.add_argument("--prompt", default=None, help="Text prompt to generate 3D asset from scratch")
    parser.add_argument("--blend", default=None, help="Optional existing .blend project to inject asset into")
    parser.add_argument("--height", type=float, default=1.8, help="Target real-world height in meters")
    parser.add_argument("--lods", action="store_true", help="Generate 3 game-ready LOD meshes")
    parser.add_argument("--open-blender", action="store_true", help="Launch Blender GUI when complete")
    args = parser.parse_args()

    config = load_config()
    from core.hardware_detector import hardware_detector
    blender_exe = config.get("blender_path") or hardware_detector.find_blender() or "blender"
    venv_python = os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
    
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    print("=" * 60)
    gpu_label = config.get("gpu_name", "CUDA Auto-Tuned")
    print(f"AI 3D ENGINE: TRIPOSR + BLENDER PIPELINE ({gpu_label})")
    print("=" * 60)

    image_path = args.image
    if not image_path and args.prompt:
        print(f"[Text-to-3D] Generating concept image from prompt: '{args.prompt}'...")
        from core.prompt_to_image import prompt_engine
        import time
        inputs_dir = os.path.join(SCRIPT_DIR, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        image_path = os.path.join(inputs_dir, f"prompt_{int(time.time())}.png")
        prompt_engine.generate(prompt=args.prompt, output_path=image_path)
        print(f"[Text-to-3D] Concept image ready: {image_path}")

    if not image_path or not os.path.exists(image_path):
        print("Error: Please provide either --image or --prompt.")
        sys.exit(1)

    # 1. TripoSR Generation
    from core.triposr_runner import run_triposr_inference
    mesh_output_dir = os.path.join(SCRIPT_DIR, "output", "meshes")
    generated_mesh = run_triposr_inference(image_path, output_dir=mesh_output_dir, python_exe=venv_python)

    # 2. Blender Automated Bridge
    bridge_script = os.path.join(SCRIPT_DIR, "bridge", "blender_bridge.py")
    out_dir = os.path.join(SCRIPT_DIR, "output")
    
    blender_cmd = [
        blender_exe,
        "--background",
        "--python", bridge_script,
        "--",
        "--mesh", generated_mesh,
        "--image", os.path.abspath(image_path),
        "--out", out_dir,
        "--height", str(args.height)
    ]
    if args.lods:
        blender_cmd.append("--lods")
    if args.blend:
        blender_cmd.extend(["--blend", os.path.abspath(args.blend)])

    print(f"\n[Blender Engine] Launching Blender processing...")
    try:
        res = subprocess.run(blender_cmd, capture_output=True, text=True, timeout=180.0)
        if res.returncode != 0:
            print("[Blender Warning/Error]:", res.stderr)
        else:
            print(res.stdout)
    except subprocess.TimeoutExpired:
        print("[Blender Error]: Subprocess execution timed out after 180 seconds.")

    sanitized_mesh = os.path.join(out_dir, "mesh.glb")
    final_mesh = sanitized_mesh if os.path.exists(sanitized_mesh) and os.path.getsize(sanitized_mesh) > 0 else generated_mesh
    if not os.path.exists(final_mesh) or os.path.getsize(final_mesh) == 0:
        raise RuntimeError(f"Pipeline finished but 3D mesh is missing or empty at {final_mesh}")

    print("\n" + "=" * 60)
    print("[COMPLETED] 3D Asset Successfully Generated & Sanitized!")
    print(f"3D Model (GLB): {final_mesh} ({os.path.getsize(final_mesh):,} bytes)")
    print(f"Project File:   {os.path.join(out_dir, 'project.blend')}")
    print(f"Preview Render: {os.path.join(out_dir, 'preview.png')}")
    print("=" * 60)

    if args.open_blender:
        blend_file = os.path.join(out_dir, "project.blend")
        subprocess.Popen([blender_exe, blend_file])

if __name__ == "__main__":
    main()
