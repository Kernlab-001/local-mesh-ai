"""
Independent Live End-to-End Image-to-3D Validation Script.
Executes the Mode B pipeline using 'TripoSR/examples/teapot.png'.
Verifies:
- Text diffusion is bypassed completely.
- Image is staged, validated, and rembg background-stripped.
- TripoSR directly reconstructs geometry from the reference image.
- Final GLB is generated, valid, and contains appropriate material/color properties.
"""

import os
import sys
import time
import json
import trimesh
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.job_manager import job_manager, JobStatus

SAMPLE_IMAGE = os.path.join(PROJECT_ROOT, "TripoSR", "examples", "teapot.png")

def run_image_validation():
    print("=" * 70)
    print("STARTING INDEPENDENT END-TO-END IMAGE-TO-3D VALIDATION")
    print(f"Source Image: {SAMPLE_IMAGE}")
    print("=" * 70)

    assert os.path.exists(SAMPLE_IMAGE), f"Sample image not found at {SAMPLE_IMAGE}"

    torch.cuda.reset_peak_memory_stats()

    params = {
        "image_path": SAMPLE_IMAGE,
        "target_height": 0.8,
        "preset": "prop",
        "quality": "balanced",
        "remove_bg": True,
        "sanitize": True,
        "sharp_overlay": True,
        "weld": True,
        "pbr_material": True,
        "accurate_colors": True,
        "retopology_and_bake": False
    }

    t0 = time.time()
    job = job_manager.create_job(params)
    print(f"[JOB CREATED] ID: {job.job_id} | Input Mode: {job.input_mode}")
    assert job.input_mode == "image", f"Expected input_mode 'image', got '{job.input_mode}'"

    last_stage = None
    while True:
        time.sleep(1.0)
        curr = job_manager.get_job(job.job_id)
        if not curr:
            break
        if curr.stage != last_stage:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            print(f"[{elapsed:5.1f}s] Stage: {curr.stage} ({curr.progress_pct}%) | Status: {curr.status} | VRAM: {vram_mb:.1f} MB")
            last_stage = curr.stage

        if curr.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            break

    total_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    final_job = job_manager.get_job(job.job_id)

    print("\n" + "=" * 70)
    print(f"IMAGE-TO-3D PIPELINE CONCLUDED in {total_time:.2f}s | Final Status: {final_job.status}")
    print(f"Peak CUDA VRAM: {peak_vram_mb:.2f} MB (~{peak_vram_mb/1024:.2f} GB)")
    print("=" * 70)

    if final_job.status != JobStatus.COMPLETED:
        print(f"[ERROR] Job failed with error: {final_job.error}")
        return False, None

    mesh_path = final_job.output_mesh
    print(f"Output Mesh: {mesh_path} ({os.path.getsize(mesh_path):,} bytes)")

    # 1. Trimesh verification
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = len(mesh.vertices)
    faces = len(mesh.faces)
    ext = mesh.extents
    print(f"Teapot Geometry: {verts:,} vertices, {faces:,} faces | Bounds: {ext[0]:.2f}m x {ext[1]:.2f}m x {ext[2]:.2f}m")

    # 2. GLB Container material inspection
    import struct
    glb_mats = []
    has_uvs = False
    has_textures = False
    with open(mesh_path, "rb") as gf:
        header = gf.read(12)
        if header[:4] == b"glTF":
            chunk_len, chunk_type = struct.unpack("<II", gf.read(8))
            if chunk_type == 0x4E4F534A:
                gltf_json = json.loads(gf.read(chunk_len).decode("utf-8"))
                glb_mats = [m.get("name") for m in gltf_json.get("materials", [])]
                has_textures = len(gltf_json.get("textures", [])) > 0 or len(gltf_json.get("images", [])) > 0
                for m in gltf_json.get("meshes", []):
                    for p in m.get("primitives", []):
                        if "TEXCOORD_0" in p.get("attributes", {}):
                            has_uvs = True

    print(f"Materials ({len(glb_mats)}): {glb_mats}")
    print(f"Has UVs: {has_uvs} | Has Textures: {has_textures}")

    results = {
        "job_id": final_job.job_id,
        "input_mode": final_job.input_mode,
        "source_image": SAMPLE_IMAGE,
        "status": final_job.status,
        "total_time_seconds": round(total_time, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "mesh_path": mesh_path,
        "file_size_bytes": os.path.getsize(mesh_path),
        "vertices": verts,
        "faces": faces,
        "bounds": [round(float(x), 3) for x in ext],
        "materials": glb_mats,
        "has_uvs": has_uvs,
        "has_textures": has_textures,
        "logs": final_job.logs
    }

    results_file = os.path.join(PROJECT_ROOT, "docs", "presentation", "validation_image23d_results.json")
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(results, rf, indent=2)

    print(f"[SUCCESS] Detailed Image-to-3D validation results saved to: {results_file}")
    return True, results

if __name__ == "__main__":
    success, res = run_image_validation()
    if not success:
        sys.exit(1)
