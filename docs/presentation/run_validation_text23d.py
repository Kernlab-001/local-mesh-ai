"""
Independent Live End-to-End Text-to-3D Validation Script.
Executes the full pipeline with the prompt specified in Section 3 of the prompt:
'A traditional dining chair with a dark walnut-brown wooden frame and carved legs, a warm ivory-cream upholstered seat cushion, a matching cream tufted backrest with fabric buttons, realistic wood grain, woven upholstery fabric, subtle stitching, and detailed trim.'
Measures:
- Elapsed time per stage
- Peak CUDA VRAM allocation
- Output asset validity (manifest, GLB size, materials, textures, UVs)
- GLB export/re-import integrity
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
from core.appearance_analyzer import appearance_analyzer

PROMPT = (
    "A traditional dining chair with a dark walnut-brown wooden frame and carved legs, "
    "a warm ivory-cream upholstered seat cushion, a matching cream tufted backrest with fabric buttons, "
    "realistic wood grain, woven upholstery fabric, subtle stitching, and detailed trim."
)

def run_validation():
    print("=" * 70)
    print("STARTING INDEPENDENT END-TO-END TEXT-TO-3D VALIDATION")
    print(f"Prompt: {PROMPT}")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()
    start_vram = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    params = {
        "prompt": PROMPT,
        "target_height": 0.9,
        "preset": "furniture",
        "quality": "balanced",
        "remove_bg": True,
        "sanitize": True,
        "sharp_overlay": True,
        "weld": True,
        "pbr_material": True,
        "accurate_colors": True,
        "retopology_and_bake": True,
        "target_faces": 8000
    }

    t0 = time.time()
    job = job_manager.create_job(params)
    print(f"[JOB CREATED] ID: {job.job_id}")

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
    print(f"PIPELINE CONCLUDED in {total_time:.2f}s | Final Status: {final_job.status}")
    print(f"Peak CUDA VRAM: {peak_vram_mb:.2f} MB (~{peak_vram_mb/1024:.2f} GB)")
    print("=" * 70)

    if final_job.status != JobStatus.COMPLETED:
        print(f"[ERROR] Job failed with error: {final_job.error}")
        return False, None

    job_dir = os.path.join(PROJECT_ROOT, "jobs", final_job.job_id)
    mesh_path = final_job.output_mesh
    print(f"Output Mesh: {mesh_path} ({os.path.getsize(mesh_path):,} bytes)")

    # 1. Trimesh inspection
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = len(mesh.vertices)
    faces = len(mesh.faces)
    watertight = getattr(mesh, "is_watertight", False)
    ext = mesh.extents
    print(f"Geometry: {verts:,} vertices, {faces:,} faces | Watertight: {watertight} | Bounds: {ext[0]:.2f}m x {ext[1]:.2f}m x {ext[2]:.2f}m")

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

    # 3. Roundtrip GLB Export and Re-import test
    roundtrip_path = os.path.join(job_dir, "roundtrip_test.glb")
    mesh.export(roundtrip_path)
    assert os.path.exists(roundtrip_path) and os.path.getsize(roundtrip_path) > 0, "Roundtrip export failed!"
    reloaded = trimesh.load(roundtrip_path, force="mesh")
    assert len(reloaded.vertices) == verts, "Roundtrip vertex count mismatch!"
    print(f"[PASSED] Roundtrip export/re-import verified: {len(reloaded.vertices):,} vertices preserved.")

    results = {
        "job_id": final_job.job_id,
        "status": final_job.status,
        "total_time_seconds": round(total_time, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "mesh_path": mesh_path,
        "file_size_bytes": os.path.getsize(mesh_path),
        "vertices": verts,
        "faces": faces,
        "watertight": watertight,
        "bounds": [round(float(x), 3) for x in ext],
        "materials": glb_mats,
        "has_uvs": has_uvs,
        "has_textures": has_textures,
        "manifest": final_job.manifest,
        "logs": final_job.logs
    }

    results_file = os.path.join(PROJECT_ROOT, "docs", "presentation", "validation_text23d_results.json")
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(results, rf, indent=2)

    print(f"[SUCCESS] Detailed validation results saved to: {results_file}")
    return True, results

if __name__ == "__main__":
    success, res = run_validation()
    if not success:
        sys.exit(1)
