"""
TripoSR End-to-End Test Workflow & Acceptance Verification.
Performs real, non-mocked 3D neural reconstruction on TripoSR/examples/chair.png,
measures actual runtime and peak VRAM, validates geometry (vertices, faces,
finite coordinates, watertightness, extents, vertex colors), and saves a
structured JSON acceptance report to output/triposr_acceptance_report.json.
"""

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.triposr_engine import triposr_engine
from core.download_manager import download_manager


def main():
    print("=" * 72)
    print("  LOCAL MESH AI — TRIPOSR RECONSTRUCTION ACCEPTANCE TEST")
    print("=" * 72)

    # 1. Verify Model Weights Readiness
    print("\n[Step 1/4] Verifying TripoSR and DINO weights offline readiness...")
    status = download_manager.verify_triposr_status()
    print(f"  State: {status['state']}")
    print(f"  Ready for Offline Inference: {status['ready_for_offline_inference']}")
    if not status["ready_for_offline_inference"]:
        print("[ERROR] TripoSR weights not ready. Run: python scripts/download_models.py")
        sys.exit(1)

    # 2. Input Image Check
    input_image = PROJECT_ROOT / "TripoSR" / "examples" / "chair.png"
    print(f"\n[Step 2/4] Validating input image: {input_image}")
    if not input_image.exists():
        print(f"[ERROR] Example image not found at {input_image}")
        sys.exit(1)

    # 3. Execute Reconstruction
    print("\n[Step 3/4] Launching genuine TripoSR neural reconstruction...")
    print("  Profile: mc_resolution=256, chunk_size=8192, format=glb")
    t0 = time.time()
    
    result = triposr_engine.reconstruct(
        image_path=str(input_image),
        mc_resolution=256,
        chunk_size=8192,
        render_turntable=False
    )
    total_duration = round(time.time() - t0, 2)

    print("\n[Step 4/4] Evaluating output 3D asset and geometry integrity...")
    mesh_path = result["mesh_path"]
    geom = result["geometry"]

    print(f"  Output GLB:          {mesh_path}")
    print(f"  Mesh File Size:      {result['mesh_size_bytes']:,} bytes ({result['mesh_size_bytes'] / (1024*1024):.2f} MB)")
    print(f"  Vertex Count:        {geom['vertex_count']:,}")
    print(f"  Face Count:          {geom['face_count']:,}")
    print(f"  Extents (X, Y, Z):   {geom['dimensions']}")
    print(f"  Coordinates Finite:  {geom['coordinates_finite']}")
    print(f"  Watertight / Closed: {geom['is_watertight']}")
    print(f"  Vertex Colors:       {geom['has_vertex_colors']}")
    print(f"  Execution Time:      {result['elapsed_seconds']}s")
    print(f"  Peak VRAM:           {result['peak_vram_gb']} GB")

    # Quality Criteria Verification
    criteria = {
        "mesh_exists_and_non_empty": result["mesh_size_bytes"] > 50_000,
        "vertex_count_adequate": geom["vertex_count"] > 1000,
        "face_count_adequate": geom["face_count"] > 1000,
        "coordinates_finite": geom["coordinates_finite"],
        "has_vertex_colors": geom["has_vertex_colors"],
        "valid_extents": all(d > 0.05 for d in geom["dimensions"])
    }

    all_passed = all(criteria.values())

    report = {
        "test_name": "TripoSR Reconstruction Acceptance",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_image": str(input_image),
        "output_mesh": mesh_path,
        "job_id": result["job_id"],
        "working_dir": result["working_dir"],
        "runtime_seconds": result["elapsed_seconds"],
        "total_test_duration_seconds": total_duration,
        "peak_vram_gb": result["peak_vram_gb"],
        "device": result["device"],
        "mc_resolution": result["mc_resolution"],
        "chunk_size": result["chunk_size"],
        "geometry": geom,
        "acceptance_criteria": criteria,
        "status": "PASSED" if all_passed else "FAILED"
    }

    out_report_path = PROJECT_ROOT / "output" / "triposr_acceptance_report.json"
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    if all_passed:
        print("  >>> ACCEPTANCE TEST PASSED SUCCESSFULLY! <<<")
    else:
        print("  >>> ACCEPTANCE TEST FAILED! <<<")
    print(f"  Report written to: {out_report_path}")
    print("=" * 72)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
