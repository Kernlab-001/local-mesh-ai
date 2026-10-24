"""
Real Local Generation Verification Script for Local Mesh AI.
Executes actual local inference on host hardware:
1. TripoSR Image-to-3D using TripoSR/examples/teapot.png
2. Hybrid Text-to-3D (SD-Turbo + TripoSR) using prompt 'polished dark walnut dining chair with ivory cushion backrest'
Measures:
- Real runtime
- Real peak VRAM via torch.cuda.max_memory_allocated()
- File size, vertex count, face count, watertightness, dimensions
- Runs Headless Blender 5.2 verification and renders preview images
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engines.registry import engine_registry
from core.geometry_validator import geometry_validator
from tests.benchmark_suite import run_benchmark_suite, execute_real_generation_for_case, BENCHMARK_CASES


def main():
    print("=================================================================")
    print("LOCAL MESH AI — REAL HARDWARE GENERATION PROOF")
    print("=================================================================")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available on host system.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    print(f"Host GPU: {device_name} ({total_vram} GB VRAM)")

    # Execute benchmark suite with real generation on Case 04 (Image-to-3D) and Case 02 (Text-to-3D)
    cases_to_run = ["case_04", "case_02"]
    print(f"\nRunning benchmark suite with verified generation on: {cases_to_run}...\n")

    summary = run_benchmark_suite(verify_real_generation_cases=cases_to_run)

    print("\n=================================================================")
    print("BENCHMARK EXECUTION SUMMARY")
    print("=================================================================")
    print(f"Total Cases Evaluated: {summary['total_cases']}")
    print(f"Architecture Smoke Passed: {summary['architecture_smoke_passed']}/{summary['total_cases']} ({summary['architecture_smoke_pass_rate_pct']}%)")
    print(f"Real Generation Tested: {summary['real_generation_tested']}")
    print(f"Real Generation Passed: {summary['real_generation_passed']}")
    print(f"Blocked Cases: {summary['blocked_cases']}")
    print(f"Not Run Cases: {summary['not_run_cases']}")
    print(f"Total Suite Runtime: {summary['total_duration_seconds']}s")

    # Copy rendered preview images to artifacts directory if available
    artifacts_dir = Path(r"C:\Users\diyaj\.gemini\antigravity-cli\brain\efc491ce-8d5f-4517-9d46-10ef382ab23b")
    for r in summary["results"]:
        if r["generation_status"] == "PASS":
            gd = r["generation_details"]
            bv = gd.get("blender_validation", {})
            render_path = bv.get("render_path")
            if render_path and os.path.exists(render_path):
                dest = artifacts_dir / f"{r['case_id']}_blender_render.png"
                shutil.copy2(render_path, dest)
                print(f"[Artifact Created] Copied Blender render for {r['case_id']} to {dest}")


if __name__ == "__main__":
    main()
