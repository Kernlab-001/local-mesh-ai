# Local Mesh AI — Project Technical Facts & Evidence Ledger

This document compiles the **authoritative, verified technical facts, software versions, test counts, architectural components, and known limitations** of the **Local Mesh AI** codebase.

Every claim in this document is linked to explicit files, line numbers, and test assertions in repository:
`https://github.com/Kernlab-001/local-mesh-ai`

---

## 1. Verified Software Environment & Dependencies

| Component | Verified Version | Configuration / Location in Repository | Evidence Source |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 x64 (Build 26100) | Native Windows host execution | Environment runtime |
| **Python** | Python 3.12.10 | Virtual environment: `E:\GPUPLAT\.venv` | `requirements.txt` |
| **PyTorch** | `2.6.0+cu124` | CUDA 12.4 build with cuDNN acceleration | `requirements.txt#L3` |
| **Torchvision** | `0.21.0+cu124` | PyTorch vision bundle | `requirements.txt#L4` |
| **CUDA Driver** | CUDA 12.4 compatible | NVIDIA GeForce RTX 3050 Laptop (6GB VRAM) | `detect_hardware.py#L18` |
| **Blender** | Blender 5.2.2 LTS / 4.2 LTS | `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` | `core/hardware_detector.py#L75` |
| **Diffusers** | $\ge 0.30.0$ | SD-Turbo concept diffusion in FP16 | `core/prompt_to_image.py#L25` |
| **Transformers** | `4.49.0` | Tokenizers, CLIP text encoders | `requirements.txt#L7` |
| **TripoSR** | Local submodule | Feed-forward neural 3D marching cubes | `TripoSR/run.py` |
| **rembg / BiRefNet** | `2.0.86` | GPU accelerated foreground isolation (ONNX) | `requirements.txt#L11` |
| **Trimesh** | `5.1.1` | Geometry verification, extents & manifold checks | `requirements.txt#L10` |
| **FastAPI** | $\ge 0.115.0$ | Async REST backend | `requirements.txt#L28` |
| **Gradio** | $\ge 5.0.0$ | Studio workspace frontend | `requirements.txt#L30` |
| **Three.js** | r128 | WebGL interactive 3D viewer | `ui/static/viewer.html#L21` |
| **python-pptx** | `1.0.2` | Native 16:9 widescreen presentation deck generation | `build_presentation.py` |

---

## 2. Verified Test Suite Counts

The automated test suite was executed on the Windows 11 host with zero failures and zero skipped tests:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0
rootdir: E:\GPUPLAT
collected 79 items

tests\test_api.py ...........                                            [ 14%]
tests\test_color_and_materials.py ............                           [ 29%]
tests\test_color_calibrator.py ..                                        [ 32%]
tests\test_command_schemas.py ....                                       [ 37%]
tests\test_dependency_verification.py ......                             [ 44%]
tests\test_dual_mode_pipeline.py ....                                    [ 50%]
tests\test_gpu_broker.py ....                                            [ 55%]
tests\test_hardware_detector.py ....                                     [ 60%]
tests\test_image_validation.py ...                                       [ 64%]
tests\test_job_lifecycle.py ...                                          [ 67%]
tests\test_mesh_serving_and_viewer.py .....                              [ 74%]
tests\test_onboarding_and_diagnostics.py .......                         [ 83%]
tests\test_production_pipeline.py ...                                    [ 87%]
tests\test_prompt_engine.py ..                                           [ 89%]
tests\test_ui_dashboard.py .........                                      [100%]

============================= 79 passed in 55.12s =============================
```

- **Total Tests Collected:** 79
- **Passed:** 79
- **Failed:** 0
- **Skipped:** 0
- **Deselected:** 0

### Breakdown by Verification Tier
1. **Blender Subprocess Tests (7 Tests):**
   - Tests in `tests/test_color_and_materials.py` and `tests/test_mesh_serving_and_viewer.py` spawn real headless Blender 5.2.2 LTS processes (`blender.exe --background --python-expr`) to verify shader node linkages, glTF roundtrip export, and add-on repair operators.
2. **Local AI / Neural Inference Tests (5 Tests):**
   - Tests in `tests/test_api.py`, `tests/test_dual_mode_pipeline.py`, and `tests/test_color_and_materials.py` execute real PyTorch CUDA inference, loading the SD-Turbo pipeline in FP16 precision.
3. **Appearance, Color & Material Tests (12 Tests):**
   - Tests in `tests/test_color_and_materials.py` validate material slot configurations, monochrome white detection, vertex color multiply bypass, and distinct palettes for diverse objects (e.g. sleek red sports car, green ceramic vase).
4. **Mocked / Isolated Unit Tests:**
   - Isolated unit tests in `tests/test_hardware_detector.py` and `tests/test_dependency_verification.py` verify fallback paths when dependencies or GPUs are simulated as missing.

---

## 3. Verified Architecture & Data Flow

### A. Dual-Entry Ingestion
- **Mode A (Text-to-3D):**
  - Handled by `services/job_manager.py` (lines 231–257).
  - Prompts are validated by `core/input_validator.py`.
  - Token budget prioritized for appearance/material keywords before CLIP 77-token ceiling (`core/prompt_to_image.py#L40-L100`).
  - Synthesizes 512x512 concept image via SD-Turbo FP16 (`core/prompt_to_image.py#L180-L240`).
- **Mode B (Image-to-3D):**
  - Handled by `services/job_manager.py` (lines 258–282).
  - Validates uploaded image format and dimensions.
  - Bypasses text diffusion completely to prevent hallucination.
  - Strips background clutter using `rembg` with BiRefNet (`core/input_validator.py#L90-L140`).

### B. Concurrency & GPU Leasing
- Managed by `core/gpu_broker.py`.
- Employs an OS file lock (`gpu_lease.lock`) combined with a Python `threading.Lock`.
- Guarantees that only **one job holds the GPU lease at any time**, preventing VRAM contention between SD-Turbo, TripoSR, and Blender.
- Executes `torch.cuda.empty_cache()` and `gc.collect()` at all task boundaries.

### C. Appearance Profile & Shader Repair
- `core/appearance_analyzer.py`: Generates `AppearanceProfile` and `MaterialSlotConfig`.
- Both classes support `to_dict()` and `from_dict()` for inter-process serialization.
- `services/job_manager.py` writes `profile.json` into the job workspace (`jobs/{job_id}/profile.json`) and passes `--profile` to Blender bridge subprocesses.
- `addon/triposr_blender_addon.py`:
  - `_inspect_material_slot()`: Walks material shader nodes. Detects unshaded monochrome white (`abs(R-G)<0.025 and abs(G-B)<0.025`), missing textures, or unlinked BSDF inputs.
  - `_apply_in_viewport_pbr()`: Preserves valid textured materials, repairs default white materials, and unlinks glTF multiply nodes directly into Base Color.
  - `TRIPOSR_OT_SwitchViewport`: Synchronizes diffuse colors and switches viewport to Material Preview mode.

---

## 4. Real Verified Assets On Disk

Inspected from golden reference job `jobs/91aa63f4/`:
- **Concept Art:** `jobs/91aa63f4/input/concept.png` (512x512 PNG, 392,124 bytes).
- **Multi-View Grid:** `jobs/91aa63f4/multiview/composite_grid.png` (1024x1024 PNG, 1,144,284 bytes).
- **Reconstructed Mesh:** `jobs/91aa63f4/mesh.glb` (5,900,756 bytes, 3 PBR material slots, 6 embedded textures).
- **Game-Ready Mesh:** `jobs/91aa63f4/low_poly_asset/model_game_ready.glb` (12,815,204 bytes).
- **Blender Project:** `jobs/91aa63f4/project.blend` (5,065,495 bytes).
- **Baked Normal Map:** `jobs/91aa63f4/textures/baked_normal.png` (10,579,639 bytes, 16-bit tangent space).
- **Baked Diffuse Map:** `jobs/91aa63f4/textures/baked_diffuse.png` (1,949,918 bytes).
- **EEVEE Viewport Render:** `jobs/91aa63f4/preview.png` (1,897,861 bytes) & `output/chair_preview_render.png0001.png` (1,979,922 bytes).

---

## 5. Known Limitations & Technical Constraints

1. **Marching-Cubes Topology:**
   - Raw neural outputs from TripoSR are dense, uniform triangle meshes. While automated quad remeshing decimates them to an 8,000-quad budget, edge loops are not oriented for skeletal joint deformation without manual retopology.
2. **Heuristic Material Face Partitioning:**
   - Boundary classification uses bounding-box height thresholds and surface normals. While highly reliable for structured categories (chairs, tables, vehicles), abstract organic sculptures may require manual face grouping in Blender Edit Mode.
3. **Single-Perspective Occlusion:**
   - The backside of 3D objects is inferred by TripoSR's neural priors rather than measured from real multi-view photographs.
4. **VRAM Constraints (6GB Budget):**
   - Models must be swapped sequentially (diffusion unloads before TripoSR loads). This adds ~3–5 seconds of model-loading overhead per job.
