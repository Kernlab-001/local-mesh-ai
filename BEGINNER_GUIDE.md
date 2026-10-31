# 🔰 Local Mesh AI — Complete Beginner Guide & Operations Manual

Welcome to **Local Mesh AI**! This guide is designed for everyone—whether you are a 3D technical artist, game developer, indie creator, or a complete beginner who has never run a Python script, touched an AI model, or opened Blender before.

---

## 🧭 Table of Contents
- [1. Architectural Overview](#1-architectural-overview)
- [2. Hardware & System Requirements](#2-hardware--system-requirements)
- [3. The 7-Step Setup & Verification Workflow](#3-the-7-step-setup--verification-workflow)
  - [Step 1: Environment & Hardware Detection](#step-1-environment--hardware-detection)
  - [Step 2: Dedicated Git Branch & Codebase Integrity](#step-2-dedicated-git-branch--codebase-integrity)
  - [Step 3: Dependency Installation & CUDA Acceleration](#step-3-dependency-installation--cuda-acceleration)
  - [Step 4: Offline Model Provisioning](#step-4-offline-model-provisioning)
  - [Step 5: Truthful Readiness & Offline Verification](#step-5-truthful-readiness--offline-verification)
  - [Step 6: Executing 3D Reconstruction & Technical-Artist Planning](#step-6-executing-3d-reconstruction--technical-artist-planning)
  - [Step 7: Automated Blender Sanitation & Scene Inspection](#step-7-automated-blender-sanitation--scene-inspection)
- [4. Core Systems Explained](#4-core-systems-explained)
  - [TripoSR Volumetric Engine](#triposr-volumetric-engine)
  - [Qwen 2.5 Coder Technical-Artist Planner](#qwen-25-coder-technical-artist-planner)
  - [Zero-VRAM GPU Lease Broker](#zero-vram-gpu-lease-broker)
  - [Per-Job Isolated Workspaces](#per-job-isolated-workspaces)
- [5. Beginner's Guide to Modifiers & Settings](#5-beginners-guide-to-modifiers--settings)
- [6. Exporting Assets to Game Engines (Unreal / Unity / Godot)](#6-exporting-assets-to-game-engines-unreal--unity--godot)
- [7. Troubleshooting & Diagnostics](#7-troubleshooting--diagnostics)
- [8. 3D Graphics Glossary for Beginners](#8-3d-graphics-glossary-for-beginners)

---

## 1. Architectural Overview

Local Mesh AI is centered around **TripoSR** as its primary neural 3D reconstruction engine and **Qwen 2.5 Coder 7B** as its central Technical-Artist intelligence.

```
                    ┌────────────────────────────────────────┐
                    │      Natural Language User Request     │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │      Qwen 2.5 Coder 7B Planner         │
                    │  - Component Identification & Hierarchy│
                    │  - Proportions & Metric Dimensions     │
                    │  - PBR Materials & Color Schemes       │
                    │  - Topology Budget & Mesh Style        │
                    │  - Deterministic Blender Commands      │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
┌──────────────────────┐    ┌────────────────────────────────────────┐
│  2D Concept Artwork  │───>│         TripoSR Neural Engine          │
│  (PNG, JPG, WebP)    │    │  - Transparent Background Preparation  │
└──────────────────────┘    │  - Volumetric NeRF / Marching Cubes    │
                            │  - Real Peak VRAM Measurement (~3.1GB) │
                            │  - Strict Isolated Job Folders         │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                            ┌────────────────────────────────────────┐
                            │    Geometry Validation & Inspection    │
                            │  - Vertices (>10K), Faces (>20K)       │
                            │  - Watertightness & Finite Coordinates │
                            │  - Extents & Vertex Colors Verified    │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                            ┌────────────────────────────────────────┐
                            │       Blender 4.2 / 5.2 Bridge         │
                            │  - Safe Command Execution              │
                            │  - Real-world Scaling & PBR Materials  │
                            │  - Final GLB & .blend Scene Export     │
                            └────────────────────────────────────────┘
```

---

## 2. Hardware & System Requirements

| Component | Minimum Requirement | Verified Test Platform |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit) | Windows 11 (64-bit) |
| **Processor (CPU)** | Intel Core i5 / AMD Ryzen 5 | Intel Core i5-13420H (8 Cores, 16 Threads) |
| **Graphics Card (GPU)** | NVIDIA GPU with 6.0 GB VRAM (CUDA 12+) | NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM) |
| **System Memory (RAM)**| 16 GB DDR4/DDR5 | 16 GB RAM |
| **Storage (Persistent)**| 15 GB free space on SSD | High-speed SSD (`E:\LocalMeshAI-Data\models`) |
| **Blender Version** | Blender 3.6 LTS | Blender 5.2 / 4.2 LTS (`C:\Program Files\Blender Foundation\Blender 5.2`) |

> [!IMPORTANT]
> **Single-Occupancy GPU Budgeting:** On 6GB VRAM GPUs (such as RTX 3050), TripoSR requires ~3.1 GB and Qwen 2.5 Coder requires ~4.2 GB. They **must never run simultaneously** in VRAM. The system enforces an atomic cross-process lease broker (`core/gpu_broker.py`) to guarantee zero OOM collisions.

---

## 3. The 7-Step Setup & Verification Workflow

### Step 1: Environment & Hardware Detection
Detect your GPU, available VRAM, CUDA runtime, and Blender installation:
```powershell
python -c "from core.hardware_detector import hardware_detector; print(hardware_detector.detect_gpu())"
```

### Step 2: Dedicated Git Branch & Codebase Integrity
Confirm you are on the dedicated Phase 1 branch (`phase-1/triposr-foundation`):
```powershell
git status
git branch
```

### Step 3: Dependency Installation & CUDA Acceleration
Activate your virtual environment and verify PyTorch CUDA and `llama-cpp-python`:
```powershell
.venv\Scripts\activate
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
python -c "import llama_cpp; print('llama_cpp Version:', llama_cpp.__version__)"
```

### Step 4: Offline Model Provisioning
Download model weights to persistent storage outside Git (`E:\LocalMeshAI-Data\models`):
```powershell
# Provision TripoSR and DINO ViT weights (~1.67 GB)
python scripts/download_models.py --triposr

# Provision Qwen 2.5 Coder 7B GGUF weights (~4.7 GB)
python scripts/download_models.py --qwen
```

### Step 5: Truthful Readiness & Offline Verification
Verify all model files and checksums offline without network requests:
```powershell
python scripts/download_models.py --check
```
*Output should report `TripoSR Status: fully_installed`, `Ready for Offline Inference: True`, and `Qwen Status: fully_installed`.*

### Step 6: Executing 3D Reconstruction & Technical-Artist Planning
Run the end-to-end acceptance test workflow:
```powershell
# 1. Run TripoSR Neural Reconstruction on reference chair
python scripts/test_triposr_workflow.py

# 2. Test Qwen Technical-Artist Planning
python scripts/test_qwen_planner.py
```

### Step 7: Automated Blender Sanitation & Scene Inspection
Generate a complete asset and open the project directly inside Blender:
```powershell
python run_master.py --image TripoSR/examples/chair.png --open-blender
```

---

## 4. Core Systems Explained

### TripoSR Volumetric Engine
- **Module:** `core/triposr_engine.py`
- Converts a single 2D image into 3D volumetric representations using feedforward neural reconstruction.
- Uses Marching Cubes with chunk size `8192` and resolution `256` calibrated for 6GB laptop GPUs.
- Performs automatic background isolation, watertightness checking, coordinate finiteness checks, and vertex color inspection.

### Qwen 2.5 Coder Technical-Artist Planner
- **Module:** `core/qwen_planner.py` & `core/command_schemas.py`
- Operates as the lead technical artist:
  1. Identifies components and structural hierarchy (e.g. blade, crossguard, grip, pommel).
  2. Sets real-world metric dimensions (height, width, depth, symmetry).
  3. Formulates PBR material specifications (preset, base color, roughness, metallic).
  4. Prescribes topology budgets and geometry correction targets.
  5. Translates requirements into an allowlist of deterministic Blender operations.

### Zero-VRAM GPU Lease Broker
- **Module:** `core/gpu_broker.py`
- Uses atomic OS file locks and thread synchronizers to prevent concurrent neural operations.
- Automatically purges PyTorch CUDA cache and reclaims memory when releasing a lease.

### Per-Job Isolated Workspaces
- All generated meshes, textures, logs, and intermediary files are stored in unique timestamped folders (`output/triposr_<timestamp>_<uuid>/`).
- Shared collision-prone folders (such as `output/meshes`) have been eliminated.

---

## 5. Beginner's Guide to Modifiers & Settings

| Setting | Default Value | Plain-Language Description | When to Change It |
| :--- | :--- | :--- | :--- |
| **mc_resolution** | `256` | Marching Cubes voxel grid resolution. | Set to `128` for ultra-fast draft mode (~10s); `256` for standard (~30s). |
| **chunk_size** | `8192` | Point evaluation batch size during meshing. | Lower to `4096` if VRAM is under 4GB to prevent OOM. |
| **target_height_m**| `1.0m` | Desired real-world height in meters. | Adjust per asset (e.g., `0.3m` for a bottle, `1.2m` for a sword, `1.8m` for a character). |
| **watertight** | `True` | Requires mesh to have 0 open boundary edges. | Always keep `True` for 3D printing and physics simulations. |
| **decimate_ratio** | `0.5` | Target polygon triangle reduction ratio. | Use `0.3`–`0.5` for mobile games; leave empty for high-poly raw sculpt. |

---

## 6. Exporting Assets to Game Engines (Unreal / Unity / Godot)

1. **Unreal Engine 5:**
   - Drag and drop the generated `.glb` file into the Content Browser.
   - UE5 will automatically import the geometry, hierarchy, and materials.
2. **Unity:**
   - Place the `.glb` into your `Assets/Models` folder.
   - Use the Unity glTFast package or convert to FBX via Blender.
3. **Godot 4:**
   - Drag the `.glb` directly into the Godot FileSystem panel. Godot provides native glTF 2.0 support.

---

## 7. Troubleshooting & Diagnostics

### Issue 1: "PermissionError: [WinError 5] Access is denied: 'output'"
- **Cause:** Process running in a sandboxed shell attempting to write to external drive `E:\`.
- **Solution:** Run with standard user execution privileges or allow script bypass permissions.

### Issue 2: "llama.dll: The specified module could not be found (WinError 126)"
- **Cause:** Windows DLL loader cannot find CUDA 12.4 runtime libraries (`cudart64_12.dll`, `cublas64_12.dll`).
- **Solution:** The system includes automatic DLL pre-loading in `core/qwen_planner.py` pointing to `.venv/Lib/site-packages/torch/lib`. Ensure PyTorch CUDA 12.4 is installed in `.venv`.

### Issue 3: "CUDA Out of Memory (OOM)"
- **Cause:** Concurrent processes holding VRAM or unreleased cache.
- **Solution:** Verify the lock file with `python -c "from core.gpu_broker import gpu_broker; gpu_broker.purge_vram()"` and ensure no other background tasks are holding the GPU.

---

## 8. 3D Graphics Glossary for Beginners

* **GLB / glTF:** GL Transmission Format. The modern standard for 3D web and game engines, storing geometry, materials, and textures in a single binary file.
* **Vertex / Face:** Vertices are the points in 3D space; faces are the polygons formed by connecting vertices.
* **Watertight:** A mesh with no holes or self-intersecting boundaries—solid and manifold.
* **Marching Cubes:** An algorithm that turns volumetric neural density fields into polygonal 3D surfaces.
* **PBR (Physically Based Rendering):** Modern realistic shading based on physical light transport (Base Color, Metallic, Roughness, Normal).
