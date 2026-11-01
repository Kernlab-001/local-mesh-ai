# 🔰 Local Mesh AI — Complete Beginner Guide & Operations Manual

Welcome to **Local Mesh AI**! This guide is designed for creators of all experience levels—whether you are an experienced 3D technical artist, game developer, indie creator, or a beginner launching their very first generative 3D tool.

---

## 🧭 Table of Contents
- [1. Executive Architectural Overview](#1-executive-architectural-overview)
- [2. Hardware & System Requirements](#2-hardware--system-requirements)
- [3. The Quick-Start Launch Workflow](#3-the-quick-start-launch-workflow)
- [4. Core Systems & Model Identity Explained](#4-core-systems--model-identity-explained)
  - [TripoSR Neural 3D Engine](#triposr-neural-3d-engine)
  - [AI Technical Artist: Qwen Coder & Auxiliary Classifier](#ai-technical-artist-qwen-coder--auxiliary-classifier)
  - [Zero-VRAM GPU Lease Broker](#zero-vram-gpu-lease-broker)
  - [Blender 5.2 Studio Addon Suite](#blender-52-studio-addon-suite)
- [5. Beginner's Guide to Modifiers & Settings](#5-beginners-guide-to-modifiers--settings)
- [6. Exporting Assets to Game Engines (Unreal / Unity / Godot)](#6-exporting-assets-to-game-engines-unreal--unity--godot)
- [7. Troubleshooting & Diagnostics](#7-troubleshooting--diagnostics)
- [8. 3D Graphics Glossary for Beginners](#8-3d-graphics-glossary-for-beginners)
- [9. Trust, Privacy & Legal References](#9-trust-privacy--legal-references)

---

## 1. Executive Architectural Overview

Local Mesh AI combines feedforward neural reconstruction, large language model spatial reasoning, procedural PBR texturing, and automated Blender remeshing into a unified, local-first studio:

```
                    ┌────────────────────────────────────────┐
                    │      Natural Language User Request     │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │  AI Technical Artist Reasoning Engine  │
                    │  - Qwen 2.5 Coder 3B GGUF / 7B Base   │
                    │  - Auxiliary Technical Artist Adapter  │
                    │  - Bounded Pydantic ComponentEditPlan  │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
┌──────────────────────┐    ┌────────────────────────────────────────┐
│  2D Concept Artwork  │───>│         TripoSR Neural Engine          │
│  (PNG, JPG, WebP)    │    │  - Transparent Background Isolation    │
└──────────────────────┘    │  - Marching Cubes Watertight Meshing   │
                            │  - Real Peak VRAM Measurement (~2.5GB) │
                            │  - Isolated Staging & Job Folders      │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                            ┌────────────────────────────────────────┐
                            │    Geometry Validation & Inspection    │
                            │  - Coplanar Intersections (2D SAT)     │
                            │  - Manifold Health & UV Overlap Checks │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                            ┌────────────────────────────────────────┐
                            │       Blender 5.2 Finishing Studio     │
                            │  - QuadriFlow True Quad Retopology     │
                            │  - Cycles High-to-Low Texture Baking   │
                            │  - Multi-Tier LOD Generation (LOD0-3)  │
                            │  - Validated GLB, OBJ & FBX Export     │
                            └────────────────────────────────────────┘
```

---

## 2. Hardware & System Requirements

- **Operating System:** Windows 11 64-bit (or Ubuntu 22.04+ LTS).
- **GPU (Recommended Target):** NVIDIA RTX 3050 Laptop GPU (6 GB physical GDDR6 VRAM) or higher.
- **System RAM:** 16 GB Physical RAM.
- **Blender:** Blender 5.2 LTS or 4.2+ LTS.
- **Python:** Python 3.12 64-bit.
- **Storage:** ~10 GB free space for virtual environment, models, and cache.

---

## 3. The Quick-Start Launch Workflow

### Method A: Master Launcher (Recommended)
1. Double-click `Start_Local_Mesh_AI.bat` in the repository root.
2. Select option **`[1] Start Full Studio`**.
3. The launcher automatically verifies the Python virtual environment and spawns:
   - Backend Server on `http://127.0.0.1:8000` (API & Swagger Docs at `/docs`).
   - 3D Studio Workspace UI on `http://127.0.0.1:7860`.

### Method B: Blender Studio Addon
1. Open Blender 5.2.
2. In **Edit** $\rightarrow$ **Preferences** $\rightarrow$ **Add-ons**, install `addon/triposr_blender_addon.py`.
3. In the 3D Viewport, press `N` to open the sidebar and navigate to the single unified tab: **`LOCAL MESH AI`**.
4. Click **Start Backend Console** in Panel 1. The addon will automatically launch `run_server.py` using `.venv\Scripts\python.exe` with process-lifecycle tracking.

---

## 4. Core Systems & Model Identity Explained

### TripoSR Neural 3D Engine
- **Module:** `core/triposr_engine.py`
- Converts a single 2D concept image into a watertight 3D mesh via feedforward neural reconstruction.
- Uses Marching Cubes with chunk size `8192` and resolution `256`, optimized for 6GB laptop GPUs.

### AI Technical Artist: Qwen Coder & Auxiliary Classifier
- **Truthful Model Identity:**
  - **Primary LLM Reasoning:** Uses `Qwen2.5-Coder 3B (GGUF Q4_K_M)` on 6 GB VRAM GPUs (or 7B on 12GB+ workstations) for high-level semantic planning.
  - **Auxiliary Technical Artist Classifier:** An on-device PyTorch sequence classifier with trainable low-rank adaptation decomposition matrices (`services/lora_training_manager.py`). It predicts component targets (`backrest`, `legs`, `wheels`) and bounded scale parameters without altering the base LLM weights.
- **Bounded Safety:** AI models never execute arbitrary Python, shell code, or raw scripts. Every operation is strictly bounded to structured Pydantic `ComponentEditPlan` schemas.

### Zero-VRAM GPU Lease Broker
- **Module:** `core/gpu_broker.py`
- Enforces cross-process serialized GPU access via atomic disk lock files (`data/gpu_lease.lock`).
- Purges PyTorch CUDA cache between operations to prevent Out-Of-Memory (OOM) collisions.

### Blender 5.2 Studio Addon Suite
- All features are consolidated under a **single sidebar tab: `LOCAL MESH AI`**:
  - **Panel 1 — Connection & Console:** Live server status, latency, and GPU telemetry.
  - **Panel 2 — Generate AI 3D Asset:** Text-to-3D and Image-to-3D generation.
  - **Panel 3 — Asset Browser:** Inspect, filter, and import generated GLB assets.
  - **Panel 4 — Material Studio:** Synthesizes Principled BSDF procedural materials.
  - **Panel 5 — Transform Calibration:** Calibrates real-world metric scales (furniture, weapons, characters).
  - **Panel 6 — Scene Synchronization:** Bi-directional sync between Blender and backend.
  - **Panel 7 — Mesh Controls:** Subdivisions, normals, bevels, and manifold repairs.
  - **Panel 8 — AI Technical Artist:** Component decomposition and conversational mesh edits.
  - **Panel 9 — 3D Finishing & Retopo:** QuadriFlow remeshing, UV unwrapping, Cycles baking, LODs, and multi-format export.
  - **Panel 10 — System Health & Learning:** Autonomous diagnostics, two-stage dry-run preview recovery, and model lifecycle management.

---

## 5. Beginner's Guide to Modifiers & Settings

| Setting | Default Value | Plain-Language Description | When to Change It |
| :--- | :--- | :--- | :--- |
| **mc_resolution** | `256` | Marching Cubes voxel grid resolution. | Set to `128` for ultra-fast draft mode (~10s); `256` for standard (~30s). |
| **chunk_size** | `8192` | Point evaluation batch size during meshing. | Lower to `4096` if VRAM is under 4GB to prevent OOM. |
| **target_height_m**| `1.0m` | Desired real-world height in meters. | Adjust per asset (e.g., `0.9m` for a chair, `1.0m` for a sword, `1.8m` for a character). |
| **watertight** | `True` | Requires mesh to have 0 open boundary edges. | Always keep `True` for 3D printing and physics simulations. |
| **baking_resolution**| `1024` | Texture map dimensions (Normal, Diffuse, Roughness). | Use `2048` or `4096` for high-resolution hero game assets. |

---

## 6. Exporting Assets to Game Engines (Unreal / Unity / Godot)

1. **Unreal Engine 5:**
   - Drag and drop the generated `.glb` or `.fbx` into the Content Browser. UE5 automatically imports geometry, LODs, and materials.
2. **Unity:**
   - Place the `.glb` or `.fbx` into your `Assets/Models` folder. Use the Unity glTFast package or native FBX importer.
3. **Godot 4:**
   - Drag the `.glb` directly into the Godot FileSystem panel. Godot provides native glTF 2.0 support.

---

## 7. Troubleshooting & Diagnostics

### Issue 1: "Backend Offline" shown in Blender
- **Solution:** Click **Start Backend Console** in Panel 1 or run `Start_Local_Mesh_AI.bat` and select Option 1. Ensure `.venv` has dependencies installed.

### Issue 2: "llama.dll: The specified module could not be found (WinError 126)"
- **Solution:** Windows DLL loader requires CUDA runtime libraries. Ensure PyTorch CUDA is installed via `launchers\1_setup_environment.bat`.

### Issue 3: Staging Disk Cleanup Preview
- **Solution:** In Panel 10, click **1. Preview Safe Recovery (Dry Run)**. Review the identified files and unique Preview ID, then click **2. Execute Approved Recovery** to securely delete only reviewed files.

---

## 8. 3D Graphics Glossary for Beginners

* **GLB / glTF:** Standard 3D web and game format storing geometry, materials, and textures in a single binary file.
* **QuadriFlow:** State-of-the-art quad remeshing algorithm creating clean 4-sided polygon edge loops.
* **Selected-to-Active Baking:** Baking detailed surface geometry from a high-poly sculpt onto a clean low-poly game model.
* **LOD (Level of Detail):** Multiple versions of a 3D model with decreasing polygon counts used to optimize game rendering performance at distance.

---

## 9. Trust, Privacy & Legal References

- **Privacy Policy:** [PRIVACY.md](PRIVACY.md) — 100% on-device inference guarantee.
- **Security Policy:** [SECURITY.md](SECURITY.md) — Vulnerability reporting and sandboxed AI execution.
- **Third-Party Notices:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — Open-source license attribution.
- **Model Licenses:** [MODEL_LICENSES.md](MODEL_LICENSES.md) — Upstream AI model terms.
- **Windows Packaging:** [docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md](docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md) — Release packaging specifications.
