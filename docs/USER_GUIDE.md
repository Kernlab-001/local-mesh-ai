# Local Mesh AI — Complete User Manual & Operations Guide

Welcome to **Local Mesh AI**! This guide is designed for creators of all experience levels—whether you are an experienced 3D artist, game developer, or a complete beginner.

---

## Table of Contents
1. [What is Local Mesh AI?](#1-what-is-local-mesh-ai)
2. [Hardware & Software Requirements](#2-hardware--software-requirements)
3. [First-Time Installation & Startup](#3-first-time-installation--startup)
4. [Understanding the Two Creation Modes](#4-understanding-the-two-creation-modes)
   - [Mode A: Text-to-3D](#mode-a-text-to-3d)
   - [Mode B: Image-to-3D](#mode-b-image-to-3d)
5. [Step-by-Step Tutorial: Your First 3D Project](#5-step-by-step-tutorial-your-first-3d-project)
6. [The 3D Viewport & Inspection Tools](#6-the-3d-viewport--inspection-tools)
7. [Beginner's Guide to Modifiers & Settings](#7-beginners-guide-to-modifiers--settings)
8. [Blender 5.2 Studio Addon Suite](#8-blender-52-studio-addon-suite)
9. [Exporting Assets to Unreal, Unity, and Godot](#9-exporting-assets-to-unreal-unity-and-godot)
10. [Where Your Files Are Saved](#10-where-your-files-are-saved)
11. [Troubleshooting & Common Errors](#11-troubleshooting--common-errors)
12. [Frequently Asked Questions (FAQ)](#12-frequently-asked-questions-faq)
13. [Beginner 3D Graphics Glossary](#13-beginner-3d-graphics-glossary)
14. [Trust, Privacy & Governance](#14-trust-privacy--governance)

---

## 1. What is Local Mesh AI?

**Local Mesh AI** is an all-in-one local production tool that transforms text descriptions or 2D images into real, game-ready 3D models right on your desktop PC.

Unlike cloud services that require subscriptions, upload tokens, and queue wait times:
- **100% Local & Private:** All neural models run locally on your graphics card. No data leaves your machine.
- **Game-Engine Ready:** Automatically cleans and converts dense neural meshes into quad-dominant low-poly geometry (5K–15K faces) with unwrapped UVs and baked normal + diffuse textures.
- **Native Blender Integration:** Deep integration with Blender 5.2 LTS via a unified 10-panel studio addon.

---

## 2. Hardware & Software Requirements

| Component | Minimum Requirement | Recommended Specification (Target) |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit) or Linux (x86_64) | Windows 11 (64-bit) |
| **Graphics Card (GPU)** | NVIDIA GPU with 6 GB GDDR6 VRAM | NVIDIA RTX 3050 Laptop GPU (6 GB) / RTX 3060 / 4060 |
| **System Memory (RAM)** | 16 GB DDR4/DDR5 | 16 GB–32 GB DDR4/DDR5 |
| **Storage Space** | 10 GB free space (SSD recommended) | 25 GB free SSD space |
| **Blender** | Blender 4.2 LTS | Blender 5.2.3 LTS |

---

## 3. First-Time Installation & Startup

Everything in Local Mesh AI launches through the interactive master launcher:

```text
Local Mesh AI Root/
├── Start_Local_Mesh_AI.bat       <-- Master interactive user launcher
└── launchers/
    ├── 1_setup_environment.bat   <-- Environment setup & dependency installation
    ├── 2_detect_hardware.bat     <-- Verify GPU, CUDA, and Blender
    ├── 3_start_backend.bat       <-- Start FastAPI orchestrator (Port 8000)
    ├── 4_run_tests.bat           <-- Run automated test suites
    └── 5_start_studio_ui.bat     <-- Open Gradio 3D Studio (Port 7860)
```

### Quick 3-Step Setup:
1. **Double-click `Start_Local_Mesh_AI.bat`:** If the virtual environment (`.venv`) is not yet installed, the launcher prompts you to set it up automatically.
2. **Select Option `[1] Start Full Studio`:** Spawns both the FastAPI backend server (port 8000) and the Studio Workspace UI (port 7860) in parallel.
3. **Open your browser:** Navigate to **`http://127.0.0.1:7860`**.

---

## 4. Understanding the Two Creation Modes

### Mode A: Text-to-3D
* **When to use:** You have an idea and want the system to generate the visual concept and 3D mesh.
* **How it works:**
  1. You type a descriptive text prompt (e.g., *"a medieval wooden chest with iron hinges and carved lock"*).
  2. The local diffusion model (SD-Turbo) synthesizes isolated reference artwork.
  3. The reference image feeds the TripoSR neural reconstruction engine.
  4. The model is remeshed into clean quads with baked PBR materials.

### Mode B: Image-to-3D
* **When to use:** You already have an artwork, game sprite, concept drawing, or photo.
* **How it works:**
  1. Drag and drop your image file (PNG, JPG, or WEBP).
  2. **Strict Diffusion Bypass:** Diffusion is completely bypassed.
  3. The image is directly reconstructed into watertight 3D geometry.

---

## 5. Step-by-Step Tutorial: Your First 3D Project

Follow this 5-minute walkthrough to generate your very first 3D asset:

1. **Launch:** Run `Start_Local_Mesh_AI.bat` and select Option 1.
2. **Open UI:** Go to `http://127.0.0.1:7860`.
3. **Select Mode:** Choose **Mode A: Text-to-3D**.
4. **Enter Prompt:**
   ```text
   an old medieval wooden treasure chest with iron hinges and carved padlock, centered, studio lighting
   ```
5. **Configure:** Preset: `prop`, Quality: `Balanced`, Target Height: `1.0m`.
6. **Execute:** Click **`⚡ EXECUTE 3D PIPELINE`**.
7. **Inspect & Export:** Orbit the model in the WebGL viewport and export to GLB, OBJ, or FBX.

---

## 6. The 3D Viewport & Inspection Tools

- **Orbit Camera:** Left-click and drag.
- **Pan Camera:** Right-click and drag (or Shift + Left Click).
- **Zoom:** Mouse wheel up / down.
- **Reset Camera:** Returns to default 45° perspective.
- **Toggle Solid / Wireframe:** Inspect quad edge flow and polygon distribution.

---

## 7. Beginner's Guide to Modifiers & Settings

| Setting | Default Value | Plain-Language Description | When to Change It | Performance Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Asset Preset** | `prop` | Helps the AI scale and align orientation properly. | Change to `weapon` for swords/guns, `character` for figurines, `furniture` for chairs/tables. | No performance impact. |
| **Reconstruction Quality** | `Balanced` | Controls Marching Cubes grid resolution (Draft=128, Balanced=256, High=384). | Use `Draft` for rapid prototyping on 4GB–6GB GPUs. Use `High` for intricate, detailed hero assets. | Draft: ~15s, Balanced: ~35s, High: ~60s. |
| **Target Height (Meters)** | `1.8m` | Sets true real-world dimensions in meters. | Adjust to match real life (e.g. 0.3m for a potion bottle, 2.5m for a monster). | No performance impact. |
| **Sanitize Geometry** | `Checked` | Automatically caps open holes and deletes floating debris triangles. | Leave checked for watertight game meshes. | Adds ~1s to run. |
| **Target Quad Budget** | `8,000` | Target number of 4-sided faces for the retopologized game asset. | Increase to `15,000` for high-detail hero models; decrease to `3,000` for mobile games. | Higher budgets take slightly longer to bake. |
| **Bake Texture Resolution** | `2048` | Resolution of the baked Tangent Normal and Diffuse maps. | Use `1024` for lightweight assets; `4096` for 4K hero textures. | 2048: ~10s bake, 4096: ~25s bake. |

---

## 8. Blender 5.2 Studio Addon Suite

All Blender functionality is organized under a **single sidebar tab: `LOCAL MESH AI`**:

- **Panel 1 — Connection & Console:** Shows real-time connection, latency, and GPU telemetry. Includes **Start Backend Console** to launch `run_server.py`.
- **Panel 2 — Generate AI 3D Asset:** Text-to-3D and Image-to-3D generation inside Blender.
- **Panel 3 — Asset Browser:** Inspects, filters, and imports generated GLB assets into the active scene.
- **Panel 4 — Material Studio:** Assigns procedural Principled BSDF shaders (Wood, Metal, Fabric).
- **Panel 5 — Transform Calibration:** Enforces metric scale standards across asset presets.
- **Panel 6 — Scene Synchronization:** Synchronizes scene state with the backend server.
- **Panel 7 — Mesh Controls:** Applies subdivisions, bevels, normal recalculations, and manifold repairs.
- **Panel 8 — AI Technical Artist:** Decomposes assets into semantic components and executes conversational edits.
- **Panel 9 — 3D Finishing & Retopo:** QuadriFlow remeshing, Smart UV unwrapping, Cycles selected-to-active baking, LOD generation, and multi-format export.
- **Panel 10 — System Health & Learning:** System diagnostics, two-stage dry-run recovery previews with cryptographic verification, and model lifecycle management.

---

## 9. Exporting Assets to Unreal, Unity, and Godot

- **Unreal Engine 5:** Drag and drop `.glb` or `.fbx` into the Content Browser. UE5 automatically imports LOD hierarchies and materials.
- **Unity:** Place `.glb` or `.fbx` in `Assets/Models`. Unity glTFast or native FBX importer handles mesh and texture maps.
- **Godot 4:** Drag `.glb` directly into the FileSystem panel for native glTF 2.0 import.

---

## 10. Where Your Files Are Saved

```text
Local Mesh AI/
├── output/                     <-- Primary destination folder for exports
│   └── <job_id>/               <-- Per-job validated exports (GLB, FBX, OBJ)
└── data/
    ├── staging/                <-- Active session meshes & intermediate baking textures
    ├── models/                 <-- Neural network checkpoints (TripoSR, Qwen)
    ├── learning/               <-- Opt-in local instruction pairs
    └── recovery_audit.json     <-- Cryptographic log of approved cleanup actions
```

---

## 11. Troubleshooting & Common Errors

### Error: "CUDA Out of Memory (OOM)"
- **Fix:** In the Generation Shelf, change **Reconstruction Quality** to `Draft`. Local Mesh AI automatically isolates leases, but ensure no background games are consuming VRAM.

### Error: "Backend Offline" in Blender
- **Fix:** In Panel 1, click **Start Backend Console** or launch `Start_Local_Mesh_AI.bat` (Option 1).

### Error: "Preview Safe Recovery"
- **Fix:** In Panel 10, click **1. Preview Safe Recovery (Dry Run)** to inspect candidate files. Click **2. Execute Approved Recovery** to securely delete only reviewed staging files.

---

## 12. Frequently Asked Questions (FAQ)

**Q: Do I need to be connected to the internet to generate 3D models?**  
A: No! Once model checkpoints are provisioned, Local Mesh AI runs 100% offline.

**Q: Can I use the generated models commercially in my game?**  
A: Yes! Assets generated by TripoSR and the Auxiliary Technical Artist belong solely to you. See [MODEL_LICENSES.md](MODEL_LICENSES.md) for detailed terms.

**Q: How do I edit my 3D model after it is generated?**  
A: Use Panel 8 in Blender for natural-language component modifications (*"Make the backrest 20% taller"*) or edit vertices manually in Blender Edit Mode.

---

## 13. Beginner 3D Graphics Glossary

* **Mesh:** The 3D object constructed from vertices, edges, and polygonal faces.
* **QuadriFlow:** State-of-the-art quad remeshing creating clean 4-sided edge loops.
* **Selected-to-Active Baking:** Baking micro-details from a high-poly sculpt onto a clean low-poly mesh.
* **LOD (Level of Detail):** Hierarchical versions of a model with decreasing polygon counts for real-time performance.
* **PBR (Physically Based Rendering):** Modern shading standard simulating real-world light bounce.

---

## 14. Trust, Privacy & Governance

- **Privacy Policy:** [PRIVACY.md](PRIVACY.md) — 100% local inference guarantee.
- **Security Policy:** [SECURITY.md](SECURITY.md) — Vulnerability reporting and sandboxed AI execution.
- **Third-Party Notices:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — Open-source license attribution.
- **Model Licenses:** [MODEL_LICENSES.md](MODEL_LICENSES.md) — Upstream AI model terms.
- **Windows Packaging:** [docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md](docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md) — Release packaging specifications.
