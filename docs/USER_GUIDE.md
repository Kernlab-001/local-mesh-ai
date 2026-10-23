# Local Mesh AI — Complete Beginner User Manual & Operations Guide

Welcome to **Local Mesh AI**! This guide is designed for everyone—whether you are a 3D artist, game developer, hobbyist, or a complete beginner who has never run a Python script, touched an AI model, or opened Blender before.

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
8. [Exporting Assets & Opening in Blender, Unreal, and Unity](#8-exporting-assets--opening-in-blender-unreal-and-unity)
9. [Where Your Files Are Saved](#9-where-your-files-are-saved)
10. [Troubleshooting & Common Errors](#10-troubleshooting--common-errors)
11. [Frequently Asked Questions (FAQ)](#11-frequently-asked-questions-faq)
12. [Beginner 3D Graphics Glossary](#12-beginner-3d-graphics-glossary)

---

## 1. What is Local Mesh AI?

**Local Mesh AI** is an all-in-one local production tool that transforms text descriptions or 2D images into real, game-ready 3D models right on your desktop PC.

Unlike cloud services that require subscriptions, upload tokens, and queue wait times:
- **100% Local & Private:** All neural models run locally on your graphics card. No data leaves your machine.
- **Game-Engine Ready:** Automatically cleans and converts dense neural meshes into quad-dominant low-poly geometry (5K–15K faces) with unwrapped UVs and baked normal + diffuse textures.
- **Blender Integration:** Fully integrated with Blender for automated headless processing or interactive scene editing.

---

## 2. Hardware & Software Requirements

| Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit) or Linux (x86_64) | Windows 11 (64-bit) |
| **Graphics Card (GPU)** | NVIDIA GPU with 6 GB GDDR6 VRAM | NVIDIA RTX 3060 / 4060 or higher (8GB–12GB VRAM) |
| **System Memory (RAM)** | 16 GB DDR4/DDR5 | 32 GB DDR4/DDR5 |
| **Storage Space** | 10 GB free space (SSD recommended) | 25 GB free SSD space |
| **Blender (Optional)** | Blender 3.6 LTS | Blender 4.2 LTS or 5.2 |

> [!NOTE]
> If your GPU has less than 6 GB of VRAM, the application automatically uses **Draft Mode** to reduce memory consumption. If no NVIDIA GPU is detected, the system falls back to host CPU inference.

---

## 3. First-Time Installation & Startup

Everything in Local Mesh AI is configured to launch with simple double-click batch launchers located in the `launchers/` directory:

```
E:\GPUPLAT\launchers\
├── 1_setup_environment.bat   <-- Run once on fresh installation
├── 2_detect_hardware.bat     <-- Verify GPU, CUDA, and Blender
├── 3_start_backend.bat       <-- Start FastAPI orchestrator (Port 8000)
├── 4_run_tests.bat           <-- Run automated test suite
└── 5_start_studio_ui.bat     <-- Open Blender-inspired 3D Studio (Port 7860)
```

### Quick 3-Step Setup:
1. **Double-click `launchers/1_setup_environment.bat`:** This automatically creates your local Python virtual environment (`.venv`) and installs PyTorch CUDA, Diffusers, Trimesh, Gradio, and required libraries.
2. **Double-click `launchers/2_detect_hardware.bat`:** Scans your GPU and locates Blender on your system.
3. **Double-click `launchers/5_start_studio_ui.bat`:** Launches the 3D Studio workstation. Your default web browser will open automatically at **`http://127.0.0.1:7860`**.

---

## 4. Understanding the Two Creation Modes

Local Mesh AI has two dedicated creation paths. Choose the one that matches your starting material:

### Mode A: Text-to-3D
* **When to use:** You have an idea in your head and want the AI to design the object for you.
* **How it works:**
  1. You type a descriptive text prompt (e.g., *"an ancient dwarven warhammer with carved runes and polished steel head"*).
  2. The system executes a local diffusion model (SD-Turbo) to paint a clean concept art image on a neutral background.
  3. The generated artwork is automatically fed into the 3D neural reconstruction engine.
  4. The model is remeshed and textured.

### Mode B: Image-to-3D
* **When to use:** You already have a drawing, concept art, game sprite, or photo that you want to turn into a 3D model.
* **How it works:**
  1. You drag-and-drop or upload your image file (PNG, JPG, or WEBP).
  2. **Strict Diffusion Bypass:** The application skips the text diffusion stage completely.
  3. The image is processed directly into 4 canonical multi-angle projections and reconstructed into 3D geometry.

---

## 5. Step-by-Step Tutorial: Your First 3D Project

Follow this 5-minute walkthrough to generate your very first 3D asset:

### Step 1: Open the Application
Launch `launchers/5_start_studio_ui.bat` and navigate to `http://127.0.0.1:7860`.

### Step 2: Choose Creation Mode
In the left **Generation Shelf**, make sure **Mode A: Text-to-3D** is selected.

### Step 3: Enter a Beginner Prompt
Copy and paste this beginner-friendly prompt into the **Concept Description Prompt** box:
```text
an old medieval wooden treasure chest with iron hinges and carved padlock, centered, studio lighting
```

### Step 4: Choose Settings
- **Asset Preset:** Select `prop`.
- **Reconstruction Quality:** Select `Balanced`.
- **Target Height:** Leave at default `1.8m` (or adjust to `1.0m` for a chest).

### Step 5: Execute Generation
Click the prominent orange button: **`⚡ EXECUTE 3D PIPELINE`**.

### Step 6: Watch the Progress Timeline
At the bottom of the screen, watch the **10-Stage Pipeline Bar**:
1. *Validation* ➔ Checks prompt and settings.
2. *Concept Art* ➔ Paints reference image via SD-Turbo (~3s).
3. *Multi-View* ➔ Synthesizes 4 canonical viewing angles (~1s).
4. *3D Neural* ➔ Reconstructs volumetric geometry via TripoSR (~12s).
5. *Color Transfer* ➔ Calibrates colors from concept art (~1s).
6. *Quad Retopo* ➔ Decimates dense polygons to clean quads (~8s).
7. *UV Unwrap* ➔ Generates seamless UV islands (~4s).
8. *Texture Bake* ➔ Ray-casts normal map and diffuse map in Blender (~10s).
9. *Blender Scene* ➔ Assembles final `.blend` project and exports `.glb`.
10. *Completed* ➔ Asset is ready!

### Step 7: Inspect and Export
- Use your mouse in the central **3D Mesh Viewport** to orbit around your new model.
- Check the **Mesh Info** tab in the right panel to see real vertex and face counts.
- Click **Export GLB** or **Open in Blender** to inspect the file inside Blender!

---

## 6. The 3D Viewport & Inspection Tools

The central area is your interactive 3D inspection workspace:

- **Orbit Camera:** Click and drag with the Left Mouse Button.
- **Pan Camera:** Click and drag with the Right Mouse Button (or Shift + Left Click).
- **Zoom:** Scroll the Mouse Wheel up or down.
- **Reset Camera:** Click the `Reset Camera` button above the viewport to return to the default view.
- **Toggle Solid / Wireframe:** Click `Toggle Solid / Wireframe` to inspect the clean quad wireframe topology.
- **Inspection Tabs:**
  - `3D Mesh Viewport`: The live WebGL 3D model.
  - `360° Turntable`: Video turntable playback (if enabled).
  - `Source Concept`: The exact 2D image used to build the model.
  - `Multi-View Grid`: The 2x2 multi-angle projection sheet.

---

## 7. Beginner's Guide to Modifiers & Settings

Here is what every setting actually does and when you should change it:

| Setting | Default Value | Plain-Language Description | When to Change It | Performance Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Asset Preset** | `prop` | Helps the AI scale and align orientation properly. | Change to `weapon` for swords/guns, `character` for figurines, `furniture` for chairs/tables. | No performance impact. |
| **Reconstruction Quality** | `Balanced` | Controls Marching Cubes grid resolution (Draft=128, Balanced=256, High=384). | Use `Draft` for rapid prototyping on 4GB–6GB GPUs. Use `High` for intricate, detailed hero assets. | Draft: ~15s, Balanced: ~35s, High: ~60s. |
| **Target Height (Meters)** | `1.8m` | Sets true real-world dimensions in meters. | Adjust to match real life (e.g. 0.3m for a potion bottle, 2.5m for a monster). | No performance impact. |
| **Isolate Foreground (rembg)** | `Checked` | Strips out messy backgrounds from concept images. | Keep checked unless your uploaded image already has a transparent alpha channel. | Adds ~1s to run. |
| **Sanitize Geometry** | `Checked` | Automatically caps open holes and deletes floating debris triangles. | Leave checked for watertight game meshes. | Adds ~1s to run. |
| **Adaptive Vertex Welding** | `Checked` | Merges microscopic gaps in the neural mesh surface. | Leave checked to prevent cracked edges. | Adds ~0.5s to run. |
| **Reinhard Color Calibration** | `Checked` | Uses perceptual color math to match the 3D model's palette 1:1 with source artwork. | Keep checked to prevent muddy, desaturated colors. | Adds ~0.8s to run. |
| **Target Quad Budget** | `8,000` | Target number of 4-sided faces for the retopologized game asset. | Increase to `15,000` for high-detail hero models; decrease to `3,000` for mobile games. | Higher budgets take slightly longer to bake. |
| **Bake Texture Resolution** | `2048` | Resolution of the baked Tangent Normal and Diffuse maps. | Use `1024` for lightweight assets; `4096` for 4K hero textures. | 2048: ~10s bake, 4096: ~25s bake. |

---

## 8. Exporting Assets & Opening in Blender, Unreal, and Unity

When generation completes, you can export your asset into any industry-standard format:

### Export Formats:
- **GLB (Binary glTF):** Recommended for web, Godot, and quick inspection. Contains embedded textures and materials.
- **FBX (Autodesk Filmbox):** Recommended for Unreal Engine 5, Unity, and Autodesk Maya.
- **OBJ (Wavefront):** Universal format compatible with virtually all 3D software.
- **Blender Project (.blend):** Complete source scene containing the raw high-poly mesh, retopologized low-poly mesh, Cycles baking nodes, and camera setup.

### Opening in Blender:
Click the **`Open in Blender`** button in the right Properties panel. Local Mesh AI will launch your local Blender installation and load the project file directly!

---

## 9. Where Your Files Are Saved

All generation outputs are organized cleanly on your local hard drive:

```
E:\GPUPLAT\
├── output/                     <-- Primary destination folder
│   ├── meshes/                 <-- Exported GLB, FBX, OBJ files
│   └── project.blend           <-- Latest Blender project file
└── jobs/                       <-- Detailed per-job workspaces
    └── <job_id>/
        ├── input/              <-- Original prompt & concept artwork
        ├── multiview/          <-- 4-view canonical projections
        ├── high_poly_reference/<-- Raw neural geometry (immutable backup)
        ├── low_poly_asset/     <-- Retopologized game-ready model
        ├── textures/           <-- 16-bit Normal map & Diffuse map
        └── reports/
            └── manifest.json   <-- Machine-readable generation audit log
```

---

## 10. Troubleshooting & Common Errors

### Error: "CUDA Out of Memory (OOM)"
- **Cause:** Your GPU ran out of dedicated VRAM.
- **Fix:** In the Generation Shelf, change **Reconstruction Quality** to `Draft` and reduce **Target Quad Budget** to `5000`. Ensure other heavy applications (like games or 3D viewports) are closed.

### Error: "Blender executable not found"
- **Cause:** Blender is installed in a non-standard directory or not installed.
- **Fix:** Install Blender 4.2 LTS or 5.2 from [blender.org](https://www.blender.org/download/). If already installed in a custom location, add `BLENDER_PATH` to your Windows Environment Variables pointing to your `blender.exe`. *(Note: 3D generation still works without Blender; retopology and baking will simply be bypassed).*

### Error: "Concept prompt is required for Mode A"
- **Cause:** You clicked execute in Text-to-3D mode without entering a prompt.
- **Fix:** Type a description of what you want to create (at least 2 characters long).

### Error: "2D concept image is required for Mode B"
- **Cause:** You clicked execute in Image-to-3D mode without uploading an image.
- **Fix:** Drag and drop an image file (PNG, JPG, or WEBP) into the upload box.

---

## 11. Frequently Asked Questions (FAQ)

**Q: Do I need to be connected to the internet to generate 3D models?**  
A: No! Once the model weights are downloaded during your first run (~3.5 GB), Local Mesh AI runs 100% offline.

**Q: Why does Image-to-3D not use a text prompt?**  
A: Mode B is designed for artists who already have an image. It bypasses diffusion entirely so that your exact artwork is reconstructed into 3D without random AI changes.

**Q: Can I use the generated models commercially in my game?**  
A: Yes! The assets generated by TripoSR and SD-Turbo belong to you and can be exported for use in indie games, animations, and commercial projects.

**Q: How do I edit my 3D model after it is generated?**  
A: You can click `Open in Blender` to edit vertices, sculpt, or rig the model in Blender. You can also use the **AI Artist** tab in the right panel to type natural-language requests like *"Bevel edges by 2mm and set metallic material to gold"*.

---

## 12. Beginner 3D Graphics Glossary

* **Mesh:** The 3D digital object itself, constructed from vertices, edges, and faces that form a geometric shape.
* **Vertex (plural: Vertices):** A single point in 3D space with X, Y, and Z coordinates.
* **Face (Polygon):** The flat surface enclosed by edges. Modern game engines prefer **quads** (4-sided faces) or **triangles** (3-sided faces).
* **Topology:** The layout and organization of vertices and polygons across the surface of a 3D model. Clean topology consists of even, quad-dominant grid loops that bend and animate naturally without tearing.
* **High-Poly vs. Low-Poly:** High-poly meshes have hundreds of thousands of triangles to capture microscopic detail (too heavy for games). Low-poly meshes have fewer polygons (5K–15K) to run at high frame rates in real-time games.
* **Retopology:** The process of converting a messy, dense high-poly mesh into a clean, optimized low-poly mesh. Local Mesh AI does this automatically!
* **UV Mapping:** Unfolding the 3D surface of an object onto a flat 2D plane (like peeling an orange) so that 2D pictures (textures) can be painted onto it.
* **Normal Map:** A special purple-blue texture image that tricks the lighting engine into showing fine surface bumps, grooves, and micro-details without adding extra polygons.
* **Diffuse Map (Albedo):** The base color texture map of the object without baked lighting or shadows.
* **PBR (Physically Based Rendering):** A modern shading technique that simulates how light bounces off real-world materials like metal, wood, glass, and stone.
* **Baking:** Calculating the micro-details and colors from the high-poly mesh and transferring them directly into Normal and Diffuse 2D texture maps.
