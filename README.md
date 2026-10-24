# Local Mesh AI 🚀
### Local-First Neural 3D Asset Generation & AI Technical Artist Studio

[![Tests](https://img.shields.io/badge/Tests-79%2F79%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-orange.svg)](https://pytorch.org/)
[![Blender](https://img.shields.io/badge/Blender-4.2%20%7C%205.2%20LTS-green.svg)](https://www.blender.org/)
[![Inference](https://img.shields.io/badge/Inference-100%25%20Local%20GPU-blueviolet.svg)](#-zero-vram-overlap--memory-broker)
[![Presentation](https://img.shields.io/badge/Presentation-12--Slide%20Deck%20(16%3A9)-orange.svg)](docs/presentation/Local_Mesh_AI_Project_Presentation.pptx)

---

### 🧭 Navigation & Deliverables
[**📖 Beginner Guide**](BEGINNER_GUIDE.md) • [**📘 User Manual**](docs/USER_GUIDE.md) • [**🛠️ Contributing**](CONTRIBUTING.md) • [**📊 Presentation Deck (.pptx)**](docs/presentation/Local_Mesh_AI_Project_Presentation.pptx) • [**🎤 Presenter Notes**](docs/presentation/Presenter_Notes.md) • [**📋 Demo Checklist**](docs/presentation/Live_Demo_Checklist.md) • [**📈 Release Validation Report**](docs/presentation/Release_Validation_Report.md) • [**📜 License**](LICENSE)

---

## 🌟 Executive Overview

**Local Mesh AI** is an open-source, local-first 3D asset generation, inspection, and AI technical-art studio engineered for workstation and consumer NVIDIA GPUs. It produces watertight, multi-material, PBR-textured 3D models directly from pure text descriptions or 2D concept images with **zero cloud dependencies, zero subscription fees, and 100% local privacy**.

Featuring a serialized Zero-VRAM GPU lease broker, Local Mesh AI runs full neural reconstruction and high-fidelity texture baking seamlessly on **6 GB VRAM GPUs** (such as the RTX 3050 Laptop) up to multi-GPU workstations.

---

## 🎨 Genuine Production Output Showcase

Below is an authentic dining chair generated completely end-to-end via **Mode A: Text-to-3D** using the prompt:
> *"a dining chair with a dark walnut-brown wooden frame and legs, an ivory-cream fabric upholstered seat cushion, and a matching cream tufted backrest"*

| High-Fidelity EEVEE Viewport Render | 4-View Canonical Projections |
| :---: | :---: |
| ![Local Mesh AI Chair EEVEE Render](docs/assets/chair_render_eevee.png) | ![Local Mesh AI Canonical Multi-View Grid](docs/assets/multiview_grid.png) |
| *Blender 5.2 EEVEE render: Distinct dark walnut wood frame and ivory-cream tufted upholstery (52,552 vertices, 3 PBR material slots).* | *Canonical orthographic projections (0°, 90°, 180°, 270°) feeding the TripoSR neural reconstruction engine.* |

| 2D Diffusion Concept Artwork | Procedural PBR Diffuse Maps (Wood & Fabric) |
| :---: | :---: |
| ![Concept Artwork](docs/assets/concept_art_chair.png) | ![Wood and Fabric Textures](docs/assets/tex_wood_diffuse.png) |
| *SD-Turbo 1-step concept synthesis on isolated studio background.* | *Authoritative 1024×1024 diffuse, normal, and roughness maps embedded in final GLB.* |

---

## 🌟 Core Capabilities

* **⚡ Dual-Mode 3D Generation:**
  * **Mode A (Text-to-3D):** Synthesizes crisp concept artwork via local SD-Turbo diffusion, projects canonical orthographic views, and reconstructs 3D geometry with automated appearance analysis.
  * **Mode B (Image-to-3D):** Direct upload of user concept art with **strict diffusion bypass**—reconstructing 3D geometry directly from source pixels.
* **🎨 Multi-Material PBR Shader Synthesis:**
  * Extracts distinct semantic material regions via NLP prompt analysis and CIELAB color clustering.
  * Authors dedicated Blender Principled BSDF material slots (`m_wood_frame`, `m_fabric_cushion`, `m_fabric_backrest`).
  * Generates 1024×1024 Diffuse, 16-bit Normal, and Roughness texture maps baked directly into the exported GLB.
* **📐 Preset-Aware Canonical Scale Calibration:**
  * Enforces real-world dimensional accuracy across asset classes:
    * **Furniture / Chairs:** `0.90 m` canonical height
    * **Weapons / Swords / Axes:** `1.00 m` canonical length
    * **Characters / Humanoids:** `1.80 m` canonical height
    * **Vehicles / Karts:** `1.50 m` canonical height
    * **Props / Items:** `0.80 m` canonical height
  * Eliminates oversized neural reconstructions while strictly preserving aspect ratios, UV coordinates, and vertex normals.
* **🛡️ Zero-VRAM Overlap & Memory Broker:**
  * Serializes model residency (`NONE` $\rightarrow$ `SD_TURBO` $\rightarrow$ `TRIPOSR` $\rightarrow$ `BLENDER`), purging VRAM to **0.0 MB** between stages.
  * Cross-process atomic file and thread leases eliminate Out-Of-Memory exceptions on 6GB–8GB GPUs.
* **🧠 AI Technical Artist (Qwen 2.5 Coder 7B):**
  * Natural language 3D scene editing (*"Make the frame polished brass metallic and bevel edges by 2mm"*).
  * Enforces strict Pydantic command schemas with **zero arbitrary code execution**.
* **🔌 Native Blender 4.2 / 5.2 LTS Add-on:**
  * Sidebar N-Panel extension with non-blocking async background timers (`bpy.app.timers`).
  * One-click terminal auto-launcher, isolated collection management (`AI_Generated_Assets`), and automatic viewport material preview.
* **📦 AAA Quad Retopology & UV Unwrapping:**
  * Automatic quad-dominant decimation targeting 8,000 polygons, smart UV unwrapping (0.02 island margin), and Cycles ray-cast normal/diffuse map baking.

---

## 📥 Downloads & Release Bundles

Official packaged release archives are available under [**GitHub Releases**](https://github.com/Kernlab-001/local-mesh-ai/releases):

* 📦 **Production Bundle:** **`local-mesh-ai-v4.0.6.zip`** (40.7 MB)
  * **SHA-256 Checksum:** `BF6B255FAFDF9FB8CE37CEAB3E052AB252562A5BD62C417FB2570D0AB604426A`
  * Contains the entire application, batch runners, Blender add-on, test suites, and documentation.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Clients["User Interfaces"]
        C1["Blender 4.2 / 5.2 LTS N-Panel Add-on"]
        C2["Blender-Inspired Studio UI (Port 7860)"]
        C3["Direct REST API (Port 8000)"]
    end

    subgraph Orchestration["Local Orchestration & Concurrency"]
        DET["Hardware Auto-Detector & Diagnostics"]
        API["FastAPI Orchestrator (run_server.py)"]
        JM["Async Job Manager & Production Manifest"]
        BROKER["Cross-Process GPU Lease Broker (gpu_broker.py)"]
        LIFECYCLE["Zero-VRAM Sequential Model Manager"]
    end

    subgraph GPU_Workloads["Serialized Zero-VRAM Inference Pipeline"]
        SDT["Mode A: SD-Turbo Concept Diffusion"]
        PURGE0["VRAM Purge & torch.cuda.empty_cache()"]
        MV["Multi-View Canonical Projections (0°/90°/180°/270°)"]
        TSR["TripoSR Neural 3D Engine (mc_resolution=256)"]
        PURGE1["VRAM Purge & Garbage Collection"]
    end

    subgraph Appearance_Geometry["Appearance, Scale & PBR Engine"]
        ANALYZER["Appearance Analyzer (NLP & CIELAB Clustering)"]
        SCALE["Preset-Aware Scale Calibration (Canonical Dimensions)"]
        SYNTH["Material Synthesizer & Procedural Texture Engine"]
        RETOPO["AAA Quad Remeshing (8,000 Quad Target)"]
        UV["Smart UV Unwrapping (0.02 Island Margin)"]
        BAKE["Cycles Ray-Cast Normal & Diffuse Map Bake"]
    end

    subgraph Technical_Artist["AI Scene Editor"]
        QWEN["Qwen 2.5 Coder 7B (GGUF / llama-cpp)"]
        ALLOW["Strict Pydantic Command Schemas"]
        EXEC["Deterministic Blender Operator Executor"]
    end

    Clients --> API
    DET --> API
    API --> JM --> BROKER
    BROKER --> LIFECYCLE
    LIFECYCLE -->|Stage 1: Concept Art| SDT --> PURGE0
    PURGE0 --> MV --> TSR --> PURGE1
    PURGE1 -->|Stage 2: Material & Scale| ANALYZER --> SCALE --> SYNTH
    SYNTH -->|Stage 3: Headless Blender| RETOPO --> UV --> BAKE
    BAKE -->|Production GLB Asset| Clients
    Clients -.->|Stage 4: AI Scene Modification| QWEN --> ALLOW --> EXEC
```

---

## 🚀 Quickstart Guide

### 1. System Requirements
* **Operating System:** Windows 10/11 (64-bit), Linux (x86_64), or macOS.
* **Graphics Card (GPU):** NVIDIA GPU with 6GB+ VRAM (CUDA 12+) recommended; CPU inference mode supported for basic testing.
* **Blender:** Blender 4.2 LTS or Blender 5.2 LTS installed.
* **Python:** Python 3.10 – 3.12.

### 2. 1-Click Windows Setup (`launchers/` folder)

1. **Step 1 — Environment Setup:**
   Double-click [`launchers/1_setup_environment.bat`](launchers/1_setup_environment.bat).
   * Automatically initializes the `.venv` virtual environment and installs PyTorch CUDA 12.4, diffusers, trimesh, and requirements.
2. **Step 2 — Hardware Detection:**
   Double-click [`launchers/2_detect_hardware.bat`](launchers/2_detect_hardware.bat).
   * Probes GPU model, physical VRAM, CUDA capabilities, and auto-locates Blender executable.
3. **Step 3 — Launch 3D Studio Workstation:**
   Double-click [`launchers/5_start_studio_ui.bat`](launchers/5_start_studio_ui.bat).
   * Spawns the studio workspace at `http://127.0.0.1:7860`.

*(For Linux / macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 detect_hardware.py`)*

---

## 🎨 Native Blender Add-on Setup

### Installation in Blender (4.2 LTS & 5.2 LTS)
1. Open **Blender 5.2** (or 4.2 LTS).
2. Go to **Edit > Preferences > Add-ons**.
3. In the upper-right corner, click **Dropdown Arrow (⌄) > Install from Disk...**
4. Select [`addon/triposr_blender_addon.py`](addon/triposr_blender_addon.py) and enable **Local Neural 3D Studio**.

### In-Blender Generation Workflow
1. In the 3D Viewport, press **`N`** to reveal the sidebar $\rightarrow$ select the **`AI 3D Generator`** tab.
2. **Server & Path Settings (One-time):** Expand the panel and ensure **Repo Root** points to your `local-mesh-ai` folder (persisted in preferences).
3. **⚡ Zero-Click Terminal Auto-Launcher:** Clicking **⚡ Generate 3D Asset** automatically verifies if the background server is running; if not, it launches the server terminal window automatically!
4. Choose **Prompt to 3D** or **Image to 3D**, enter your prompt or select an image, and click generate.
5. Assets stream directly into an isolated `AI_Generated_Assets` collection with fully wired PBR shaders, centered pivots, and authentic materials.

---

## 📊 Hardware Benchmarks & Performance Profiles

| Hardware Profile | Target GPUs / Hardware | VRAM / RAM | Quality Profile | Recommended Local LLM | Reconstruction | Total End-to-End |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Workstation / Enthusiast** | NVIDIA RTX 4090, 3090, RTX 6000 Ada, A5000 / A6000 | 24 GB – 48 GB | High (`res=384, chunk=4096`) | **Qwen 2.5 Coder 14B / 32B** | ~8–10s | **~12–15s** |
| **Pro / Performance** | NVIDIA RTX 4080, 4070 Ti Super, AMD RX 7900 XTX | 16 GB | High (`res=384, chunk=4096`) | **Qwen 2.5 Coder 14B** | ~10–12s | **~15–18s** |
| **Upper Mid-Range** | NVIDIA RTX 4070, 3080, 4060 Ti 16GB, AMD RX 7800 XT | 12 GB – 16 GB | Balanced (`res=256, chunk=8192`) | **Qwen 2.5 Coder 7B** | ~11–13s | **~18–22s** |
| **Mainstream Desktop** | NVIDIA RTX 4060, 3060 12GB, 2070 Super, Intel Arc A770 | 8 GB – 12 GB | Balanced (`res=256, chunk=8192`) | **Qwen 2.5 Coder 7B** | ~12–15s | **~20–24s** |
| **Entry-Level / Laptop** | NVIDIA RTX 3050 Laptop / Desktop (6GB), RTX 2060 (6GB) | 6 GB | Balanced (`res=256, chunk=8192`) | **Qwen 2.5 Coder 7B** | ~14–18s | **~28–32s** |
| **Budget / Handheld / Mac** | NVIDIA RTX 3050 (4GB), GTX 1650, Steam Deck, Apple M1/M2/M3 | 4 GB VRAM / Unified | Draft (`res=128, chunk=8192`) | **Qwen 2.5 Coder 3B / 1.5B** | ~10–12s | **~24–28s** |
| **CPU / Integrated Mode** | Intel Core i5/i7/i9 (12th–14th Gen), AMD Ryzen 5/7/9 (AVX2) | 16 GB+ System RAM | Draft (`res=128, chunk=8192`) | **Qwen 2.5 Coder 1.5B** | ~40–55s | **~60–75s** |

---

## 🧪 Automated Testing & Verification

The test suite covers API routing, multi-material shader synthesis, color calibration math, GPU concurrency leasing, geometry sanitization, and UI event binding:

```bash
# Windows
launchers\4_run_tests.bat

# Linux / macOS
pytest tests -v
```

### Test Suite Execution Summary
* **Total Tests Collected:** 79
* **Passed:** 79 (100% pass rate)
* **Failed:** 0
* **Skipped:** 0

| Test Module | Tests | Description |
| :--- | :---: | :--- |
| `tests/test_api.py` | 11 | REST endpoints, health check, job status polling |
| `tests/test_color_and_materials.py` | 12 | PBR material slots, texture generation, UV unwrap, GLB appearance |
| `tests/test_color_calibrator.py` | 2 | Perceptual $L^*a^*b^*$ Reinhard color distribution math |
| `tests/test_command_schemas.py` | 4 | Strict Pydantic command schemas for AI scene edits |
| `tests/test_dependency_verification.py` | 6 | PyTorch 2.6.0+cu124, torchvision, CUDA driver readiness |
| `tests/test_dual_mode_pipeline.py` | 4 | Mode A (Text-to-3D) and Mode B (Image-to-3D bypass) lifecycle |
| `tests/test_gpu_broker.py` | 4 | Cross-process atomic GPU locks and concurrency serialization |
| `tests/test_hardware_detector.py` | 4 | GPU VRAM profiling, CUDA capability check, Blender path resolution |
| `tests/test_image_validation.py` | 3 | Aspect ratio, alpha transparency, and image format sanitization |
| `tests/test_job_lifecycle.py` | 3 | Asynchronous queue state progression and cancel handling |
| `tests/test_mesh_serving_and_viewer.py` | 5 | glTF/GLB MIME types, Three.js viewer embedding, error states |
| `tests/test_onboarding_and_diagnostics.py` | 7 | First-launch tour, environment checks, diagnostic reports |
| `tests/test_production_pipeline.py` | 3 | End-to-end multi-view synthesis and decimation |
| `tests/test_prompt_engine.py` | 2 | 77-token CLIP sequence management and SD-Turbo prompt cleaning |
| `tests/test_ui_dashboard.py` | 9 | Blender design system tokens, wireframe toggles, preset auto-calibration |

---

## 📁 Repository Structure

```text
local-mesh-ai/
├── BEGINNER_GUIDE.md           # Beginner guide, step-by-step tutorial & 3D glossary
├── CONTRIBUTING.md             # Developer setup, coding standards, and PR guide
├── LICENSE                     # Apache 2.0 Open Source License
├── config.example.json         # Reference configuration template
├── detect_hardware.py          # Dynamic GPU & environment auto-detection
├── run_server.py               # Starts FastAPI local orchestration backend (Port 8000)
├── run_master.py               # Master CLI orchestration runner
├── requirements.txt            # Python dependencies
│
├── docs/                       # Official documentation & presentation assets
│   ├── USER_GUIDE.md           # Comprehensive operations manual
│   ├── assets/                 # High-resolution genuine renders and textures
│   │   ├── chair_render_eevee.png
│   │   ├── multiview_grid.png
│   │   ├── concept_art_chair.png
│   │   ├── tex_wood_diffuse.png
│   │   └── tex_fabric_diffuse.png
│   └── presentation/           # Technical presentation deliverables
│       ├── Local_Mesh_AI_Project_Presentation.pptx  # 12-slide widescreen deck (5.0 MB)
│       ├── Presenter_Notes.md                       # Verbatim presentation script
│       ├── Live_Demo_Checklist.md                   # 7-step fail-safe live demo guide
│       ├── Project_Facts.md                         # Authoritative technical facts ledger
│       ├── Release_Validation_Report.md             # Senior QA validation audit
│       └── Release_Demo_Results.md                  # Measured runtime telemetry
│
├── launchers/                  # 1-Click numbered Windows batch runners
│   ├── 1_setup_environment.bat # Creates .venv & installs all dependencies
│   ├── 2_detect_hardware.bat   # Auto-detects GPU, VRAM & Blender path
│   ├── 3_start_backend.bat     # Launches local orchestration server
│   ├── 4_run_tests.bat         # Runs 79 automated test suites
│   └── 5_start_studio_ui.bat   # Launches Blender-inspired 3D Studio workstation
│
├── core/
│   ├── appearance_analyzer.py  # NLP material extraction & CIELAB color clustering
│   ├── material_synthesizer.py # Smart UV unwrapping & polygon slot partitioning
│   ├── texture_generator.py    # Procedural 1024x1024 diffuse, normal, & roughness maps
│   ├── prompt_to_image.py      # Real-time concept image synthesis (SD-Turbo)
│   ├── triposr_engine.py       # Validated TripoSR neural reconstruction manager
│   ├── color_calibrator.py     # Perceptual Lab Reinhard color transfer engine
│   ├── multiview_synthesizer.py# Canonical 4-view projection & 2x2 grid engine
│   ├── geometry_sanitizer.py   # Preset-aware canonical scaling & mesh repair
│   ├── gpu_broker.py           # Cross-process GPU lease broker (atomic lock)
│   ├── model_lifecycle.py      # Zero-VRAM sequential model manager
│   ├── qwen_planner.py         # Qwen 2.5 Coder natural language planner
│   ├── command_schemas.py      # Strict Pydantic command schemas (zero arbitrary code)
│   └── blender_executor.py     # Deterministic Blender Python executor
│
├── bridge/
│   ├── blender_bridge.py       # Headless Blender scene integration & scale calibration
│   └── retopology_and_baker.py # AAA quad remeshing, UV unwrap & Cycles ray baking
│
├── api/
│   ├── app.py                  # FastAPI application definition
│   ├── routes.py               # /health, /jobs, /jobs/{id}, /edit, /export
│   └── schemas.py              # Pydantic request & response schemas
│
├── services/
│   ├── job_manager.py          # Async job queue, timing metrics & manifest writer
│   └── export_service.py       # GLB, FBX, OBJ export pipeline (with timeouts)
│
├── addon/
│   └── triposr_blender_addon.py # Native Blender 4.2 / 5.2 N-Panel extension
│
├── ui/
│   └── dashboard.py            # Blender-inspired dark 3D Studio workstation (Port 7860)
│
└── tests/                      # 79 unit, integration, and UI test suites
```

---

## 📜 License & Model Attributions

* **Application Code:** Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE).
* **TripoSR:** Neural 3D reconstruction engine licensed under the **MIT License** by VAST-AI Research & Stability AI.
* **SD-Turbo:** 1-step concept synthesis model distributed under the **Stability AI Community License** (research/non-commercial; commercial use requires a Stability AI license).
* **Qwen 2.5 Coder:** LLM planning weights licensed under the **Tongyi Qianwen License Agreement**.
