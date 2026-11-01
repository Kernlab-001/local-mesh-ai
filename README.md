# Local Mesh AI 🚀
### Local-First Neural 3D Asset Generation & AI Technical Artist Studio

[![Tests](https://img.shields.io/badge/Tests-44%2F44%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-orange.svg)](https://pytorch.org/)
[![Blender](https://img.shields.io/badge/Blender-4.2%20%7C%205.2%20LTS-green.svg)](https://www.blender.org/)
[![Hardware Target](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%203050%20(6GB)-purple.svg)](#-target-hardware-and-environment)
[![Inference](https://img.shields.io/badge/Inference-100%25%20Local%20Offline-blueviolet.svg)](PRIVACY.md)

---

### 🧭 Product Navigation & Documentation

- **🚀 Quick Start:** [**Start Local Mesh AI**](Start_Local_Mesh_AI.bat) • [**Launchers Guide**](launchers/README.md) • [**Beginner Guide**](BEGINNER_GUIDE.md) • [**User Manual**](docs/USER_GUIDE.md)
- **⚖️ Trust & Governance:** [**Privacy Policy**](PRIVACY.md) • [**Security & Sandbox Safety**](SECURITY.md) • [**Code of Conduct**](CODE_OF_CONDUCT.md) • [**Third-Party Notices**](THIRD_PARTY_NOTICES.md) • [**Model Licenses Register**](MODEL_LICENSES.md) • [**Contributing Guide**](CONTRIBUTING.md)
- **🏗️ Release & Systems Architecture:** [**Windows Distribution Architecture**](docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md) • [**Model Provisioning Specification**](docs/MODEL_PROVISIONING_SPEC.md) • [**Freemium & Roadmap Proposal**](docs/FREEMIUM_ARCHITECTURE.md) • [**Multi-Engine Architecture**](docs/MULTI_ENGINE_ARCHITECTURE.md)
- **📜 Verified Milestone Reports:** [**Phase 9.1 Reliability Closure**](docs/PHASE_9_1_RELIABILITY_CLOSURE_REPORT.md) • [**Phase 9 Self-Healing & Learning**](docs/PHASE_9_SELF_HEALING_AND_CONTROLLED_LEARNING_REPORT.md) • [**Phase 7 3D Finishing & Retopo**](docs/PHASE_7_PROFESSIONAL_3D_FINISHING_REPORT.md) • [**Phase 6 AI Technical Artist**](docs/PHASE_6_ADVANCED_AI_TECHNICAL_ARTIST_REPORT.md) • [**Phase 5 Blender Studio**](docs/PHASE_5_BLENDER_STUDIO_REPORT.md) • [**Phase 4 Generation Pipeline**](docs/PHASE_4_GENERATION_PIPELINE_REPORT.md)

---

## 🌟 Executive Overview

**Local Mesh AI** is an open-source, local-first 3D asset generation, inspection, and AI technical-artist studio engineered specifically for consumer and workstation NVIDIA GPUs. It produces watertight, multi-material, PBR-textured 3D models directly from pure text descriptions or 2D concept images with **zero cloud dependencies, zero subscription fees, and 100% on-device privacy**.

Featuring a serialized Zero-VRAM GPU lease broker (`core.gpu_broker`), Local Mesh AI executes neural 3D reconstruction (TripoSR), semantic component decomposition, technical artist reasoning, and high-fidelity Cycles baking seamlessly on **6 GB VRAM GPUs** (such as the NVIDIA RTX 3050 Laptop GPU) up to multi-GPU workstations.

---

## 🎨 Genuine Production Output Showcase

Below is an authentic dining chair generated end-to-end via **Mode A: Text-to-3D** using the prompt:
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

## 🎯 Target Hardware and Environment

Local Mesh AI is engineered and verified for real consumer laptop and workstation hardware:

- **Operating System:** Windows 11 Home / Pro 64-bit (x86_64)
- **Primary GPU Target:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB physical GDDR6 VRAM)
- **System Memory:** 16 GB Physical RAM
- **Primary 3D Host:** Blender 5.2.3 LTS (compatible with 4.2+ LTS)
- **Python Runtime:** Python 3.12 64-bit with PyTorch 2.6+ (CUDA 12.4)
- **Default VRAM Profile:** Balanced Profile (`256` reconstruction res, 3B quantized LLM reasoning, chunked baking)

---

## 📊 Truthful Capability Ledger & Evidence Register

To maintain complete engineering honesty and eliminate false claims, every feature of Local Mesh AI is cataloged with an authoritative **Evidence Rating**:

| System / Feature | Description | Evidence Rating | Target Hardware Status (RTX 3050 6GB) |
|---|---|---|---|
| **Feedforward Mesh Generation** | Single-image to 3D watertight mesh via TripoSR | `VERIFIED_ON_TARGET_HARDWARE` | Fully operational (~1.8s inference, ~2.5 GB peak VRAM) |
| **Diffusion Concept Art** | 1-step text-to-concept synthesis via SD-Turbo | `VERIFIED_ON_TARGET_HARDWARE` | Fully operational (~0.8s inference, ~3.2 GB peak VRAM) |
| **Direct Concept Bypass** | Mode B image upload with strict diffusion bypass | `VERIFIED_ON_TARGET_HARDWARE` | Instant pass-through directly to 3D reconstruction |
| **GPU Lease Broker** | Cross-process atomic lock file with heartbeat | `VERIFIED_ON_TARGET_HARDWARE` | Verified across worker success, exception, and cancel |
| **VRAM Purge & Serialization** | Zero-VRAM overlap via torch.cuda.empty_cache() | `VERIFIED_ON_TARGET_HARDWARE` | Memory purges to baseline between engine phases |
| **Component Decomposition** | Connected components & PartField segmentation | `INTEGRATION_TESTED` | Decomposes assets into hierarchy nodes and bounding boxes |
| **AI Technical Artist (Reasoning)** | Natural language to structured ComponentEditPlan | `VERIFIED_ON_TARGET_HARDWARE` | Uses Qwen 2.5 Coder 3B GGUF; bounded JSON schema |
| **Auxiliary Technical Artist** | Neural classifier & bounded scale predictor | `VERIFIED_ON_TARGET_HARDWARE` | Auxiliary sequence module with trainable LoRA heads |
| **Centroid & Invariant Safety** | Tracks non-target component drift with auto-rollback | `INTEGRATION_TESTED` | Threshold checks protect unedited geometry |
| **Session Concurrency** | Thread-safe editing sessions via threading.RLock | `INTEGRATION_TESTED` | Concurrent requests serialized; no data corruption |
| **Server-Controlled Remediation** | Two-stage dry-run preview with SHA-256 validation | `INTEGRATION_TESTED` | Deletes only approved files matching exact hashes |
| **Coplanar Triangle Detection** | Möller-Trumbore 2D SAT intersection detection | `UNIT_TESTED` | Accurately identifies coplanar intersections |
| **UV Overlap & Manifold Checks** | Comprehensive geometric health scoring | `INTEGRATION_TESTED` | Penalizes overlapping islands; gates game-readiness |
| **QuadriFlow Retopology** | Automated Blender QuadriFlow quad remeshing | `VERIFIED_ON_TARGET_HARDWARE` | Headless & GUI Blender remeshing with sharp edge preservation |
| **PBR Material Prep & Presets** | Principled BSDF node trees (Wood, Metal, Fabric) | `VERIFIED_ON_TARGET_HARDWARE` | Authors valid multi-slot material graphs |
| **Cycles Selected-to-Active Bake**| High-to-low Normal, Diffuse, Roughness baking | `VERIFIED_ON_TARGET_HARDWARE` | Channel-specific emission baking with node graph restoration |
| **LOD Generation (LOD0 - LOD3)** | Edge-preserving decimation hierarchy | `VERIFIED_ON_TARGET_HARDWARE` | Produces 4 distinct LOD levels with naming conventions |
| **Game-Ready Multi-Export** | Simultaneous export of GLB, OBJ, and FBX | `VERIFIED_ON_TARGET_HARDWARE` | Validated by headless re-import test suite |
| **Blender Studio Addon UI** | 10-Panel workflow under single `LOCAL MESH AI` tab | `VERIFIED_ON_TARGET_HARDWARE` | Consolidated N-Panel interface; legacy tab eliminated |
| **Controlled Local Learning** | Opt-in instruction data collection & family partitions| `INTEGRATION_TESTED` | Disabled by default; zero external telemetry |
| **Candidate Model Lifecycle** | Gated candidate staging, evaluation, and rollback | `VERIFIED_ON_TARGET_HARDWARE` | Automated registration; atomic rollback verified |
| **7B Model Training on 6GB GPU** | Full 7B QLoRA fine-tuning on RTX 3050 | `UNSUPPORTED` | Rejected by hardware preflight (requires 12GB+ VRAM) |
| **Windows Desktop Installer (.exe)**| Self-contained Inno Setup deployment package | `PLANNED` | Architecture specified in docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md |
| **Cloud Multi-View Diffusion** | Distributed multi-GPU cloud synthesis | `PLANNED` | Proposed in docs/FREEMIUM_ARCHITECTURE.md |

---

## ⚡ Quick Start (Windows 11)

### Option 1: Master Launcher (Recommended)
Double-click `Start_Local_Mesh_AI.bat` in the repository root:
```cmd
Start_Local_Mesh_AI.bat
```
Select **[1] Start Full Studio** to launch both the FastAPI backend (port 8000) and the Studio Workspace UI (port 7860) simultaneously.

### Option 2: Blender Studio Addon
1. Open Blender 5.2 (or 4.2+ LTS).
2. Go to **Edit** $\rightarrow$ **Preferences** $\rightarrow$ **Add-ons** $\rightarrow$ **Install...**
3. Select `addon/triposr_blender_addon.py`.
4. Enable the checkbox for **Local Mesh AI Studio**.
5. Open the 3D Viewport sidebar (press `N`) and click the **`LOCAL MESH AI`** tab.
6. Click **Start Backend Console** to automatically launch the orchestration server.

---

## 🛡️ Privacy, Security & Data Sovereignty

- **100% Local Inference:** No prompts, images, meshes, or telemetry ever leave your machine. See [PRIVACY.md](PRIVACY.md).
- **Execution Sandboxing:** AI models cannot execute arbitrary Python or shell code; all operations are bounded to strongly typed Pydantic schemas. See [SECURITY.md](SECURITY.md).
- **Responsible Open-Source Licensing:** Core backend is Apache 2.0; Blender addon is GPLv3. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [MODEL_LICENSES.md](MODEL_LICENSES.md).

---

## 🧪 Comprehensive Verification Suite

Run all 44 unit, integration, and hardware acceptance tests with one command:
```cmd
.venv\Scripts\python.exe -m pytest tests\test_phase9_1_integration.py tests\test_phase9_1_acceptance.py tests\test_phase9_acceptance.py -v
```
*Current Suite Status:* **44 passed, 0 failed in 27.30 seconds.**
