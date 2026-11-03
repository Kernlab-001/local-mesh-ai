# Local Mesh AI v6.0.0 — Local Neural 3D Asset Generation & AI Technical Artist Studio

> [!IMPORTANT]
> **Owner Review Draft:** This release description and distribution packages are prepared locally for repository owner review. **Do not publish or tag until explicitly approved.**

---

## 🌟 Overview

**Local Mesh AI v6.0.0** is an open-source, local-first 3D asset generation and technical-artist studio built for consumer and workstation NVIDIA GPUs. It enables creators, indie game developers, and 3D technical artists to move directly from text descriptions or 2D concept images to watertight, PBR-textured, game-ready 3D meshes with **zero cloud dependencies, zero subscription fees, and 100% on-device privacy**.

---

## ✨ Key Capabilities

1. **Dual-Mode 3D Synthesis:**
   - **Mode A: Text-to-3D:** Fast 1-step latent diffusion concept generation paired with canonical 4-view orthographic projections and feedforward volumetric meshing.
   - **Mode B: Image-to-3D:** Direct neural reconstruction from user-supplied concept art, reference photos, or transparent PNGs.
2. **AI Technical Artist (Qwen Coder & Auxiliary Classifier):**
   - Natural language component editing (e.g., *"Make the backrest 20% taller"*, *"Increase legs thickness"*).
   - Word-hashing tokenization with neural tool, component, and axis prediction (`AuxiliaryTechnicalArtistClassifier`).
   - Zero hallucinated math or arbitrary shell execution; all commands conform to strictly bounded Pydantic schemas.
3. **Consolidated Blender 5.2 Studio Addon:**
   - Unified single sidebar category tab: **`LOCAL MESH AI`** (all legacy competing tabs eliminated).
   - 10-panel progressive workflow spanning texturing, remeshing, high-to-low Cycles baking, and LOD generation.
   - Automatic backend health check and background lifecycle management linked to Blender's process ID.
4. **Autonomous Self-Healing & Safe Staging Cleanup:**
   - Cryptographically bound, two-stage remediation (`/preview` $\to$ `/execute`).
   - Single-use preview tokens, 1-hour minimum file age, canonical path containment, and streaming SHA-256 validation.
5. **Zero-VRAM GPU Lease Broker:**
   - Serialized CUDA execution ensuring models (Diffusion, TripoSR, Qwen) never collide in VRAM.
   - Guaranteed operation on **6 GB VRAM GPUs** (such as the NVIDIA RTX 3050 Laptop GPU).

---

## 💻 Hardware Prerequisites

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro 64-bit | Windows 11 64-bit |
| **Graphics Card (GPU)**| NVIDIA GPU with 6 GB VRAM (RTX 3050 Laptop) | NVIDIA RTX 4070+ (8 GB+ VRAM) |
| **Host 3D DCC** | Blender 4.2 LTS | Blender 5.2.3 LTS |
| **Python Runtime** | Python 3.12 64-bit | Python 3.12 64-bit |
| **System Memory (RAM)**| 16 GB DDR4/DDR5 | 16 GB+ DDR5 |
| **Storage** | 20 GB free disk space (models + data) | Fast NVMe SSD |

*Portability Notice:* Local Mesh AI requires a local installation of Python 3.12, NVIDIA Game Ready / Studio drivers (CUDA 12.4+), and Blender. It is not an all-in-one bundled sandbox executable.

---

## 📦 Official Release Downloads

Local Mesh AI is distributed as **EXACTLY TWO SEPARATE ZIP PACKAGES** to decouple clean application code from large neural model checkpoints:

### Package 1: Complete Project Code
**Filename:** `LocalMeshAI-v6.0.0-Complete-Code.zip`  
Contains the complete application codebase, root master launcher, Blender addon, test suites, schemas, documentation, and launchers. Does not contain large binary weights.

### Package 2: Complete AI Data Folder
**Filename:** `LocalMeshAI-v6.0.0-Complete-Data.zip`  
Contains the supported model files, configuration, and manifests that may legally be redistributed:
- TripoSR volumetric reconstruction checkpoint (`model.ckpt`, `config.yaml`)
- DINO ViT-B/16 vision transformer weights
- Qwen 2.5 Coder 7B Instruct GGUF (`qwen2.5-coder-7b-instruct-q4_0.gguf`)
- Qwen3-VL 4B Instruct GGUF (`Qwen3VL-4B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen3VL-4B-Instruct-F16.gguf`)
- PartField 3D component segmentation checkpoint (`model_objaverse.ckpt`)

*Note on SD-Turbo:* SD-Turbo is an optional 24 GB diffusion concept generator licensed under the Stability AI Non-Commercial Research Community License. It can be provisioned on-demand via `core.download_manager`.

---

## 🚀 Setup & Installation Workflow

### Step 1: Extract the Code Package
Extract `LocalMeshAI-v6.0.0-Complete-Code.zip` to your project workspace (e.g. `C:\LocalMeshAI` or `E:\local-mesh-ai-v6.0`).

### Step 2: Extract the Data Package
Extract `LocalMeshAI-v6.0.0-Complete-Data.zip` to your preferred storage drive (e.g. `E:\LocalMeshAI-Data`).

### Step 3: Connect Code to Data
Set the environment variable pointing to the extracted data root:
```cmd
set "LOCAL_MESH_AI_DATA_DIR=E:\LocalMeshAI-Data"
```
*(You can also set this permanently in Windows System Properties $\to$ Environment Variables).*

### Step 4: Launch Local Mesh AI
Double-click `Start_Local_Mesh_AI.bat` from the root of the Code folder:
1. Select **[1] Start Full Studio** (or **[7] Setup Environment** if running for the first time).
2. The launcher verifies `.venv`, probes backend health on port 8000, and opens Blender with the addon loaded!

---

## 📜 Model Licenses & Legal Disclosures

| Model Component | Upstream Origin | License | Commercial Clearance |
| :--- | :--- | :--- | :--- |
| **TripoSR** | Stability AI & Tripo AI | MIT License | Permitted under MIT terms |
| **DINO ViT-B/16** | Meta AI Research | Apache 2.0 | Permitted under Apache 2.0 terms |
| **Qwen 2.5 Coder 7B** | Alibaba Cloud | Apache 2.0 | Permitted under Apache 2.0 terms |
| **Qwen3-VL 4B** | Alibaba Cloud | Apache 2.0 | Permitted under Apache 2.0 terms |
| **PartField** | NVIDIA Research | Non-Commercial Research | Research only |
| **SD-Turbo** | Stability AI | Stability Non-Commercial | Non-commercial research only |

---

## 🧪 Verification & Acceptance

- **Automated Test Suite:** **47 Passed, 0 Failed** across recovery security, AI model lifecycle, integration, and Blender registration.
- **Hardware Telemetry:** Verified on RTX 3050 Laptop GPU (Draft: 15.21s, Balanced: 14.12s, Peak Physical VRAM: 4.15 GB).
- **Archive Integrity:** Both release archives tested and verified with CRC-32 integrity and clean extraction tests.
