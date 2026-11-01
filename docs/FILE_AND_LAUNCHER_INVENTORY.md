# Local Mesh AI — File & Launcher Inventory Audit

This document provides a comprehensive audit of all first-party batch scripts (`.bat`), Python utility scripts (`.py`), and helper scripts across the repository.

---

## 🚀 Root & Launcher Scripts

| Filename | Purpose | Caller / Entry Point | User Facing? | Dev / Test / Maint? | Verified Working? | Duplicates Another? | Status & Decision | Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`Start_Local_Mesh_AI.bat`** | Canonical root master launcher with interactive 8-option menu, health-check probing, and Blender launch. | End User / Explorer | **Yes** | User / Ops | **Yes** | No | **KEEP (Canonical Root)** | Primary end-user entry point for Windows 11; handles venv verification, port 8000 health polling, and Blender addon path arguments. |
| **`run_server.py`** | FastAPI orchestrator server entry point with native Windows parent-process watchdog. | `Start_Local_Mesh_AI.bat`, Blender addon | No | Runtime | **Yes** | No | **KEEP** | Essential backend server hosting endpoints, GPU broker, and job management. Exits cleanly when parent process closes. |
| **`run_master.py`** | CLI master pipeline runner for headless text-to-3D and image-to-3D generation. | CLI User / `generate_model.bat` | **Yes** | Dev / CLI | **Yes** | No | **KEEP** | Allows command-line batch asset generation without opening the Blender UI. |
| **`detect_hardware.py`** | Standalone hardware detection script reporting GPU, VRAM, RAM, and profile recommendations. | `2_detect_hardware.bat`, Master Launcher | **Yes** | Diagnostics | **Yes** | No | **KEEP** | Standalone diagnostic tool for user troubleshooting without requiring backend startup. |
| **`launchers\1_setup_environment.bat`** | Creates Python `.venv` and installs PyTorch CUDA 12.4 and project dependencies from `requirements.txt`. | Setup User / Master Launcher [7] | **Yes** | Setup | **Yes** | No | **KEEP** | Standard automated environment setup for Windows machines. |
| **`launchers\2_detect_hardware.bat`** | Executes `detect_hardware.py` using the `.venv` Python executable. | Diagnostics User / Master Launcher [5] | **Yes** | Diagnostics | **Yes** | No | **KEEP** | Modular diagnostic launcher wrapper. |
| **`launchers\3_start_backend.bat`** | Launches `run_server.py` in standalone console without Blender. | Headless User / Dev | **Yes** | Runtime | **Yes** | No | **KEEP** | Useful for developers debugging FastAPI routes and endpoints independently. |
| **`launchers\4_run_tests.bat`** | Runs full test suite using `pytest -v`. | QA / CI / Dev | No | Testing | **Yes** | No | **KEEP** | Standard 1-click test execution shortcut for Windows developers. |
| **`launchers\5_start_studio_ui.bat`** | Starts the standalone Gradio Studio workspace at `http://127.0.0.1:7860`. | Web UI User | **Yes** | Standalone UI | **Yes** | No | **KEEP** | Supports running Local Mesh AI in browser without Blender installed. |
| **`launchers\generate_model.bat`** | Drag-and-drop Windows shortcut: drops image onto bat to invoke `run_master.py`. | End User | **Yes** | Tool | **Yes** | No | **KEEP** | Convenient drag-and-drop workflow for casual 3D asset generation from 2D images. |
| **`launchers\Start_Local_Mesh_AI.bat`** | Wrapper redirecting to root `Start_Local_Mesh_AI.bat`. | Legacy Users | **Yes** | Compatibility | **Yes** | Root launcher | **KEEP (Compatibility)** | Retained as a thin 5-line wrapper to avoid breaking existing shortcuts or muscle memory. |

---

## 🛠️ Utility & Maintenance Scripts (`scripts/`)

| Filename | Purpose | Caller | User Facing? | Dev / Test / Maint? | Verified Working? | Decision | Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **`scripts\download_models.py`** | Downloads and provisions TripoSR, DINO, and Qwen weights into external model directory. | CLI / Admin | **Yes** | Maintenance | **Yes** | **KEEP** | Used for automated provisioning of initial model weights. |
| **`scripts\verify_ai_models.py`** | Offline verification of model checkpoints, SHA-256 hashes, and local cache availability. | Developer / Test | No | Maintenance | **Yes** | **KEEP** | Pre-flight validation ensuring models are present prior to running inference. |
| **`scripts\verify_real_generation.py`** | Executes an end-to-end generation test on host hardware and records timings. | Benchmark / CI | No | Testing | **Yes** | **KEEP** | Regression test verifying the full pipeline works on actual GPU. |
| **`scripts\test_qwen_planner.py`** | Standalone test script for Qwen Coder planning engine and grammar validation. | Developer | No | Testing | **Yes** | **KEEP** | Unit verification of LLM planner parsing without starting server. |
| **`scripts\test_triposr_workflow.py`** | Standalone test for TripoSR neural reconstruction from image inputs. | Developer | No | Testing | **Yes** | **KEEP** | Verifies TripoSR CUDA execution independently. |
| **`scripts\build_release_archives.py`** | Automated release builder producing Code ZIP, Data ZIP, and SHA-256 checksums. | Release Engineer | No | Packaging | **Yes** | **KEEP** | Required release packaging tool for the two-package distribution architecture. |

---

## 🎨 Presentation & Validation Scripts (`docs/presentation/`)

| Filename | Purpose | Caller | User Facing? | Dev / Test / Maint? | Verified Working? | Decision | Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **`build_presentation.py`** | Compiles the official 16:9 widescreen PowerPoint deck (`Local_Mesh_AI_Project_Presentation.pptx`). | Marketing / Pitch | No | Presentation | **Yes** | **KEEP** | Generates authoritative presentation deck containing real generation showcase slides. |
| **`run_validation_text23d.py`** | Validation script generating test assets and benchmarks for Text-to-3D mode. | Benchmark | No | Testing | **Yes** | **KEEP** | Produces validation renders and runtime logs. |
| **`run_validation_image23d.py`** | Validation script generating test assets and benchmarks for Image-to-3D mode. | Benchmark | No | Testing | **Yes** | **KEEP** | Produces validation renders and runtime logs. |

---

## 🔬 Temporary Developer Scripts (`scratch/`)

| Filename | Purpose | Caller | Working? | Decision | Rationale |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `scratch\add_aliases.py` | One-off migration script for environment variable aliases. | Migration | Obsolete | **MAINTAIN IN SCRATCH** | Scratch directory is preserved for historical reference; excluded from Code ZIP. |
| `scratch\add_pf_alias.py` | One-off migration script for PartField paths. | Migration | Obsolete | **MAINTAIN IN SCRATCH** | Historical developer scratch script; excluded from release packages. |
| `scratch\apply_vlm_loop.py` | Standalone test script for vision-language loop experiments. | Developer | Experimental | **MAINTAIN IN SCRATCH** | Excluded from release package. |
| `scratch\test_loop_live.py` | One-off live testing script for correction loops. | Developer | Experimental | **MAINTAIN IN SCRATCH** | Excluded from release package. |
| `scratch\update_readme.py` | Migration utility for Markdown banners. | Developer | Obsolete | **MAINTAIN IN SCRATCH** | Excluded from release package. |

---

## 📦 Summary of Actions Taken
- **Retained All Essential Launchers:** Every required `.bat` and `.py` script in the root and `launchers/` directory is documented, verified, and preserved.
- **Fixed Root Master Launcher:** Corrected the batch invocation bug, implemented live port 8000 health polling, and aligned menu options 1–8 per release specification.
- **Strict Release Separation:** `scratch/` files are cleanly excluded from the release Code ZIP package via explicit exclusion rules in `scripts/build_release_archives.py`.
