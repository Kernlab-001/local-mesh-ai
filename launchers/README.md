# Local Mesh AI — Launcher & Startup Architecture

This directory contains individual modular launcher and diagnostic scripts for Local Mesh AI on Windows 11.

---

## 🚀 Canonical User Entry Point

The primary, authoritative end-user launcher is located at the root of the repository:

```bat
Start_Local_Mesh_AI.bat
```

For backwards compatibility with previous developer shortcuts, `launchers\Start_Local_Mesh_AI.bat` safely delegates execution to the root launcher.

---

## 📋 Master Launcher Menu Structure

The interactive master launcher provides 8 self-contained operations:

| Menu Item | Action | Description |
| :--- | :--- | :--- |
| **[1] Start Full Studio** | Backend + Blender Studio | Probes port 8000; launches FastAPI server if offline, verifies `/health`, and opens Blender with the addon loaded. |
| **[2] Launch Blender Studio** | Blender Studio Only | Discovers Blender 5.2 LTS (or 4.3/4.2) and opens it with `--addons-path addon`. Reuses existing backend if present. |
| **[3] Start Backend Server Only** | Port 8000 Orchestrator | Launches the FastAPI orchestration server in an isolated console window with parent watchdog support. |
| **[4] Start Standalone Studio UI** | Port 7860 Gradio UI | Launches the web-based Blender-inspired Studio workspace at `http://127.0.0.1:7860`. |
| **[5] Hardware Diagnostics** | GPU Preflight & Detection | Inspects NVIDIA GPU, physical VRAM, CUDA runtime, and model profile feasibility. |
| **[6] Run Comprehensive Test Suite** | Pytest Test Runner | Executes automated test suites covering recovery, AI models, integration, and Blender communication. |
| **[7] Setup / Repair Environment** | Dependency Sync | Verifies Python 3.12, creates `.venv`, and installs PyTorch CUDA 12.4 dependencies. |
| **[8] Exit** | Terminate Launcher | Closes the launcher cleanly without touching running background tasks. |

---

## 🛠️ Helper Scripts in `launchers/`

| Script | Purpose | Target User |
| :--- | :--- | :--- |
| `1_setup_environment.bat` | Automates virtual environment creation (`.venv`) and pip installation of PyTorch CUDA 12.4. | Setup / Recovery |
| `2_detect_hardware.bat` | Runs `detect_hardware.py` using `.venv` Python to output GPU, VRAM, and RAM telemetry. | Diagnostics |
| `3_start_backend.bat` | Direct launcher for FastAPI orchestration server on port 8000 without Blender. | Developer / Headless |
| `4_run_tests.bat` | Runs full automated test suite using pytest with verbose output. | CI / QA |
| `5_start_studio_ui.bat` | Starts standalone Gradio Studio workspace on port 7860. | Standalone UI |
| `generate_model.bat` | Drag-and-drop launcher: drag any image onto this script to generate a 3D model. | Quick Testing |
| `Start_Local_Mesh_AI.bat` | Compatibility wrapper redirecting to the root `Start_Local_Mesh_AI.bat`. | Compatibility |
