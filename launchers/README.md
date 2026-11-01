# Local Mesh AI — Launcher Suite & Execution Reference

This directory contains the operational and development launchers for **Local Mesh AI** on Windows 11.

---

## Quick Start (End Users)

Double-click `Start_Local_Mesh_AI.bat` (in the root directory or here) to launch the unified interactive console:

```cmd
Start_Local_Mesh_AI.bat
```

This master launcher provides menu-driven access to:
1. **Full Studio** (spawns both Backend Server and Studio UI in parallel).
2. **Backend Server Only** (FastAPI orchestration on port 8000).
3. **Studio UI Only** (Blender-themed workspace on port 7860).
4. **Blender Studio Launch** (launches Blender with addon path preconfigured).
5. **Hardware Preflight & Diagnostics** (evaluates GPU VRAM, CUDA compatibility, and leasing state).
6. **Comprehensive Test Suite** (runs pytest integration and acceptance tests).
7. **Environment Setup & Repair** (synchronizes virtual environment dependencies).

---

## Developer Scripts Reference

| Script | Purpose | Port / Target | Prerequisites |
|---|---|---|---|
| `Start_Local_Mesh_AI.bat` | Master interactive user launcher | Ports 8000 & 7860 | `.venv` |
| `1_setup_environment.bat` | Creates Python 3.12 virtual environment, installs PyTorch CUDA 12.1, and project requirements | `.venv/` | Python 3.12 64-bit on PATH |
| `2_detect_hardware.bat` | Queries `nvidia-smi` and PyTorch to verify GPU presence and VRAM | CLI stdout | NVIDIA Driver |
| `3_start_backend.bat` | Runs FastAPI server (`run_server.py`) directly with Uvicorn | Port 8000 | `.venv` |
| `4_run_tests.bat` | Runs standard pytest suites across integration and acceptance specs | pytest CLI | `.venv` |
| `5_start_studio_ui.bat` | Runs standalone Gradio studio dashboard (`ui/dashboard.py`) | Port 7860 | `.venv` |
| `generate_model.bat` | Direct CLI single-command asset generation | CLI -> `output/` | Running backend or local models |

---

## Port Allocations

- **Port 8000**: FastAPI Orchestration Server (`http://127.0.0.1:8000`)
  - Interactive Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
  - OpenAPI Specification: `http://127.0.0.1:8000/openapi.json`
- **Port 7860**: Gradio 3D Studio Workspace UI (`http://127.0.0.1:7860`)
