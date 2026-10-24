# Contributing to Local Mesh AI

Thank you for your interest in contributing to **Local Mesh AI**! We welcome contributions from 3D technical artists, AI/ML researchers, graphics programmers, and Python engineers.

---

## 🛠️ Development Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Kernlab-001/local-mesh-ai.git
   cd local-mesh-ai
   ```

2. **Initialize Python Environment:**
   * **Windows:** Run `launchers\1_setup_environment.bat`
   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

3. **Verify Environment & Hardware:**
   * **Windows:** Run `launchers\2_detect_hardware.bat`
   * **Linux / macOS:** `python3 detect_hardware.py`

---

## 📐 Architecture & Standards

### 1. Python Code Quality
* Adhere to **PEP 8** standards with explicit type annotations where feasible.
* Keep modules decoupled:
  * `core/`: Pure AI inference, validation, and geometry math.
  * `bridge/`: Headless Blender interaction and Cycles baking scripts.
  * `services/`: Asynchronous job orchestration and file export handling.
  * `api/`: FastAPI REST endpoints and request/response models.
  * `addon/`: Blender N-Panel UI and background timer pollers.
  * `ui/`: Blender-styled Gradio workstation.

### 2. Blender Compatibility (4.2 LTS & 5.2 LTS)
* All Blender scripts must remain compatible with both **Blender 4.2 LTS** and **Blender 5.2 LTS**.
* Avoid deprecated bpy or bmesh functions.
* When importing GLBs with existing materials, inspect shader node trees and ensure BSDF nodes are connected to Material Output.

### 3. GPU Memory Leasing & Zero-VRAM Etiquette
* Never leave heavy model weights (SD-Turbo, TripoSR) resident simultaneously in VRAM.
* Always use `gpu_broker.py` atomic locks for cross-process operations.
* Explicitly invoke `torch.cuda.empty_cache()` and `gc.collect()` when transitioning between pipeline stages.

### 4. Deterministic AI Scene Editing (No Arbitrary Code Execution)
* When extending the AI Technical Artist (Qwen 2.5 Coder):
  * Define strict Pydantic schemas in `core/command_schemas.py`.
  * Validate all tool calls against allowlisted commands before executing in Blender (`core/blender_executor.py`).
  * Never use `eval()` or unvalidated `exec()` on raw LLM output.

---

## 🧪 Running the Test Suite

Before opening a pull request, ensure all tests pass cleanly:

```bash
# Windows
launchers\4_run_tests.bat

# Linux / macOS
pytest tests -v
```

* All **79 test suites** must pass with 0 failures and 0 regressions.
* If you introduce a new feature or fix a bug, add corresponding unit or regression tests under `tests/`.

---

## 🔄 Pull Request Guidelines

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Commit Message Conventions:**
   Use Conventional Commits format:
   * `feat(scope): add new feature`
   * `fix(scope): resolve bug`
   * `docs(scope): update documentation`
   * `test(scope): add unit test coverage`
   * `refactor(scope): optimize internal logic`

3. **Submit PR:**
   * Describe the problem solved and provide before/after logs or renders where applicable.
   * Verify that tests pass.

---

## 🐛 Reporting Bugs

When filing an issue, please provide:
1. **Operating System:** (e.g., Windows 11 Build 26100, Ubuntu 22.04)
2. **GPU & Driver:** (e.g., NVIDIA RTX 3050 6GB Laptop GPU, Driver 560.xx, CUDA 12.4)
3. **Blender Version:** (e.g., Blender 5.2.2 LTS)
4. **Log Excerpt:** Paste error tracebacks from the execution console or `jobs/<job_id>/reports/manifest.json`.
5. **Steps to Reproduce:** Exact prompt or image input used.

---

## 📜 License

By contributing to Local Mesh AI, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).
