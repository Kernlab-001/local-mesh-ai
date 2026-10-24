# Local Mesh AI — Legal, Licensing & Distribution Audit Report

**Audit Date:** 2026-10-24  
**Audited Repository:** `https://github.com/Kernlab-001/local-mesh-ai`  
**Audited Release Target:** `v5.0`  
**Current Commit:** `63b675e` (plus local audit documentation)  
**Lead Auditor:** Senior AI Software Architect & Graphics Pipeline Counsel (Antigravity Deepmind)  
**Legal Clearance Recommendation:** **CONDITIONAL CLEARANCE (NON-COMMERCIAL / RESEARCH ONLY)**  
*(Cannot be declared cleared for unrestricted commercial production without replacing or licensing the text-to-image diffusion model).*

---

## 1. Executive Summary

This audit independently investigates the legal, copyright, and distribution compliance of **Local Mesh AI v5.0**. It examines the codebase, configuration files, bundled assets, Git tracking rules, release packaging, and external AI model weights.

### Primary Legal Findings:
1. **Application Code License (Apache 2.0):** The core application logic authored by the repository maintainers is validly licensed under the **Apache License 2.0** (`LICENSE`).
2. **SD-Turbo Non-Commercial Restriction:** The default concept synthesis engine in `core/prompt_to_image.py` downloads and executes `stabilityai/sd-turbo`. This model is distributed under the **Stability AI Non-Commercial Research Community License**. Downstream users **cannot use the default text-to-3D pipeline for commercial purposes** without a commercial license from Stability AI Ltd.
3. **Qwen 2.5 Coder License Misattribution Corrected:** The repository previously stated that Qwen 2.5 Coder was governed by the *Tongyi Qianwen License Agreement*. Official upstream records confirm that Alibaba Cloud released **Qwen2.5-Coder-7B-Instruct** under the permissive **Apache 2.0 License**, granting commercial and modification rights.
4. **TripoSR Model & Code (MIT):** Both the codebase in `TripoSR/` and the model weights (`stabilityai/TripoSR`) are released under the **MIT License**. Commercial use and modification are permitted provided the copyright notice (`Copyright (c) 2024 Tripo AI & Stability AI`) is preserved.
5. **Blender Python Addon (`bpy`) & GPLv3 Compliance:** `addon/triposr_blender_addon.py` imports `bpy` and executes inside the Blender runtime. Under Blender Foundation's official legal policy, in-process addons are derivative works governed by the **GNU GPL**. The addon file must explicitly designate **GNU GPLv3** (or Apache 2.0 / GPLv3 dual licensing) to ensure full copyleft compliance.
6. **No Binary Model Weights Bundled in Releases:** Verified that `release/local-mesh-ai-v5.0.zip` (49.5 MB) does **not** redistribute binary weights (`.gguf`, `.safetensors`, `.bin`, `.pth`). All weights are downloaded dynamically by the end user via Hugging Face Hub.

---

## 2. Comprehensive Component License Inventory

| Component / Artifact | Type | Official License | Upstream Source Link | Commercial Use | Weight Redistribution | Attribution Required | Confidence | Required Compliance Action |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Local Mesh AI Codebase** | Application Source | **Apache 2.0** | [`LICENSE`](../LICENSE) | Yes | Yes | Yes (Apache 2.0 Notice) | **HIGH** | Maintain `LICENSE` and state modifications in headers. |
| **TripoSR Engine & Code** | Submodule / Library | **MIT License** | [VAST-AI / TripoSR](https://github.com/VAST-AI-Research/TripoSR) | Yes | Yes | Yes (`Copyright (c) 2024 Tripo AI & Stability AI`) | **HIGH** | Retain `TripoSR/LICENSE` in all distributions. |
| **TripoSR Model Weights** | Neural Weights (`model.ckpt`) | **MIT License** | [stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR) | Yes | Yes | Yes | **HIGH** | Documented on Hugging Face model card. |
| **SD-Turbo Weights** | Text-to-Image Model | **Stability AI Non-Commercial Research Community License** | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | **NO** | Allowed with terms | Yes | **HIGH** | **CRITICAL:** Prominently warn users that Mode A text-to-3D is non-commercial only unless an enterprise Stability AI license is held. |
| **Qwen 2.5 Coder 7B Instruct** | LLM Planner Weights (GGUF) | **Apache 2.0** | [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | Yes | Yes | Yes (Alibaba Cloud attribution) | **HIGH** | Update documentation to reflect official Apache 2.0 license instead of Tongyi Qianwen agreement. |
| **Blender Add-on (`bpy`)** | Blender Python Extension | **GNU GPLv3 / GPLv2+** | [Blender Foundation Policy](https://www.blender.org/about/license/) | Yes (under GPL) | Under GPL terms | Yes | **HIGH** | Add explicit GPLv3 license header to `addon/triposr_blender_addon.py`. |
| **rembg (Python Library)** | Image Background Removal | **MIT License** | [danielgatis/rembg](https://github.com/danielgatis/rembg) | Yes | Yes | Yes | **HIGH** | Permissive library. |
| **U²-Net Model Weights** | Background Removal Model | **Apache 2.0** | [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | Yes | Yes | Yes | **HIGH** | Standard segmentation checkpoint. |
| **Trimesh, ModernGL, xatlas** | 3D Processing Libraries | **MIT License** | PyPI / GitHub | Yes | Yes | Yes | **HIGH** | Compliant with Apache 2.0. |
| **Diffusers, Transformers** | ML Inference Frameworks | **Apache 2.0** | [Hugging Face](https://github.com/huggingface/diffusers) | Yes | Yes | Yes | **HIGH** | Fully compatible. |
| **PyTorch & Torchvision** | Deep Learning Framework | **BSD-3-Clause** | [pytorch.org](https://github.com/pytorch/pytorch) | Yes | Yes | Yes | **HIGH** | Standard BSD terms. |

---

## 3. Analysis of Critical Licensing Boundaries

### 3.1. The Commercial Usability Divide: Mode A vs Mode B
* **Mode B (Image-to-3D):**
  * When a user provides their own 2D concept image or drawing, the pipeline executes **only** TripoSR, Reinhard color calibration, and Blender headless geometry processing.
  * Because TripoSR code, TripoSR weights, and Blender scripts are all permissively licensed (MIT / Apache / GPL), **Mode B is fully commercially permissible** (assuming the user owns copyright to their input artwork).
* **Mode A (Text-to-3D):**
  * When a user inputs a text prompt, the pipeline calls `stabilityai/sd-turbo`.
  * Because SD-Turbo is licensed under Stability AI's Non-Commercial Research Community License, **the generated 2D concept art and derived 3D models are encumbered by non-commercial restrictions**.
  * **Remediation Path:** To achieve 100% unrestricted commercial clearance, Local Mesh AI must provide a configuration switch enabling commercial-friendly diffusion backends (e.g. `black-forest-labs/FLUX.1-schnell` with Apache 2.0 or `PixArt-alpha/PixArt-sigma` with Apache 2.0).

### 3.2. Blender GPLv3 Compliance for Add-on
* The Blender Python API (`bpy`) is covered by the GNU General Public License.
* By importing `bpy` in `addon/triposr_blender_addon.py`, the add-on runs within Blender's memory space.
* The main Local Mesh AI backend (`run_server.py`, `services/`, `core/`) is decoupled across OS process boundaries (communicating via HTTP REST on port 8000), which protects the backend from GPL copyleft contagion.
* However, the add-on script itself must be licensed under a GPL-compatible license. Since Apache 2.0 is compatible with GPLv3 (but not GPLv2), the add-on should be explicitly declared under **GNU GPLv3**.

---

## 4. Packaging and Weight Redistribution Audit

* **Git Tracking Verification:**
  * Inspected `.gitignore` lines 31-38:
    ```gitignore
    # Large Model Weights (>100MB, cannot be pushed to GitHub)
    *.gguf
    *.bin
    *.pth
    *.pt
    *.safetensors
    *.onnx
    ```
  * Executed `git ls-files` search across all binary weight extensions. **Confirmed: Zero model weights are tracked in Git.**
* **Release Archive Verification:**
  * Inspected `release/local-mesh-ai-v5.0.zip` (49.5 MB) and `release/triposr_blender_addon_v5.0.zip` (12 KB).
  * Neither archive contains pre-downloaded model checkpoints. The 4.68 GB Qwen GGUF model and SD-Turbo weights remain strictly external.

---

## 5. Required Action Plan

1. **Update `README.md` License Attributions:** Correct Qwen 2.5 Coder from "Tongyi Qianwen License Agreement" to **Apache 2.0**.
2. **Add Commercial Disclaimer:** Explicitly document the distinction between Mode A (non-commercial due to SD-Turbo) and Mode B (commercially usable).
3. **Add License Header to Blender Add-on:** Add a GNU GPLv3 header to `addon/triposr_blender_addon.py`.
4. **Final Legal Status:** The project is **CONDITIONALLY CLEARED** for open-source, academic, and non-commercial research use. Commercial deployment requires replacing SD-Turbo or acquiring commercial rights from Stability AI.
