# Local Mesh AI — Multi-Engine Architecture & Hardware-Adaptive 3D System

## 1. Executive Summary

Local Mesh AI has been upgraded from a single-engine pipeline into a modular, hardware-adaptive, local-first 3D generation platform. It coordinates neural models across differing VRAM budgets, provides dual-strategy text-to-3D workflows, enforces topological validation on generated meshes, fixes the monochromatic "white-material" syndrome, and introduces live Appearance Studio controls in Blender and the Web UI.

---

## 2. Supported Engine Roster & Hardware Requirements

| Engine ID | Name | Modality | Min VRAM | Supported Environment | Strategy / Capability |
|---|---|---|---|---|---|
| `triposr` | **TripoSR** | Image-to-3D | 3.0 GB | Windows 10/11, Linux, CUDA 11.8–12.4 | Ultra-fast feedforward single-image reconstruction baseline. Peak VRAM: 3.02GB. |
| `shap_e` | **OpenAI Shap-E** | Text-to-3D, Image-to-3D | 4.2 GB | Native PyTorch + Diffusers | **Strategy A: Direct Text-to-3D** implicit neural radiance/mesh generation without 2D diffusion. |
| `hybrid_sd_triposr` | **Hybrid SD-Turbo + TripoSR** | Text-to-3D | 3.5 GB | Windows 10/11, Linux, CUDA | **Strategy B: Multi-Stage Concept Art to 3D**. Generates 2D concept art (SD-Turbo), exposes intermediate artwork, builds 4-view projections, and reconstructs 3D asset. |
| `stable_fast_3d` | **Stable Fast 3D (SF3D)** | Image-to-3D | 8.0 GB | Requires MSVC / C++ build tools, $\ge 8\text{GB}$ VRAM | Fast UV unwrapped textured mesh synthesis with PBR materials. Gated gracefully as `UNSUPPORTED_HARDWARE` on 6GB VRAM. |
| `trellis` | **Microsoft TRELLIS** | Image-to-3D, Text-to-3D | 16.0 GB | Requires Linux / WSL2, $\ge 16\text{GB}$ VRAM | Structured latent 3D diffusion for high-fidelity assets. Gated gracefully as `UNSUPPORTED_HARDWARE` on consumer GPUs. |

---

## 3. Dynamic Hardware Adaptation & Routing

The system probes actual system hardware before engine selection:
- **Available GPU VRAM** (queried via `torch.cuda.mem_get_info()`, not just theoretical capacity).
- **System RAM** and CPU core count (via `psutil`).
- **Free disk space** for weights and caches.
- **Dependency presence** (`torch`, `diffusers`, `trimesh`, `fast_simplification`).

### Intelligent Routing Rules:
1. **Image-to-3D on Consumer GPUs (4GB–12GB VRAM):** Automatically routes to `triposr` (proven 3.02GB footprint, 12s generation time).
2. **Text-to-3D Direct (Strategy A):** Routes to `shap_e` (generates implicit 3D volumes directly from text prompts without 2D diffusion).
3. **Text-to-3D Multi-Stage (Strategy B):** Routes to `hybrid_sd_triposr` (synthesizes high-fidelity concept art with SD-Turbo under strict GPU lease, exposes preview to user, then reconstructs 3D mesh).
4. **Workstations ($\ge 16\text{GB}$ VRAM):** Unlocks `trellis` and `stable_fast_3d` when dependencies and weights are installed.
5. **Fallback Safety:** If an engine fails or exceeds memory limits, execution safely falls back to `triposr` without crashing the job.

---

## 4. Shared Geometry Quality, Validation & Finishing Pipeline

Raw neural meshes (from marching cubes or NeRF extraction) often suffer from topological defects. The `GeometryValidator` (`core/geometry_validator.py`) inspects and repairs meshes before Blender export:

1. **Non-Finite Vertices:** Purges `NaN` and `Inf` coordinates and removes orphaned incident faces.
2. **Degenerate Faces:** Eliminates zero-area and collinear faces.
3. **Winding & Normals:** Enforces consistent outward-pointing face normals.
4. **Watertightness & Manifoldness:** Checks edge topology and reports status.
5. **White-Material Syndrome Repair:** Detects meshes where vertex colors or textures have near-zero variance and luminance $> 0.94$. Synthesizes a directional ambient studio depth gradient to ensure surfaces have visual depth.
6. **Polygon Budget Decimation:** Fast quadric decimation simplifies meshes exceeding user poly budgets (e.g. 150K faces down to game-ready limits) using `fast_simplification`.

---

## 5. Appearance Studio (Blender Add-on & Web UI)

Rather than forcing users to regenerate 3D geometry when only material properties need tuning, the **Appearance Studio** enables real-time PBR material customization directly on active meshes:

### Material Presets:
- **Walnut Wood:** Dark polished wood, roughness 0.65, metallic 0.0.
- **Oak Wood:** Light amber wood, roughness 0.55, metallic 0.0.
- **Royal Velvet:** Deep rich crimson fabric, roughness 0.85, specular 0.1.
- **Brushed Steel:** Industrial metal, metallic 1.0, roughness 0.28.
- **Polished Gold:** Highly reflective gold, metallic 1.0, roughness 0.15.
- **Clean Glossy Plastic:** Dielectric surface, roughness 0.20, specular 0.5.
- **Weathered Stone:** Matte mineral finish, roughness 0.90, metallic 0.0.
- **Glazed Ceramic:** High-gloss porcelain, roughness 0.08, specular 0.7.

### Interactive Controls:
- **Base Color Tint Picker**
- **Metallic Slider (0.0 – 1.0)**
- **Roughness Slider (0.0 – 1.0)**
- **Specular Slider (0.0 – 1.0)**
- **Operators:** `Apply Preset`, `Apply Sliders`, `Reset Material`.

---

## 6. Zero-Cloud Privacy Guarantee

All neural models (`TripoSR`, `SD-Turbo`, `Shap-E`, `Qwen 2.5 Coder`) execute 100% locally on the user's workstation. No prompts, concept images, or 3D assets are transmitted over external networks.
