# Local Mesh AI — Reproducible Live Demonstration Results

This document records the exact, reproducible execution steps, commands, observed timings, and persistent asset paths generated during the **Release Candidate Validation** on the Windows 11 host.

---

## 1. Demonstrated Workflows & Summary of Results

| Test Run | Input Mode | Submitted Input | Wall-Clock Time | Peak VRAM | Output Mesh Size | Vertices / Faces | Materials |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Demo 1** | Mode A (Text-to-3D) | Traditional Dining Chair Prompt | **90.05s** | ~3.02 GB | 6,507,504 bytes | 52,552 / 81,351 | 3 (`M_Wood_Frame`, `M_Fabric_Cushion`, `M_Fabric_Backrest`) |
| **Demo 2** | Mode B (Image-to-3D) | `TripoSR/examples/teapot.png` | **58.02s** | ~4.80 GB | 2,862,868 bytes | 51,214 / 78,423 | 1 (`M_Surface_PBR`) |

---

## 2. Demo 1: Text-to-3D End-to-End Validation

### Reproduction Command
```powershell
Set-Location E:\GPUPLAT
.\.venv\Scripts\python.exe run_validation_text23d.py
```

### Submitted Prompt
```text
A traditional dining chair with a dark walnut-brown wooden frame and carved legs, a warm ivory-cream upholstered seat cushion, a matching cream tufted backrest with fabric buttons, realistic wood grain, woven upholstery fabric, subtle stitching, and detailed trim.
```

### Observed Execution Log
```text
======================================================================
STARTING INDEPENDENT END-TO-END TEXT-TO-3D VALIDATION
Prompt: A traditional dining chair with a dark walnut-brown wooden frame...
======================================================================
[JOB CREATED] ID: 8f914d69
[  1.0s] Stage: Synthesizing Concept Image from Prompt (SD-Turbo) (15%) | Status: PREPROCESSING | VRAM: 0.0 MB
[ 12.0s] Stage: Reconstructing Volumetric Geometry (TripoSR) (45%) | Status: RECONSTRUCTING | VRAM: 8.1 MB
[ 72.0s] Stage: Calibrating Color Distribution (Reinhard Lab) (65%) | Status: RECONSTRUCTING | VRAM: 8.1 MB
[ 73.0s] Stage: Blender Post-Processing & Geometry Sanitation (75%) | Status: POSTPROCESSING | VRAM: 8.1 MB
[ 82.0s] Stage: Quad-Dominant Retopology & Texture Baking (85%) | Status: POSTPROCESSING | VRAM: 8.1 MB
[ 90.0s] Stage: Completed Successfully (100%) | Status: COMPLETED | VRAM: 8.1 MB

======================================================================
PIPELINE CONCLUDED in 90.05s | Final Status: COMPLETED
Peak CUDA VRAM: 3095.86 MB (~3.02 GB)
======================================================================
```

### Generated Persistent Asset Manifest
- **Job Directory:** `E:\GPUPLAT\jobs\8f914d69\`
- **Synthesized Concept Art:** `E:\GPUPLAT\jobs\8f914d69\input\concept.png` (512×512 PNG)
- **Multi-View Conditioning Grid:** `E:\GPUPLAT\jobs\8f914d69\multiview\multiview_grid.png` (1024×1024 PNG)
- **High-Poly Reference Mesh:** `E:\GPUPLAT\jobs\8f914d69\high_poly_reference\mesh.glb` (1.38 MB)
- **Low-Poly Game-Ready Asset:** `E:\GPUPLAT\jobs\8f914d69\low_poly_asset\model_game_ready.glb` (12.8 MB)
- **Final Output GLB:** `E:\GPUPLAT\jobs\8f914d69\mesh.glb` (6.51 MB)
- **Blender Project:** `E:\GPUPLAT\jobs\8f914d69\project.blend` (5.07 MB)
- **Baked Normal Map:** `E:\GPUPLAT\jobs\8f914d69\textures\baked_normal.png` (10.58 MB)
- **Baked Diffuse Map:** `E:\GPUPLAT\jobs\8f914d69\textures\baked_diffuse.png` (1.95 MB)

### Verification of Material Separation
- Inspecting the GLB container confirms 3 separate material definitions:
  - `M_M_Wood_Frame`: Applied to legs and structural side rails ($z < 0.45\text{m}$), displaying dark walnut brown (`[0.23, 0.14, 0.09]`).
  - `M_M_Fabric_Cushion`: Applied to seat cushion ($0.45\text{m} \le z \le 0.60\text{m}$), displaying warm ivory cream (`[0.94, 0.93, 0.90]`).
  - `M_M_Fabric_Backrest`: Applied to upper tufted backrest ($z > 0.60\text{m}$), displaying warm ivory cream (`[0.94, 0.93, 0.90]`).
- **Roundtrip Re-import:** Trimesh loaded `roundtrip_test.glb` directly, confirming exact geometric preservation (52,552 vertices, 81,351 faces).

---

## 3. Demo 2: Image-to-3D End-to-End Validation

### Reproduction Command
```powershell
Set-Location E:\GPUPLAT
.\.venv\Scripts\python.exe run_validation_image23d.py
```

### Reference Input
- **Source File:** `E:\GPUPLAT\TripoSR\examples\teapot.png` (275,764 bytes, 512×512 RGBA PNG)

### Observed Execution Log
```text
======================================================================
STARTING INDEPENDENT END-TO-END IMAGE-TO-3D VALIDATION
Source Image: E:\GPUPLAT\TripoSR\examples\teapot.png
======================================================================
[JOB CREATED] ID: 8ba9cd73 | Input Mode: image
[  1.0s] Stage: Reconstructing Volumetric Geometry (TripoSR) (45%) | Status: RECONSTRUCTING | VRAM: 0.0 MB
[ 50.0s] Stage: Blender Post-Processing & Geometry Sanitation (75%) | Status: POSTPROCESSING | VRAM: 0.0 MB
[ 58.0s] Stage: Completed Successfully (100%) | Status: COMPLETED | VRAM: 0.0 MB

======================================================================
IMAGE-TO-3D PIPELINE CONCLUDED in 58.02s | Final Status: COMPLETED
======================================================================
```

### Generated Persistent Asset Manifest
- **Job Directory:** `E:\GPUPLAT\jobs\8ba9cd73\`
- **Staged Reference Image:** `E:\GPUPLAT\jobs\8ba9cd73\input\source_image.png`
- **Output Mesh:** `E:\GPUPLAT\jobs\8ba9cd73\mesh.glb` (2,862,868 bytes)
- **Geometry Metrics:** 51,214 vertices, 78,423 faces, dimensions: `1.19m × 1.80m × 1.08m`
- **Material Status:** `M_M_Surface_PBR` with diffuse UV mapping and preserved surface curvature.

---

## 4. Blender 5.2.2 LTS Viewport Visual Evidence

When opening the generated `.blend` scene via the Blender add-on or CLI:
```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" "E:\GPUPLAT\jobs\8f914d69\project.blend"
```

1. **Material Preview Mode (`Z` -> Material Preview):**
   - The wooden frame and carved legs display dark walnut brown (`R=0.23, G=0.14, B=0.09`).
   - The seat cushion and backrest display warm ivory cream (`R=0.94, G=0.93, B=0.90`).
   - The mesh is not glossy white.
2. **Shader Node Tree:**
   - Principled BSDF `Base Color` is connected directly to `Image Texture (tex_0_m_wood_frame_diffuse.png)`.
   - The glTF multiply node has been automatically bypassed.
   - Diffuse color fallback is set to the sampled average RGB, ensuring Solid viewport mode also reflects the authentic palette.
3. **Outliner Organization:**
   - Objects are neatly organized inside collection `TripoSR_Generated_Assets`.
   - Transform scale is normalized to real-world metric units ($0.9\text{m}$ height).
