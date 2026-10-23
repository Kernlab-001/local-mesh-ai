"""
Local Neural 3D Studio - Standalone Gradio Dual-Pane Dashboard.
Connects directly to the local FastAPI orchestration service.
"""

import os
import sys
import json
import time
import subprocess
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_manager import job_manager
from services.export_service import export_service
from core.qwen_planner import qwen_planner
from core.hardware_detector import hardware_detector

def generate_3d_asset(
    gen_mode,
    concept_prompt,
    input_image,
    target_height,
    preset,
    quality,
    remove_bg,
    sanitize,
    sharp_overlay,
    weld,
    pbr_material,
    accurate_colors,
    render_tt,
    progress=gr.Progress()
):
    params = {
        "target_height": target_height,
        "preset": preset,
        "quality": quality,
        "remove_bg": remove_bg,
        "sanitize": sanitize,
        "sharp_overlay": sharp_overlay,
        "weld": weld,
        "pbr_material": pbr_material,
        "accurate_colors": accurate_colors,
        "render_turntable": render_tt
    }

    if gen_mode == "Text Prompt (Text-to-3D)":
        if not concept_prompt or len(concept_prompt.strip()) < 2:
            raise gr.Error("Please enter a concept prompt first!")
        params["prompt"] = concept_prompt.strip()
    else:
        if input_image is None:
            raise gr.Error("Please upload or drag & drop a 2D image first!")
        # Save uploaded image to temp/inputs
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inputs_dir = os.path.join(project_root, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        temp_img_path = os.path.join(inputs_dir, f"input_{int(time.time())}.png")
        input_image.save(temp_img_path)
        params["image_path"] = temp_img_path

    progress(0.1, desc="Submitting job to GPU Orchestrator...")
    job = job_manager.create_job(params)
    
    # Poll job until finished
    while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
        job = job_manager.get_job(job.job_id)
        pct = max(0.1, min(0.95, job.progress_pct / 100.0))
        progress(pct, desc=f"{job.stage} ({job.progress_pct}%)")
        time.sleep(1.0)

    if job.status == "FAILED":
        raise gr.Error(f"Generation failed: {job.error}")

    status_text = f"✅ Generation Complete in {job.elapsed_seconds}s!\nModel saved at: {job.output_mesh}\nBlender Project: {job.output_blend}"
    return job.output_mesh, job.turntable_mp4, status_text, job.job_id

def apply_qwen_edit(job_id, edit_prompt):
    if not job_id:
        return "Please generate or select a job first!"
    if not edit_prompt:
        return "Please enter a natural language modification request!"

    job = job_manager.get_job(job_id)
    if not job or not job.output_blend:
        return "Active job blend file not found."

    try:
        plan = qwen_planner.generate_plan(user_prompt=edit_prompt)
        
        blend_in = job.output_blend
        blend_out = os.path.join(os.path.dirname(blend_in), f"project_edited_{job_id}.blend")
        temp_plan = os.path.join(os.path.dirname(blend_in), f"plan_{job_id}.json")

        with open(temp_plan, "w") as f:
            f.write(plan.model_dump_json())

        blender_exe = hardware_detector.find_blender() or "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
        executor_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "blender_executor.py")
        cmd = [
            blender_exe, "--background",
            "--python", executor_script,
            "--", temp_plan, blend_in, blend_out
        ]
        subprocess.run(cmd, check=True)
        return f"✅ Applied Qwen Plan ({len(plan.commands)} operations):\n{plan.model_dump_json(indent=2)}\nSaved: {blend_out}"
    except Exception as e:
        return f"❌ Edit failed: {str(e)}"

def export_file(job_id, fmt):
    if not job_id:
        return None, "No active job selected."
    job = job_manager.get_job(job_id)
    if not job or not job.output_blend:
        return None, "Blend project not found."
    try:
        res = export_service.export_asset(job.output_blend, fmt, output_name=f"asset_{job_id}")
        return res["export_path"], f"Exported {fmt.upper()} successfully ({res['size_bytes'] / (1024*1024):.2f} MB)"
    except Exception as e:
        return None, f"Export failed: {str(e)}"

def open_in_blender(job_id):
    job = job_manager.get_job(job_id) if job_id else None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_blend = os.path.join(root, "output", "project.blend")
    blend_path = job.output_blend if job and job.output_blend else default_blend
    blender_exe = hardware_detector.find_blender() or "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
    if os.path.exists(blend_path):
        subprocess.Popen([blender_exe, blend_path])
        return f"Opened {blend_path} in Blender"
    return "No blend project found to open."

def open_output_folder():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    os.startfile(out_dir)
    return "Opened output folder."

# --- Gradio UI Layout ---
with gr.Blocks(title="Local Mesh AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Local Mesh AI (TripoSR + SD-Turbo + Blender)")
    gr.Markdown("Transform text prompts and 2D concept art into clean, watertight, textured 3D models locally with GPU acceleration.")

    with gr.Row():
        # LEFT PANE: Inputs & Controls
        with gr.Column(scale=1):
            gen_mode = gr.Radio(
                choices=["Text Prompt (Text-to-3D)", "Image Upload (Image-to-3D)"],
                value="Text Prompt (Text-to-3D)",
                label="Generation Mode"
            )
            concept_prompt = gr.Textbox(
                label="Concept Prompt",
                placeholder="Describe your 3D asset (e.g. 'medieval glowing battleaxe with carved runes')",
                value="medieval glowing battleaxe with carved runes and iron handle",
                lines=2
            )
            input_image = gr.Image(type="pil", label="Input Concept Image (Optional in Text Mode)", sources=["upload", "clipboard"])
            
            with gr.Row():
                preset = gr.Dropdown(choices=["prop", "character", "weapon", "furniture"], value="prop", label="Asset Preset")
                quality = gr.Dropdown(choices=["draft", "balanced", "high"], value="balanced", label="Reconstruction Quality")

            target_height = gr.Slider(minimum=0.1, maximum=10.0, value=1.8, step=0.1, label="Target Height (Meters)")

            with gr.Accordion("⚙️ Advanced Mesh & Shader Processing", open=False):
                remove_bg = gr.Checkbox(value=True, label="Remove Background (rembg)")
                sanitize = gr.Checkbox(value=True, label="Sanitize Geometry (Clean floating islands & cap holes)")
                sharp_overlay = gr.Checkbox(value=True, label="Sharp Decal / UV Projection Overlay")
                weld = gr.Checkbox(value=True, label="Adaptive Welding (Weld micro-gaps)")
                pbr_material = gr.Checkbox(value=True, label="Synthesize Principled BSDF PBR Material")
                accurate_colors = gr.Checkbox(value=True, label="Accurate Color Calibration (Tripo AI Style)")
                render_tt = gr.Checkbox(value=False, label="Render 360° Turntable Video")

            gen_btn = gr.Button("⚡ Generate 3D Asset", variant="primary", size="lg")
            active_job_id = gr.Textbox(visible=False)

            # Qwen Natural Language Edit Section
            gr.Markdown("### 🧠 AI Technical Artist (Qwen 2.5 Coder)")
            edit_prompt = gr.Textbox(placeholder="e.g. 'Make it shiny gold metallic and bevel edges by 2mm'", label="Natural Language Edit Request")
            edit_btn = gr.Button("Apply AI Modification Plan", variant="secondary")
            edit_output = gr.Textbox(label="AI Edit Result", lines=4)

        # RIGHT PANE: Interactive 3D Viewer & Media
        with gr.Column(scale=1):
            model_3d = gr.Model3D(label="Interactive 3D Viewport (Three.js)", display_mode="solid")
            turntable_video = gr.Video(label="360° Turntable Video", visible=True)
            status_box = gr.Textbox(label="Execution Status & Benchmark", lines=3)

    # FOOTER ACTION BAR
    gr.Markdown("### 📦 Asset Export & Destination Actions")
    with gr.Row():
        exp_glb_btn = gr.Button("Export GLB")
        exp_fbx_btn = gr.Button("Export FBX")
        exp_obj_btn = gr.Button("Export OBJ")
        open_blend_btn = gr.Button("Open in Blender 5.2")
        open_folder_btn = gr.Button("Open Output Folder")

    export_file_out = gr.File(label="Exported File Download")
    action_log = gr.Textbox(label="Action Log", lines=2)

    # Handlers
    gen_btn.click(
        fn=generate_3d_asset,
        inputs=[gen_mode, concept_prompt, input_image, target_height, preset, quality, remove_bg, sanitize, sharp_overlay, weld, pbr_material, accurate_colors, render_tt],
        outputs=[model_3d, turntable_video, status_box, active_job_id]
    )

    edit_btn.click(
        fn=apply_qwen_edit,
        inputs=[active_job_id, edit_prompt],
        outputs=[edit_output]
    )

    exp_glb_btn.click(fn=lambda j: export_file(j, "glb"), inputs=[active_job_id], outputs=[export_file_out, action_log])
    exp_fbx_btn.click(fn=lambda j: export_file(j, "fbx"), inputs=[active_job_id], outputs=[export_file_out, action_log])
    exp_obj_btn.click(fn=lambda j: export_file(j, "obj"), inputs=[active_job_id], outputs=[export_file_out, action_log])
    open_blend_btn.click(fn=open_in_blender, inputs=[active_job_id], outputs=[action_log])
    open_folder_btn.click(fn=open_output_folder, outputs=[action_log])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
