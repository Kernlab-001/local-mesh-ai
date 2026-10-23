"""
Unit and Integration Tests for Blender-Inspired UI/UX Studio Dashboard.
Verifies design tokens, layout helpers, metrics extraction, and mode switching logic.
"""

import os
import pytest
from ui.dashboard import (
    BLENDER_CSS,
    render_header,
    render_stage_bar,
    render_stats_table,
    render_job_status,
    get_mesh_metrics,
    on_mode_change,
    cancel_active_job,
    demo
)

def test_blender_design_tokens():
    """Verifies that all visual design tokens match Blender dark charcoal palette."""
    assert "#1E1E1E" in BLENDER_CSS, "Main workspace background must be #1E1E1E"
    assert "#282828" in BLENDER_CSS, "Secondary panel background must be #282828"
    assert "#303030" in BLENDER_CSS, "Elevated panel background must be #303030"
    assert "#1A1A1A" in BLENDER_CSS, "Input background must be #1A1A1A"
    assert "#3B3B3B" in BLENDER_CSS, "Subtle border color must be #3B3B3B"
    assert "#E58B35" in BLENDER_CSS, "Accent active orange must be #E58B35"
    assert "#E6E6E6" in BLENDER_CSS, "Primary text must be #E6E6E6"

    # Strictly disallow blue-purple gradients, glassmorphism, or electric blue
    assert "linear-gradient(" not in BLENDER_CSS.lower()
    assert "#60a5fa" not in BLENDER_CSS.lower()
    assert "glassmorphism" not in BLENDER_CSS.lower()

def test_application_header():
    """Verifies application header renders clean status pills and brand info."""
    html = render_header()
    assert "LOCAL MESH AI" in html
    assert "blender-header" in html
    assert "GPU:" in html
    assert "blender-status-dot" in html

def test_stage_bar_dual_mode_handling():
    """Verifies stage bar reflects Mode A vs Mode B accurately."""
    # Mode A (Prompt / Text-to-3D)
    prompt_bar = render_stage_bar(1, "prompt")
    assert "2. Concept Art" in prompt_bar
    assert "stage-pill current" in prompt_bar

    # Mode B (Image / Image-to-3D)
    image_bar = render_stage_bar(1, "image")
    assert "2. Diffusion (Bypassed)" in image_bar

def test_mode_switch_visibility():
    """Verifies mode switcher updates control visibility without data loss."""
    text_up, img_up, bar_html = on_mode_change("Mode A: Text-to-3D")
    assert text_up["visible"] is True
    assert img_up["visible"] is False
    assert "2. Concept Art" in bar_html

    text_up2, img_up2, bar_html2 = on_mode_change("Mode B: Image-to-3D")
    assert text_up2["visible"] is False
    assert img_up2["visible"] is True
    assert "Diffusion (Bypassed)" in bar_html2

def test_mesh_metrics_extraction():
    """Verifies that mesh metrics are derived from actual files or safe fallbacks."""
    empty_stats = get_mesh_metrics(None)
    assert empty_stats["verts"] == "—"
    assert empty_stats["faces"] == "—"

    # Test with existing mesh if present on disk
    sample_glb = "output/meshes/0/mesh.glb"
    if os.path.exists(sample_glb):
        real_stats = get_mesh_metrics(sample_glb)
        assert real_stats["verts"] != "—"
        assert real_stats["faces"] != "—"
        assert "m ×" in real_stats["dims"]

def test_job_status_rendering():
    """Verifies job status badge styling and values."""
    completed_html = render_job_status("job_001", "COMPLETED", 100, 34.2)
    assert "COMPLETED (100%)" in completed_html
    assert "#4E8B58" in completed_html
    assert "34.2s" in completed_html

    failed_html = render_job_status("job_002", "FAILED", 45, 12.1)
    assert "FAILED (45%)" in failed_html
    assert "#9B3838" in failed_html

    cancelled_html = render_job_status("job_003", "CANCELLED", 20, 5.0)
    assert "CANCELLED (20%)" in cancelled_html
    assert "#B58A30" in cancelled_html

def test_cancel_active_job_handler():
    """Verifies cancel handler returns clean informational notice."""
    notice_empty = cancel_active_job("")
    assert "No active job selected" in notice_empty

    notice_nonexistent = cancel_active_job("non_existent_job_xyz")
    assert "could not be cancelled" in notice_nonexistent

def test_gradio_demo_structure():
    """Verifies the Blocks demo initialized properly with expected title and theme."""
    assert demo.title == "Local Mesh AI - 3D Studio Workspace"
