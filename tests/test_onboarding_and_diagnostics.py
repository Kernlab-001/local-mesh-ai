"""
Unit and Integration Tests for Phase 15: Beginner Onboarding, Interactive Tour,
System Diagnostics, and User Manual.
"""

import os
from pathlib import Path
import pytest

from core.system_diagnostics import system_diagnostics
from ui.dashboard import (
    TOUR_STEPS,
    get_tour_card,
    render_diagnostics_html,
    get_user_manual_content,
    load_beginner_text_prompt,
    load_beginner_image_prompt,
    load_latest_job
)

def test_system_diagnostics_checks():
    """Verifies all system diagnostic checks run and return valid structures."""
    diag = system_diagnostics.run_all_checks()
    assert diag["verdict"] in ("READY", "WARNING", "FAILED")
    assert len(diag["checks"]) >= 5

    names = [c["name"] for c in diag["checks"]]
    assert "Python & Dependencies" in names
    assert "GPU & CUDA Acceleration" in names
    assert "Blender 3D Integration" in names
    assert "Neural Model Weights" in names
    assert "Disk Permissions & Storage" in names

    for c in diag["checks"]:
        assert c["status"] in ("READY", "WARNING", "FAILED")
        assert len(c["message"]) > 0
        assert len(c["fix"]) > 0

def test_diagnostics_sanitized_report():
    """Verifies that the markdown report is clean, copyable, and does not leak user names."""
    report = system_diagnostics.generate_sanitized_markdown()
    assert "# Local Mesh AI - System Diagnostics Report" in report
    assert "| Component | Status | Details | Recommended Fix |" in report
    assert "[READY]" in report or "[WARNING]" in report

    # Verify sensitive user home path is sanitized
    user_home = str(Path.home())
    assert user_home not in report or "<USER_HOME>" in report

def test_interactive_tour_steps():
    """Verifies the complete 7-step guided tour content and boundaries."""
    assert len(TOUR_STEPS) == 7

    for idx, step in enumerate(TOUR_STEPS):
        assert f"Step {idx + 1} of 7" in step["title"]
        assert len(step["content"]) > 20

    # Test card rendering
    card0 = get_tour_card(0)
    assert "Step 1 of 7" in card0
    assert "Mode A (Text-to-3D)" in card0

    card_last = get_tour_card(6)
    assert "Step 7 of 7" in card_last
    assert "Timeline, Logs & Error Recovery" in card_last

    # Test boundary clamping
    assert "Step 1 of 7" in get_tour_card(-5)
    assert "Step 7 of 7" in get_tour_card(999)

def test_beginner_quick_start_actions():
    """Verifies the beginner prompt and image quick-start loaders."""
    # Text Quick-Start
    mode, prompt, preset, quality, height, text_grp, img_grp, log = load_beginner_text_prompt()
    assert mode == "Mode A: Text-to-3D"
    assert "treasure chest" in prompt.lower()
    assert preset == "prop"
    assert quality == "balanced"
    assert height == 1.0
    assert text_grp["visible"] is True
    assert img_grp["visible"] is False
    assert "[BEGINNER EXAMPLE LOADED]" in log

    # Image Quick-Start
    mode_b, img, preset_b, quality_b, height_b, text_grp_b, img_grp_b, log_b = load_beginner_image_prompt()
    assert mode_b == "Mode B: Image-to-3D"
    assert text_grp_b["visible"] is False
    assert img_grp_b["visible"] is True
    assert "[BEGINNER SAMPLE LOADED]" in log_b

def test_load_latest_job_action():
    """Verifies loading previous results handles both populated and empty states safely."""
    res = load_latest_job()
    assert len(res) == 10
    # res[4] is stats_table_html, res[5] is job_status_html, res[6] is stage_bar_html
    assert "stat-table" in res[4]
    assert "stage-bar" in res[6]

def test_user_guide_manual_content():
    """Verifies that the complete beginner manual exists and contains required sections."""
    content = get_user_manual_content()
    assert len(content) > 5000
    assert "What is Local Mesh AI?" in content
    assert "Mode A: Text-to-3D" in content
    assert "Mode B: Image-to-3D" in content
    assert "Step-by-Step Tutorial" in content
    assert "Glossary" in content
    assert "Troubleshooting" in content

def test_diagnostics_html_component():
    """Verifies HTML rendering of the live diagnostics table."""
    html = render_diagnostics_html()
    assert "SYSTEM READINESS VERIFICATION" in html
    assert "diag-table" in html
    assert "Python & Dependencies" in html
    assert "GPU & CUDA Acceleration" in html
