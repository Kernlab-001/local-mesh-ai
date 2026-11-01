"""
Automated Product Documentation & Governance Verification Suite.
Validates the presence, structural integrity, licensing disclosures, and cross-references
of all release documents, policies, architecture specifications, and Blender addon registrations.
"""

import os
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_required_policy_documents_exist():
    """Verifies that all mandatory product governance documents are present."""
    required_files = [
        "README.md",
        "BEGINNER_GUIDE.md",
        "PRIVACY.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "THIRD_PARTY_NOTICES.md",
        "MODEL_LICENSES.md",
        "LICENSE",
        "Start_Local_Mesh_AI.bat",
        "launchers/README.md",
        "docs/USER_GUIDE.md",
        "docs/WINDOWS_DISTRIBUTION_ARCHITECTURE.md",
        "docs/MODEL_PROVISIONING_SPEC.md",
        "docs/FREEMIUM_ARCHITECTURE.md",
    ]
    for rel_path in required_files:
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), f"Required file '{rel_path}' is missing."
        assert full_path.stat().st_size > 100, f"Required file '{rel_path}' is suspiciously small or empty."


def test_readme_cross_references_resolve():
    """Verifies that internal markdown file links in README.md point to real existing files."""
    readme_path = REPO_ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    # Match markdown links: [text](path)
    links = re.findall(r"\[.*?\]\((?!https?://)(.*?)\)", content)
    for target in links:
        # Strip anchor links like #section
        clean_target = target.split("#")[0]
        if not clean_target:
            continue
        target_path = (REPO_ROOT / clean_target).resolve()
        assert target_path.exists(), f"Broken link in README.md: '{target}' (resolved to {target_path})"


def test_privacy_policy_declarations():
    """Verifies that PRIVACY.md documents 100% on-device inference and opt-in learning guarantees."""
    privacy_path = REPO_ROOT / "PRIVACY.md"
    content = privacy_path.read_text(encoding="utf-8")

    assert "100% Local Inference Guarantee" in content or "Local-First Privacy" in content
    assert "Opt-In" in content
    assert "Zero External Inference Telemetry" in content
    assert "Purge All Learning Data" in content


def test_security_policy_declarations():
    """Verifies that SECURITY.md documents vulnerability reporting and strict AI execution sandboxing."""
    security_path = REPO_ROOT / "SECURITY.md"
    content = security_path.read_text(encoding="utf-8")

    assert "Reporting Security Vulnerabilities" in content
    assert "No Arbitrary Python Execution" in content
    assert "Path Security" in content
    assert "preview_id" in content or "preview approval" in content


def test_licensing_audit_disclosures():
    """Verifies that THIRD_PARTY_NOTICES.md and MODEL_LICENSES.md detail licenses accurately."""
    tpn = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Apache License 2.0" in tpn
    assert "GNU General Public License v3.0" in tpn or "GPLv3" in tpn
    assert "FastAPI" in tpn
    assert "PyTorch" in tpn

    ml = (REPO_ROOT / "MODEL_LICENSES.md").read_text(encoding="utf-8")
    assert "TripoSR" in ml
    assert "Qwen2.5-Coder" in ml
    assert "Auxiliary Technical Artist Classifier" in ml


def test_historical_milestone_banners():
    """Verifies that all historical milestone reports in docs/ contain the historical banner."""
    docs_dir = REPO_ROOT / "docs"
    reports = list(docs_dir.glob("PHASE_*_REPORT.md"))
    assert len(reports) >= 8, f"Expected at least 8 phase reports, found {len(reports)}."

    for rep in reports:
        text = rep.read_text(encoding="utf-8")
        assert "Historical Milestone Record" in text, f"Missing historical banner in {rep.name}"


def test_blender_addon_unified_tab_and_no_legacy_panel():
    """Verifies that Blender addon has strictly one sidebar category and no legacy panel."""
    addon_path = REPO_ROOT / "addon" / "triposr_blender_addon.py"
    text = addon_path.read_text(encoding="utf-8")

    assert 'bl_category = "AI 3D Generator"' not in text, "Found legacy 'AI 3D Generator' tab in addon."
    assert 'TRIPOSR_PT_MainPanel' not in text, "Found deprecated TRIPOSR_PT_MainPanel in addon."
    assert 'LOCAL MESH AI' in text, "Addon must register under 'LOCAL MESH AI' sidebar category."
    assert 'recovery_preview_id' in text, "Addon must track recovery_preview_id."


def test_master_launcher_exists_and_references_venv():
    """Verifies master launcher checks for .venv and includes menu options."""
    launcher_path = REPO_ROOT / "Start_Local_Mesh_AI.bat"
    text = launcher_path.read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe" in text
    assert "Start Full Studio" in text
    assert "run_server.py" in text
    assert "dashboard.py" in text
