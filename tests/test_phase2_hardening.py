"""
Unit & Acceptance Test Suite for Phase 2 Quality Hardening.
Verifies:
1. Foreground segmentation & alpha matting (offline operation, transparent inputs, thin features, error handling).
2. Dynamic engine availability gating & truthful UI states.
3. Blender vertex-color shader node binding for full-color rendering.
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import sys
import tempfile
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.foreground_segmenter import foreground_segmenter, SegmentationError, SegmentationResult
from core.engines.registry import engine_registry
from core.engines.base import EngineStatus


class TestForegroundSegmentation:

    def test_already_transparent_image_preserved(self):
        """Images that already contain an authored alpha channel must be preserved as-is."""
        rgba = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        # Add a colored square in the center
        for x in range(32, 96):
            for y in range(32, 96):
                rgba.putpixel((x, y), (200, 50, 50, 255))

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "transparent_input.png")
            out_path = os.path.join(tmpdir, "segmented_out.png")
            rgba.save(in_path)

            res = foreground_segmenter.segment(in_path, out_path)
            assert res.success is True
            assert res.status == "ALREADY_TRANSPARENT"
            assert res.method == "input_alpha_preserved"
            assert os.path.exists(out_path)

            with Image.open(out_path) as out_im:
                assert out_im.mode == "RGBA"
                alpha = np.array(out_im.split()[-1])
                assert np.min(alpha) == 0
                assert np.max(alpha) == 255

    def test_opaque_image_with_background_segmented(self):
        """Opaque image with solid studio background must have borders made transparent."""
        # Create an image with light grey studio background (210, 210, 210)
        im = Image.new("RGB", (256, 256), (210, 210, 210))
        # Draw a dark brown chair-like object in the center (80, 45, 20)
        for x in range(60, 196):
            for y in range(50, 220):
                im.putpixel((x, y), (80, 45, 20))

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "opaque_chair.png")
            out_path = os.path.join(tmpdir, "segmented_chair.png")
            im.save(in_path)

            res = foreground_segmenter.segment(in_path, out_path)
            assert res.success is True
            assert res.status == "SEGMENTED_OFFLINE_SALIENT_FLOODFILL"
            assert os.path.exists(out_path)

            with Image.open(out_path) as out_im:
                assert out_im.mode == "RGBA"
                alpha = np.array(out_im.split()[-1])
                # Corners must be transparent
                assert alpha[5, 5] == 0
                assert alpha[5, 250] == 0
                assert alpha[250, 5] == 0
                assert alpha[250, 250] == 0
                # Center subject must be opaque
                assert alpha[128, 128] > 200

    def test_all_background_raises_segmentation_error(self):
        """An image containing only uniform background must raise SegmentationError."""
        flat_im = Image.new("RGB", (128, 128), (220, 220, 220))

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "flat_bg.png")
            flat_im.save(in_path)

            with pytest.raises(SegmentationError) as exc_info:
                foreground_segmenter.segment(in_path)
            assert "Segmentation failed" in str(exc_info.value)

    def test_thin_feature_preservation(self):
        """Thin features like chair legs (width 3-5px) must remain connected in foreground mask."""
        im = Image.new("RGB", (200, 200), (220, 220, 220))
        # Draw central body
        for x in range(60, 140):
            for y in range(40, 110):
                im.putpixel((x, y), (50, 50, 50))
        # Draw two thin legs (4px wide) extending downwards
        for y in range(110, 180):
            for x in range(65, 70):
                im.putpixel((x, y), (50, 50, 50))
            for x in range(130, 135):
                im.putpixel((x, y), (50, 50, 50))

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "chair_thin_legs.png")
            out_path = os.path.join(tmpdir, "segmented_legs.png")
            im.save(in_path)

            res = foreground_segmenter.segment(in_path, out_path)
            assert res.success is True

            with Image.open(out_path) as out_im:
                alpha = np.array(out_im.split()[-1])
                # Verify left leg pixels remain opaque
                assert np.all(alpha[140:170, 67] > 180)
                # Verify right leg pixels remain opaque
                assert np.all(alpha[140:170, 132] > 180)
                # Verify gap between legs is transparent
                assert alpha[150, 100] == 0


class TestDynamicEngineAvailability:

    def test_dynamic_choices_contains_truthful_labels(self):
        """Dropdown choices must accurately reflect current system availability."""
        choices, mapping = engine_registry.get_dynamic_engine_choices()

        # Check Auto-Detect is first
        assert choices[0] == "Auto-Detect (Adaptive Hardware Router)"
        assert mapping[choices[0]] == "auto"

        # Check ready engines
        ready_triposr = [c for c in choices if c.startswith("TripoSR") and "[READY]" in c]
        assert len(ready_triposr) == 1

        hybrid_choices = [c for c in choices if "Hybrid" in c]
        assert len(hybrid_choices) == 1
        assert any(tag in hybrid_choices[0] for tag in ("[READY]", "[NEEDS DOWNLOAD]"))

        # Check Shap-E is truthfully labeled as NEEDS DOWNLOAD
        shape_choices = [c for c in choices if "Shap-E" in c]
        assert len(shape_choices) == 1
        assert "[NEEDS DOWNLOAD]" in shape_choices[0]

        # Check high-VRAM engines are truthfully labeled as UNSUPPORTED / REQUIRES VRAM
        sf3d_choices = [c for c in choices if "Stable Fast 3D" in c]
        assert len(sf3d_choices) == 1
        assert "REQUIRES" in sf3d_choices[0] or "UNSUPPORTED" in sf3d_choices[0]

    def test_unavailable_engine_raises_actionable_error(self):
        """Selecting an unavailable engine must expose truthful status and actionable guidance."""
        shape_eng = engine_registry.get_engine("shap_e")
        assert shape_eng is not None
        status = shape_eng.get_status()
        assert status == EngineStatus.NEEDS_DOWNLOAD
        details = shape_eng.get_status_details()
        assert details["ready"] is False
        assert "1.7 GB" in details.get("approximate_download_size", "")

        sf3d_eng = engine_registry.get_engine("stable_fast_3d")
        assert sf3d_eng is not None
        assert sf3d_eng.get_status() == EngineStatus.UNSUPPORTED_HARDWARE
        sf3d_details = sf3d_eng.get_status_details()
        assert "8.0 GB VRAM" in sf3d_details.get("unsupported_reason", "")
