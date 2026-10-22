"""
Tests for TripoSR Image Input Validation.
"""

import os
import pytest
from PIL import Image
from core.triposr_engine import TripoSREngine

def test_missing_image(tmp_path):
    engine = TripoSREngine()
    with pytest.raises(FileNotFoundError):
        engine.validate_image(str(tmp_path / "non_existent.png"))

def test_unsupported_format(tmp_path):
    engine = TripoSREngine()
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        engine.validate_image(str(bad_file))

def test_valid_image(tmp_path):
    engine = TripoSREngine()
    img_path = str(tmp_path / "test.png")
    img = Image.new("RGBA", (256, 256), color=(255, 0, 0, 255))
    img.save(img_path)

    meta = engine.validate_image(img_path)
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["has_alpha"] is True
