"""
Unit tests for Color Calibration & Transfer Engine.
"""

import os
import numpy as np
import pytest
from PIL import Image
from core.color_calibrator import color_calibrator

def test_foreground_color_extraction(tmp_path):
    img_path = str(tmp_path / "test_color_art.png")
    # Red circle on transparent background
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[16:48, 16:48] = [255, 50, 50, 255]
    Image.fromarray(arr).save(img_path)

    fg = color_calibrator.extract_foreground_colors(img_path)
    assert len(fg) > 0
    # Mean should be heavily red
    assert fg[:, 0].mean() > 0.8
    assert fg[:, 1].mean() < 0.4
    assert fg[:, 2].mean() < 0.4

def test_reinhard_color_transfer():
    # Target is neutral gray
    target = np.full((100, 3), 0.5, dtype=np.float32)
    # Reference is vibrant gold [0.9, 0.75, 0.1]
    reference = np.array([[0.9, 0.75, 0.1]] * 100, dtype=np.float32)

    calibrated = color_calibrator.transfer_color_distribution(target, reference, vibrancy_boost=1.0)
    assert calibrated.shape == (100, 3)
    # Calibrated colors should be pulled towards gold
    assert calibrated[:, 0].mean() > 0.7
    assert calibrated[:, 1].mean() > 0.6
    assert calibrated[:, 2].mean() < 0.3
