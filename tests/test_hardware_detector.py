"""
Unit tests for Hardware Auto-Detection Engine.
"""

import pytest
from core.hardware_detector import hardware_detector

def test_detect_gpu():
    gpu = hardware_detector.detect_gpu()
    assert "available" in gpu
    assert "device" in gpu
    assert "name" in gpu
    assert "vram_gb" in gpu
    assert isinstance(gpu["vram_gb"], float)

def test_detect_cpu_ram():
    cpu_ram = hardware_detector.detect_cpu_ram()
    assert "physical_cores" in cpu_ram
    assert "ram_gb" in cpu_ram
    assert cpu_ram["physical_cores"] > 0
    assert cpu_ram["ram_gb"] > 0

def test_tuning_calculation():
    # 24GB VRAM
    high_tuning = hardware_detector.calculate_optimal_tuning({"vram_gb": 24.0})
    assert high_tuning["chunk_size"] == 4096
    assert high_tuning["mc_resolution"] == 256
    assert high_tuning["qwen_gpu_layers"] == 33

    # 6GB VRAM
    balanced_tuning = hardware_detector.calculate_optimal_tuning({"vram_gb": 6.0})
    assert balanced_tuning["chunk_size"] == 8192
    assert balanced_tuning["mc_resolution"] == 256
    assert balanced_tuning["qwen_gpu_layers"] == 24

    # CPU / 2GB
    low_tuning = hardware_detector.calculate_optimal_tuning({"vram_gb": 2.0})
    assert low_tuning["mc_resolution"] == 128
    assert low_tuning["qwen_gpu_layers"] == 0

def test_find_blender():
    path = hardware_detector.find_blender()
    # If blender installed on machine, returns path string
    if path:
        assert "blender" in path.lower()
