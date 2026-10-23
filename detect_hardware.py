"""
Local Mesh AI - Hardware Autodetection & Environment Setup.
Run this script to automatically detect your local GPU, calibrate VRAM bounds,
and locate your Blender installation.
"""

import os
from pathlib import Path
from core.hardware_detector import hardware_detector

if __name__ == "__main__":
    prof = hardware_detector.scan_and_configure(save_config=True)
    print("=" * 68)
    print("  LOCAL MESH AI - HARDWARE AUTODETECTION & ENVIRONMENT CALIBRATION")
    print("=" * 68)
    print(f"  OS:       {prof['system']['os']}")
    print(f"  CPU:      {prof['system']['cpu_cores']} Physical Cores | RAM: {prof['system']['ram_gb']} GB")
    print(f"  GPU:      {prof['gpu']['name']} ({prof['gpu']['vram_gb']} GB VRAM)")
    print(f"  Device:   {prof['gpu']['device']} (CUDA: {prof['gpu']['cuda_version']})")
    print(f"  Blender:  {prof['blender']['path']}")
    print(f"  Profile:  {prof['tuning']['tier']}")
    print(f"  Settings: Resolution={prof['tuning']['mc_resolution']}, Chunk={prof['tuning']['chunk_size']}, VRAM Limit={prof['tuning']['vram_limit_gb']}GB")
    
    # Check if a GGUF model file actually exists on disk
    proj_root = Path(__file__).resolve().parent
    found_llm = None
    try:
        for f in os.listdir(proj_root):
            if f.endswith(".gguf"):
                found_llm = f
                break
    except Exception:
        pass

    if found_llm:
        print(f"  Local LLM: Installed ({found_llm})")
    else:
        print("  Local LLM: None installed (Optional - only needed for AI scene editing)")
        print(f"  Suggested Model for this GPU: {prof['tuning'].get('recommended_llm', 'Qwen 2.5 Coder 7B')}")
    print("=" * 68)
    print("  [SUCCESS] System hardware profile saved and calibrated in config.json!")
    print("=" * 68)
