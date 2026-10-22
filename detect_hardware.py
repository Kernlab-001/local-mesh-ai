"""
Local Mesh AI - Hardware Autodetection & Environment Setup.
Run this script to automatically detect your local GPU, calibrate VRAM bounds,
and locate your Blender installation.
"""

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
    print(f"  LLM Pick: {prof['tuning'].get('recommended_llm', 'Qwen 2.5 Coder 7B')}")
    print("=" * 68)
    print("  [SUCCESS] System hardware profile saved and calibrated in config.json!")
    print("=" * 68)
