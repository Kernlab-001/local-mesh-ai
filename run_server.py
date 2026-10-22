"""
Starts the Local Mesh AI FastAPI server on 127.0.0.1:8000.
"""

import uvicorn
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hardware_detector import hardware_detector

if __name__ == "__main__":
    prof = hardware_detector.scan_and_configure(save_config=True)
    gpu_name = prof["gpu"]["name"]
    vram = prof["gpu"]["vram_gb"]

    print("=" * 60)
    print("STARTING LOCAL MESH AI API SERVER")
    print("Host: 127.0.0.1:8000 (Localhost only)")
    print("API Docs: http://127.0.0.1:8000/docs")
    print(f"Hardware: {gpu_name} ({vram} GB VRAM) - {prof['tuning']['tier']}")
    print("=" * 60)
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, log_level="info")
