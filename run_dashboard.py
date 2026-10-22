"""
Starts the Local Mesh AI Standalone Dashboard on 127.0.0.1:7860.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hardware_detector import hardware_detector
from ui.dashboard import demo

if __name__ == "__main__":
    prof = hardware_detector.scan_and_configure(save_config=True)
    gpu_name = prof["gpu"]["name"]
    vram = prof["gpu"]["vram_gb"]

    print("=" * 60)
    print("STARTING LOCAL MESH AI DASHBOARD")
    print("URL: http://127.0.0.1:7860")
    print(f"Detected Hardware: {gpu_name} ({vram} GB VRAM)")
    print(f"Profile: {prof['tuning']['tier']}")
    print("=" * 60)
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
