"""
Starts the Local Mesh AI FastAPI server on 127.0.0.1:8000.
Features parent process (Blender) watchdog monitoring so the terminal
console automatically closes when Blender is closed.
"""

import os
import sys
import time
import argparse
import threading
import ctypes
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hardware_detector import hardware_detector

def start_parent_watchdog(parent_pid: int):
    """
    Monitors the parent process (Blender).
    When Blender exits (window closed, quit, or crashed), this server process automatically terminates.
    """
    if not parent_pid or parent_pid <= 0:
        return

    def _watch():
        # Windows native event wait (0% CPU, instant wakeup on exit)
        if sys.platform == "win32":
            try:
                SYNCHRONIZE = 0x00100000
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
                if handle:
                    kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                    kernel32.CloseHandle(handle)
                    print(f"\n[INFO] Linked parent process (PID {parent_pid}) has closed. Terminating server...")
                    os._exit(0)
            except Exception:
                pass

        # Fallback polling via psutil
        try:
            import psutil
            while True:
                time.sleep(1.0)
                if not psutil.pid_exists(parent_pid):
                    print(f"\n[INFO] Linked parent process (PID {parent_pid}) has closed. Terminating server...")
                    os._exit(0)
        except Exception:
            pass

    t = threading.Thread(target=_watch, daemon=True, name="ParentProcessWatchdog")
    t.start()

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleTitleW("Local Mesh AI - Backend Server Console (Port 8000)")
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, default=0, help="Parent process ID (e.g. Blender PID) to monitor for auto-exit")
    args, _ = parser.parse_known_args()

    parent_pid = args.parent_pid or int(os.environ.get("LOCAL_MESH_AI_PARENT_PID", 0))
    if parent_pid > 0:
        start_parent_watchdog(parent_pid)

    prof = hardware_detector.scan_and_configure(save_config=True)
    gpu_name = prof["gpu"]["name"]
    vram = prof["gpu"]["vram_gb"]

    print("=" * 60)
    print("STARTING LOCAL MESH AI API SERVER")
    print("Host: 127.0.0.1:8000 (Localhost only)")
    print("API Docs: http://127.0.0.1:8000/docs")
    print(f"Hardware: {gpu_name} ({vram} GB VRAM) - {prof['tuning']['tier']}")
    if parent_pid > 0:
        print(f"Linked to Blender (PID {parent_pid}): Auto-closing when Blender exits")
    print("=" * 60)
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, log_level="info")
