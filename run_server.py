"""
Starts the Local Mesh AI FastAPI server on 127.0.0.1:8000.

Features:
* A versioned identity/health contract (see ``core.backend_identity``) so that
  clients can prove they are talking to Local Mesh AI and not to an unrelated
  service that happens to own port 8000.
* A session-aware ownership watchdog. The backend no longer dies as soon as the
  process that spawned it exits -- it only shuts down once no client session
  holds a live lease (see ``services.session_registry``).
* Optional persistent log file while the console stays visible.
"""

import os
import sys
import time
import argparse
import threading
import ctypes
from typing import Optional

import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hardware_detector import hardware_detector

#: How long the backend waits (after its spawner disappeared) before it may
#: consider shutting itself down while no client session holds a lease.
SESSION_SHUTDOWN_GRACE_SECONDS = 12.0


class _Tee:
    """Writes to both stdout/stderr and a persistent log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            try:
                self.stream.write(data)
                self.stream.flush()
            except Exception:
                pass
            if self.log_file is not None:
                try:
                    self.log_file.write(data)
                    self.log_file.flush()
                except Exception:
                    pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return False


def _install_log_tee(log_path: Optional[str]):
    """Tees stdout/stderr into ``log_path``. Returns the open file handle."""
    if not log_path:
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        handle = open(log_path, "a", encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[WARN] Could not open log file '{log_path}': {exc}")
        return None
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    return handle


def start_parent_watchdog(parent_pid: int):
    """
    Monitors the process that spawned this server (usually Blender).

    The old behaviour terminated the backend the moment the spawner exited,
    which killed a backend that a second Blender instance was still using.
    The backend now shuts down only when *no* client session holds a live
    lease, so closing one Blender never breaks another one.
    """
    if not parent_pid or parent_pid <= 0:
        return

    from services.session_registry import session_registry

    def _watch():
        parent_gone = False
        # Windows native event wait (0% CPU, instant wakeup on exit)
        if sys.platform == "win32":
            try:
                SYNCHRONIZE = 0x00100000
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
                if handle:
                    kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                    kernel32.CloseHandle(handle)
                    parent_gone = True
            except Exception:
                parent_gone = False

        if not parent_gone:
            # Fallback polling via psutil
            try:
                import psutil
                while psutil.pid_exists(parent_pid):
                    time.sleep(1.0)
            except Exception:
                return

        print(f"\n[INFO] Spawning process (PID {parent_pid}) has closed. "
              f"Waiting for client sessions to finish...")

        idle_since: Optional[float] = None
        while True:
            time.sleep(2.0)
            session_registry.prune_stale()
            active = session_registry.active_session_count()
            if active > 0:
                idle_since = None
                continue
            if idle_since is None:
                idle_since = time.time()
                continue
            if (time.time() - idle_since) >= SESSION_SHUTDOWN_GRACE_SECONDS:
                print("\n[INFO] No active Local Mesh AI sessions remain. Terminating server...")
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="ParentProcessWatchdog")
    t.start()


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleTitleW("Local Mesh AI - Backend Server Console (Port 8000)")
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="Process ID (e.g. Blender PID) that spawned this server")
    parser.add_argument("--log-file", default="",
                        help="Optional path for a persistent backend log (console stays visible)")
    args, _ = parser.parse_known_args()

    log_handle = _install_log_tee(args.log_file or None)

    parent_pid = args.parent_pid or int(os.environ.get("LOCAL_MESH_AI_PARENT_PID", 0))
    if parent_pid > 0:
        start_parent_watchdog(parent_pid)

    prof = hardware_detector.scan_and_configure(save_config=True)
    gpu_name = prof["gpu"]["name"]
    vram = prof["gpu"]["vram_gb"]

    from core.backend_identity import backend_identity

    print("=" * 60)
    print("STARTING LOCAL MESH AI API SERVER")
    print("Host: 127.0.0.1:8000 (Localhost only)")
    print("API Docs: http://127.0.0.1:8000/docs")
    print(f"Hardware: {gpu_name} ({vram} GB VRAM) - {prof['tuning']['tier']}")
    print(f"Identity: {backend_identity.application} v{backend_identity.app_version} "
          f"(api {backend_identity.api_version}, build {backend_identity.build_id}, pid {backend_identity.pid})")
    print(f"Python: {sys.executable}")
    print(f"Project root: {os.path.dirname(os.path.abspath(__file__))}")
    if args.log_file:
        print(f"Log file: {os.path.abspath(args.log_file)}")
    if parent_pid > 0:
        print(f"Spawned by PID {parent_pid}: auto-closes once no client session remains")
    print("=" * 60)
    try:
        uvicorn.run("api.app:app", host="127.0.0.1", port=8000, log_level="info")
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
