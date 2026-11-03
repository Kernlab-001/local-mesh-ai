"""
Backend control helper for the Local Mesh AI Windows launcher.

The launcher used to contain the backend start-up logic inline in the batch
file, which made it impossible to verify identity, capture logs or report a
meaningful failure. This module owns that logic instead:

    python launchers/backend_control.py probe      # identity check (exit 0/1)
    python launchers/backend_control.py start      # start the backend
    python launchers/backend_control.py wait       # wait until it is ready
    python launchers/backend_control.py stop       # stop *our* backend only
    python launchers/backend_control.py status     # human readable summary

Safety rules implemented here:
  * A backend is only considered "ours" when it answers the identity contract
    (``application == "Local Mesh AI"``) -- a bare HTTP 200 is not enough.
  * The backend is never stopped unless it is a process this project started
    (recorded PID file) *and* its command line still points at this project's
    ``run_server.py``. No broad taskkill is ever used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PID_FILE = os.path.join(LOG_DIR, "backend.pid")
LOG_FILE = os.path.join(LOG_DIR, "backend.log")

sys.path.insert(0, PROJECT_ROOT)

try:
    from core.backend_identity import (  # type: ignore
        APPLICATION_NAME,
        SUPPORTED_API_MAJORS,
        verify_identity,
    )
except Exception:  # pragma: no cover - the module is always present in the repo
    APPLICATION_NAME = "Local Mesh AI"
    SUPPORTED_API_MAJORS = ("1",)

    def verify_identity(payload, expected_application=APPLICATION_NAME, expected_api_major="1"):  # type: ignore
        if not isinstance(payload, dict):
            return {"ok": False, "reason": "Backend did not return a JSON object."}
        if payload.get("application") != expected_application:
            return {"ok": False, "reason": "Different application owns the port."}
        if not payload.get("api_version"):
            return {"ok": False, "reason": "Backend did not report an api_version."}
        major = str(payload["api_version"]).split(".", 1)[0]
        if major not in SUPPORTED_API_MAJORS:
            return {"ok": False, "reason": "Incompatible API version."}
        if not payload.get("ready", False):
            return {"ok": False, "reason": "Backend is not ready yet."}
        return {"ok": True, "reason": ""}


def find_python() -> str:
    """Returns the interpreter this project must use."""
    candidates = [
        os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe"),
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def health_url() -> str:
    return os.environ.get("LOCAL_MESH_AI_BACKEND_URL", "http://127.0.0.1:8000")


def probe(timeout: float = 2.5) -> Dict[str, Any]:
    """Probes the backend and validates its identity."""
    result: Dict[str, Any] = {"ok": False, "reason": "", "payload": None, "latency_ms": 0.0}
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"{health_url()}/health", timeout=timeout) as resp:
            result["latency_ms"] = (time.time() - t0) * 1000.0
            if resp.status != 200:
                result["reason"] = f"Backend answered HTTP {resp.status}."
                return result
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        result["latency_ms"] = (time.time() - t0) * 1000.0
        result["reason"] = f"Backend answered HTTP {exc.code}."
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        return result

    verdict = verify_identity(payload)
    result["payload"] = payload
    result["ok"] = bool(verdict.get("ok"))
    result["reason"] = verdict.get("reason", "")
    return result


def describe(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return "no payload"
    return (
        f"{payload.get('application')} v{payload.get('app_version')} "
        f"(api {payload.get('api_version')}, build {payload.get('backend_build')}, "
        f"pid {payload.get('pid')})"
    )


def cmd_probe(args) -> int:
    result = probe()
    if result["ok"]:
        print(f"IDENTITY OK: {describe(result['payload'])} ({result['latency_ms']:.1f} ms)")
        return 0
    print(f"IDENTITY FAIL: {result['reason']}")
    return 1


def cmd_status(args) -> int:
    result = probe()
    if result["ok"]:
        payload = result["payload"] or {}
        print(f"Backend: {describe(payload)}")
        gpu = payload.get("gpu") or {}
        if isinstance(gpu, dict):
            print(f"GPU: {gpu.get('device_name', 'unknown')} "
                  f"({gpu.get('free_vram_mb', 0):.0f} / {gpu.get('total_vram_mb', 0):.0f} MB free)")
        sessions = payload.get("sessions") or {}
        if isinstance(sessions, dict):
            print(f"Client sessions: {sessions.get('active', 0)} active")
    else:
        print(f"Backend: NOT RUNNING / NOT OURS ({result['reason']})")
    return 0


def cmd_start(args) -> int:
    result = probe()
    if result["ok"]:
        print(f"Backend already running: {describe(result['payload'])}")
        return 0

    python_exe = find_python()
    server_script = os.path.join(PROJECT_ROOT, "run_server.py")
    if not os.path.exists(server_script):
        print(f"ERROR: run_server.py not found at {server_script}")
        return 1
    if not os.path.exists(python_exe):
        print(f"ERROR: project interpreter not found at {python_exe}")
        return 1

    os.makedirs(LOG_DIR, exist_ok=True)
    cmd = [
        python_exe,
        server_script,
        "--log-file",
        LOG_FILE,
    ]

    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT
    existing = env.get("PYTHONPATH", "")
    if existing and PROJECT_ROOT not in existing.split(os.pathsep):
        env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + existing
    elif not existing:
        env["PYTHONPATH"] = PROJECT_ROOT

    proc = None
    # The backend must keep running after this launcher exits, and its output
    # must be captured even when a console window cannot be created. The child's
    # streams are therefore redirected into the log file *and* a console window
    # is requested; run_server.py tees its output to both.
    try:
        sink = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
        flags = 0
        if sys.platform == "win32":
            flags |= subprocess.CREATE_NEW_CONSOLE
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=sink,
                stderr=sink,
                creationflags=flags,
                env=env,
            )
        except Exception as console_error:
            # Some environments (services, remote shells) refuse a new console.
            # The backend must still run -- fall back to a detached process.
            print(f"Could not create a console window ({console_error}); "
                  f"starting the backend detached. Output goes to {LOG_FILE}")
            flags = getattr(subprocess, "DETACHED_PROCESS", 0)
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=sink,
                stderr=sink,
                creationflags=flags,
                env=env,
            )
        finally:
            try:
                sink.close()
            except Exception:
                pass
    except Exception as exc:
        print(f"ERROR: could not start the backend: {exc}")
        return 1

    time.sleep(1.5)
    if proc.poll() is not None:
        print(f"ERROR: the backend process exited immediately (code {proc.returncode}).")
        if os.path.exists(LOG_FILE):
            print("--- last 25 log lines ---")
            try:
                with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as handle:
                    for line in handle.readlines()[-25:]:
                        print(line.rstrip())
            except Exception:
                pass
        return 1

    try:
        with open(PID_FILE, "w", encoding="utf-8") as handle:
            handle.write(str(proc.pid))
    except Exception:
        pass

    print(f"Backend console started (pid {proc.pid}).")
    print(f"Python:   {python_exe}")
    print(f"Log file: {LOG_FILE}")
    return 0


def cmd_wait(args) -> int:
    timeout = float(args.timeout or 60)
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        result = probe(timeout=2.0)
        if result["ok"]:
            print(f"IDENTITY OK: {describe(result['payload'])} ({result['latency_ms']:.1f} ms)")
            return 0
        last = result["reason"]
        time.sleep(1.0)

    print(f"TIMEOUT: the backend did not become ready within {timeout:.0f}s.")
    print(f"Last error: {last}")
    if os.path.exists(LOG_FILE):
        print("--- last 30 log lines ---")
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
            for line in lines[-30:]:
                print(line.rstrip())
        except Exception as exc:
            print(f"(could not read the log file: {exc})")
        print(f"Full log: {LOG_FILE}")
    return 1


def _process_matches(pid: int) -> bool:
    """True when the PID is still a backend process of THIS project."""
    try:
        import psutil  # type: ignore
    except Exception:
        return False
    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
    except Exception:
        return False
    expected = os.path.join(PROJECT_ROOT, "run_server.py").lower()
    return expected in cmdline.lower()


def cmd_stop(args) -> int:
    if not os.path.exists(PID_FILE):
        print("No backend PID file found - nothing started by this launcher is running.")
        return 0
    try:
        with open(PID_FILE, "r", encoding="utf-8") as handle:
            pid = int(handle.read().strip())
    except Exception as exc:
        print(f"Could not read the backend PID file: {exc}")
        return 1

    if not _process_matches(pid):
        print(f"PID {pid} is not running our backend (or it already exited). Nothing stopped.")
        return 0

    try:
        import psutil  # type: ignore
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
        print(f"Backend (pid {pid}) stopped.")
        return 0
    except Exception as exc:
        print(f"Could not stop the backend: {exc}")
        return 1


def cmd_blender(args) -> int:
    """Prints (or writes) the resolved Blender executable path.

    ``--emit-file <path>`` writes the path to a file so that the batch
    launcher can read it with ``set /p`` instead of parsing ``for /f``
    output, which is unreliable when the path contains spaces.
    """
    path = ""
    try:
        from core.hardware_detector import hardware_detector  # type: ignore
        path = hardware_detector.find_blender() or ""
    except Exception as exc:
        print(f"ERROR: could not resolve Blender: {exc}")
        return 1
    if not path:
        print("NOT FOUND")
        return 1

    emit_file = getattr(args, "emit_file", "") or ""
    if emit_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(emit_file)), exist_ok=True)
            with open(emit_file, "w", encoding="utf-8") as handle:
                handle.write(path)
        except Exception as exc:
            print(f"ERROR: could not write '{emit_file}': {exc}")
            return 1

    print(path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local Mesh AI backend control")
    parser.add_argument(
        "command",
        choices=["probe", "start", "wait", "stop", "status", "blender"],
        help="probe=identity check, start=launch, wait=wait until ready, "
             "stop=stop our backend, status=summary, blender=print Blender path",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait")
    parser.add_argument(
        "--emit-file",
        default="",
        help="for 'blender': also write the resolved path to this file",
    )
    args = parser.parse_args(argv)

    handlers = {
        "probe": cmd_probe,
        "start": cmd_start,
        "wait": cmd_wait,
        "stop": cmd_stop,
        "status": cmd_status,
        "blender": cmd_blender,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
