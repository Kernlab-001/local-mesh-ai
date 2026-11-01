"""
Global Cross-Process & Intra-Process GPU Lease Broker for Local Mesh AI.
Prevents concurrent VRAM contention across local processes and threads.
Ensures TripoSR, Qwen, LoRA training, and Blender Cycles never collide on GPU resources.
Enforces real hardware measurement (free, allocated, reserved, total VRAM),
safe heartbeat tracking, and strict release verification.
"""

import os
import sys
import time
import json
import psutil
import gc
import uuid
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any

from core.path_security import get_data_dir


def _get_default_lock_file() -> str:
    env_lock = os.environ.get("LOCAL_MESH_AI_LOCK_FILE")
    if env_lock:
        return env_lock
    return str(get_data_dir() / "gpu_lease.lock")


class GPUBroker:
    def __init__(self, lock_file: Optional[str] = None):
        self.lock_file = lock_file or _get_default_lock_file()
        self._thread_lock = threading.RLock()
        self._current_lease_id: Optional[str] = None
        self._current_client: Optional[str] = None

    def _is_pid_alive(self, pid: int) -> bool:
        """Checks if process PID is alive and not a zombie."""
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def is_locked(self) -> bool:
        """Determines if a valid, non-stale lease is currently active."""
        if not os.path.exists(self.lock_file) or os.path.getsize(self.lock_file) <= 2:
            return False
        try:
            with open(self.lock_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = data.get("pid")
            # If holder process has terminated, lock is stale and can be cleaned
            if not pid or not self._is_pid_alive(pid):
                self._force_release()
                return False

            # If process is alive, check heartbeat age.
            # A live process must refresh its heartbeat at least every 900s (15 min).
            timestamp = data.get("timestamp", 0)
            if time.time() - timestamp > 900:
                self._force_release()
                return False

            return True
        except Exception:
            self._force_release()
            return False

    def _force_release(self):
        """Forces removal or clearance of a stale lock file."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except OSError:
            try:
                with open(self.lock_file, "w", encoding="utf-8") as f:
                    f.write("{}")
            except Exception:
                pass

    def acquire(self, client_name: str, timeout: float = 90.0) -> Optional[str]:
        """
        Atomically acquires the GPU lease across threads and processes.
        Returns a unique lease_id string on success, or None on timeout.
        """
        start_time = time.time()
        lease_id = uuid.uuid4().hex
        pid = os.getpid()

        while time.time() - start_time < timeout:
            # 1. Acquire thread lock first for intra-process synchronization
            if self._thread_lock.acquire(blocking=False):
                try:
                    # 2. Check if disk lock is stale
                    if os.path.exists(self.lock_file):
                        try:
                            with open(self.lock_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            holder_pid = data.get("pid")
                            if holder_pid and not self._is_pid_alive(holder_pid):
                                self._force_release()
                            elif time.time() - data.get("timestamp", 0) > 900:
                                self._force_release()
                        except Exception:
                            self._force_release()

                    # 3. Attempt atomic file creation (O_CREAT | O_EXCL)
                    now = time.time()
                    payload = {
                        "client": client_name,
                        "pid": pid,
                        "lease_id": lease_id,
                        "thread_id": threading.get_ident(),
                        "timestamp": now,
                        "init_timestamp": now,
                        "time_str": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                    fd = None
                    try:
                        fd = os.open(self.lock_file, flags)
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                        fd = None  # fdopen transfers ownership
                        self._current_lease_id = lease_id
                        self._current_client = client_name
                        return lease_id
                    except FileExistsError:
                        if os.path.exists(self.lock_file) and os.path.getsize(self.lock_file) <= 2:
                            self._force_release()
                    finally:
                        if fd is not None:
                            os.close(fd)
                finally:
                    # If we failed to acquire disk lock, release thread lock
                    if self._current_lease_id != lease_id:
                        try:
                            self._thread_lock.release()
                        except RuntimeError:
                            pass

            time.sleep(0.2)
        return None

    def heartbeat(self, lease_id: Optional[str] = None):
        """Updates the timestamp on an active lease to prevent expiration during long tasks."""
        target_id = lease_id or self._current_lease_id
        if not target_id or not os.path.exists(self.lock_file):
            return
        try:
            with open(self.lock_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("lease_id") == target_id:
                    data["timestamp"] = time.time()
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
        except Exception:
            pass

    def release(self, client_name: Optional[str] = None, lease_id: Optional[str] = None):
        """
        Releases the GPU lease safely.
        Enforces that only the designated holder client, process, or lease_id can release the lock.
        Prevents cross-job or accidental foreign releases.
        """
        # Ergonomic support: if caller passes 32-char hex lease_id as first argument
        if lease_id is None and client_name and len(client_name) == 32 and all(c in "0123456789abcdefABCDEF" for c in client_name):
            lease_id = client_name
            client_name = None

        try:
            if os.path.exists(self.lock_file):
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                holder_client = data.get("client")
                holder_pid = data.get("pid")
                holder_lease = data.get("lease_id")
                current_pid = os.getpid()

                can_release = False
                if lease_id is not None:
                    if holder_lease == lease_id:
                        can_release = True
                elif client_name is not None:
                    if holder_client == client_name and holder_pid == current_pid:
                        can_release = True
                else:
                    if holder_pid == current_pid:
                        can_release = True

                if can_release:
                    self._force_release()
                    self._current_lease_id = None
                    self._current_client = None
        except Exception:
            pass
        finally:
            try:
                self._thread_lock.release()
            except RuntimeError:
                pass
            self.purge_vram()

    def purge_vram(self):
        """
        Purges PyTorch CUDA cache and runs garbage collection.
        CRITICAL: Never modifies or deletes the disk lock file!
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    @contextmanager
    def lease(self, client_name: str, timeout: float = 90.0):
        lease_id = self.acquire(client_name, timeout=timeout)
        if not lease_id:
            raise TimeoutError(f"Could not acquire GPU lease for '{client_name}' within {timeout}s.")

        stop_event = threading.Event()
        def _heartbeat_worker():
            while not stop_event.wait(30.0):
                self.heartbeat(lease_id=lease_id)

        heartbeat_thread = threading.Thread(target=_heartbeat_worker, daemon=True)
        heartbeat_thread.start()

        try:
            yield lease_id
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1.0)
            self.release(client_name=client_name, lease_id=lease_id)

    def get_status(self) -> Dict[str, Any]:
        """
        Returns unified, measured GPU status dictionary.
        Never substitutes 6144 MB when physical detection is unavailable.
        """
        locked = self.is_locked()
        holder = None
        lease_owner = None
        current_lease_id = None
        lease_timestamp = None
        lease_age_seconds = 0.0
        heartbeat_age_seconds = 0.0

        if locked and os.path.exists(self.lock_file):
            try:
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    holder = json.load(f)
                lease_owner = holder.get("client")
                current_lease_id = holder.get("lease_id")
                lease_timestamp = holder.get("timestamp")
                if lease_timestamp:
                    heartbeat_age_seconds = round(time.time() - lease_timestamp, 1)
                init_time = holder.get("init_timestamp", lease_timestamp)
                if init_time:
                    lease_age_seconds = round(time.time() - init_time, 1)
            except Exception:
                pass

        cuda_available = False
        device_name = None
        total_vram_mb = None
        free_vram_mb = None
        allocated_vram_mb = None
        reserved_vram_mb = None
        hardware_status = "unavailable"

        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                total_vram_mb = round(total_bytes / (1024 * 1024), 2)
                free_vram_mb = round(free_bytes / (1024 * 1024), 2)
                allocated_vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
                reserved_vram_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
                hardware_status = "healthy"
        except Exception as e:
            hardware_status = f"degraded ({str(e)})"

        return {
            "locked": locked,
            "active_lease_state": locked,
            "lease_active": locked,
            "lease_owner": lease_owner,
            "current_owner": lease_owner,
            "lease_id": current_lease_id,
            "current_lease_id": current_lease_id,
            "lease_timestamp": lease_timestamp,
            "lease_age_seconds": lease_age_seconds,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "cuda_available": cuda_available,
            "hardware_status": hardware_status,
            "gpu_device_name": device_name,
            "device_name": device_name,
            "total_vram_mb": total_vram_mb,
            "free_vram_mb": free_vram_mb,
            "allocated_vram_mb": allocated_vram_mb,
            "reserved_vram_mb": reserved_vram_mb,
            "holder": holder,
            "vram": {
                "total_vram_mb": total_vram_mb,
                "free_vram_mb": free_vram_mb,
                "vram_allocated_mb": allocated_vram_mb or 0.0,
                "vram_reserved_mb": reserved_vram_mb or 0.0,
                "device_name": device_name,
            }
        }


# Global singleton
gpu_broker = GPUBroker()
