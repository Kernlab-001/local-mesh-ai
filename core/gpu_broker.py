"""
Global Cross-Process & Intra-Process GPU Lease Broker.
Prevents concurrent VRAM contention across local processes and threads.
Ensures TripoSR, SD-Turbo, and Blender never run heavy GPU operations simultaneously.
Uses atomic OS file creation (O_CREAT | O_EXCL) and intra-process thread synchronization.
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

LOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_lease.lock")

class GPUBroker:
    def __init__(self, lock_file: str = LOCK_FILE):
        self.lock_file = lock_file
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
        if not os.path.exists(self.lock_file):
            return False
        try:
            with open(self.lock_file, "r") as f:
                data = json.load(f)
            pid = data.get("pid")
            # If process is dead, lock is stale
            if pid and not self._is_pid_alive(pid):
                self._force_release()
                return False
            # Check for excessive timeout if no heartbeat (e.g. > 15 minutes stale)
            timestamp = data.get("timestamp", 0)
            if time.time() - timestamp > 900:
                self._force_release()
                return False
            return True
        except Exception:
            return False

    def _force_release(self):
        """Forces removal of a stale lock file."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except OSError:
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
                            with open(self.lock_file, "r") as f:
                                data = json.load(f)
                            holder_pid = data.get("pid")
                            if holder_pid and not self._is_pid_alive(holder_pid):
                                self._force_release()
                            elif time.time() - data.get("timestamp", 0) > 900:
                                self._force_release()
                        except Exception:
                            self._force_release()

                    # 3. Attempt atomic file creation (O_CREAT | O_EXCL)
                    payload = {
                        "client": client_name,
                        "pid": pid,
                        "lease_id": lease_id,
                        "thread_id": threading.get_ident(),
                        "timestamp": time.time(),
                        "time_str": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                    fd = None
                    try:
                        fd = os.open(self.lock_file, flags)
                        with os.fdopen(fd, "w") as f:
                            json.dump(payload, f)
                        fd = None  # fdopen transfers ownership
                        self._current_lease_id = lease_id
                        self._current_client = client_name
                        return lease_id
                    except FileExistsError:
                        pass
                    finally:
                        if fd is not None:
                            os.close(fd)
                finally:
                    # If we failed to acquire disk lock, release thread lock
                    if self._current_lease_id != lease_id:
                        self._thread_lock.release()

            time.sleep(0.3)
        return None

    def heartbeat(self, lease_id: Optional[str] = None):
        """Updates the timestamp on an active lease to prevent expiration during long tasks."""
        target_id = lease_id or self._current_lease_id
        if not target_id or not os.path.exists(self.lock_file):
            return
        try:
            with open(self.lock_file, "r+") as f:
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
        Enforces that only the designated holder client and process can release the lock.
        Prevents cross-job or accidental foreign releases.
        """
        try:
            if os.path.exists(self.lock_file):
                with open(self.lock_file, "r") as f:
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

                if can_release and os.path.exists(self.lock_file):
                    os.remove(self.lock_file)
                    self._current_lease_id = None
                    self._current_client = None
        except Exception:
            pass
        finally:
            # Release thread lock if held
            try:
                self._thread_lock.release()
            except RuntimeError:
                pass
            self.purge_vram()

    def purge_vram(self):
        """Purges PyTorch CUDA cache and runs garbage collection."""
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
        try:
            yield self
        finally:
            self.release(client_name=client_name, lease_id=lease_id)

    def get_status(self) -> Dict[str, Any]:
        locked = self.is_locked()
        holder = None
        if locked:
            try:
                with open(self.lock_file, "r") as f:
                    holder = json.load(f)
            except Exception:
                pass

        vram_info = {"vram_allocated_mb": 0, "vram_reserved_mb": 0}
        try:
            import torch
            if torch.cuda.is_available():
                vram_info["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                vram_info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
                vram_info["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return {
            "locked": locked,
            "holder": holder,
            "vram": vram_info
        }

# Global singleton
gpu_broker = GPUBroker()
