"""
Global Cross-Process GPU Lease Broker.
Prevents concurrent VRAM contention across local GPU processes.
Ensures TripoSR, Qwen 2.5 Coder, and Blender never run heavy GPU inference simultaneously.
"""

import os
import sys
import time
import json
import psutil
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any

LOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_lease.lock")

class GPUBroker:
    def __init__(self, lock_file: str = LOCK_FILE):
        self.lock_file = lock_file

    def _is_pid_alive(self, pid: int) -> bool:
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def is_locked(self) -> bool:
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
            # Check for excessive timeout (e.g. > 15 minutes stale)
            timestamp = data.get("timestamp", 0)
            if time.time() - timestamp > 900:
                self._force_release()
                return False
            return True
        except Exception:
            return False

    def _force_release(self):
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except OSError:
            pass

    def acquire(self, client_name: str, timeout: float = 90.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.is_locked():
                try:
                    # Write lock atomically
                    payload = {
                        "client": client_name,
                        "pid": os.getpid(),
                        "timestamp": time.time(),
                        "time_str": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    temp_lock = f"{self.lock_file}.{os.getpid()}.tmp"
                    with open(temp_lock, "w") as f:
                        json.dump(payload, f)
                    os.replace(temp_lock, self.lock_file)
                    return True
                except Exception:
                    pass
            time.sleep(0.5)
        return False

    def release(self, client_name: Optional[str] = None):
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
                current_pid = os.getpid()

                can_release = False
                if client_name is not None:
                    if holder_client == client_name and holder_pid == current_pid:
                        can_release = True
                else:
                    if holder_pid == current_pid:
                        can_release = True

                if can_release and os.path.exists(self.lock_file):
                    os.remove(self.lock_file)
        except Exception:
            pass
        finally:
            self.purge_vram()

    def purge_vram(self):
        """Purges PyTorch CUDA cache and runs garbage collection."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        gc.collect()

    @contextmanager
    def lease(self, client_name: str, timeout: float = 90.0):
        acquired = self.acquire(client_name, timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire GPU lease for '{client_name}' within {timeout}s.")
        try:
            yield self
        finally:
            self.release(client_name)

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
