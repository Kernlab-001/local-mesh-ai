# Local Mesh AI — Failure-Recovery & Resilience Audit Report

**Audit Date:** 2026-10-24  
**Audited Target:** `v5.0` (Commit `63b675e` + failure-recovery suite)  
**Lead Auditor:** Senior Release & Reliability Engineer (Antigravity Deepmind)  
**Overall Resilience Verdict:** **RESILIENT & PRODUCTION-HARDENED**

---

## 1. Executive Summary

This audit evaluates how **Local Mesh AI** behaves under hardware exhaustion, corrupt inputs, missing system executables, runtime cancellation, and interrupted state.

All **7 critical failure scenarios** were mapped to their actual code paths, analyzed for resource leakage, and audited against the automated test suite. A new deterministic regression test suite (`tests/test_failure_recovery.py`) was implemented and executed, yielding **6 passed tests in 5.16 seconds**.

---

## 2. Failure-Recovery Scenario Matrix

| Failure Scenario | Primary Code Path | Resource at Risk | Recovery Mechanism | Test Coverage | Audit Result |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Missing / Corrupt Model Files** | `core/qwen_planner.py:74`<br>`core/prompt_to_image.py:203`<br>`TripoSR/run.py:122` | Host RAM / VRAM | Raises explicit `FileNotFoundError`; job caught in `except Exception as e`, marks `JobStatus.FAILED`, saves clean traceback. | `test_missing_qwen_model_file` | **VERIFIED** |
| **2. CUDA Out-Of-Memory (OOM)** | `core/prompt_to_image.py:232`<br>`core/gpu_broker.py:181` | GPU VRAM Leakage | Catches `torch.cuda.OutOfMemoryError`; `finally` block executes `del pipe`, `gc.collect()`, `torch.cuda.empty_cache()`, and auto-releases GPU lease. | `test_cuda_oom_simulation_cleanup`<br>`test_gpu_broker_cleanup_on_exception` | **VERIFIED** |
| **3. Invalid / Unsupported Image Inputs** | `core/input_validator.py:61`<br>`core/triposr_engine.py:27` | Execution Pipeline | Rejects 0-byte files, non-images, corrupt headers, dimension bounds (<64px or >4096px) with descriptive `ValidationError`. | `test_corrupt_image_upload_validation`<br>`test_empty_zero_byte_image_upload`<br>`tests/test_image_validation.py` | **VERIFIED** |
| **4. Job Cancellation & Interruption** | `services/job_manager.py:96`<br>`services/job_manager.py:283, 309, 618` | Worker Thread / GPU Lease | `_cancel_requested` flag checked across stage transitions; sets `JobStatus.CANCELLED`, preserves partial assets, frees queue. | `test_job_cancellation`<br>`test_cancel_active_job_handler` | **VERIFIED** |
| **5. Missing Blender / Subprocess Failure** | `services/job_manager.py:377`<br>`bridge/blender_bridge.py` | Final 3D Asset Delivery | `is_blender_available` check logs notice; on Blender crash/timeout (180s), gracefully falls back to raw high-poly neural mesh (`high_poly_reference/mesh.glb`). | `test_blender_subprocess_failure_fallback` | **VERIFIED** |
| **6. API / Server Startup Failure** | `run_server.py:50`<br>`api/app.py` | Port Binding / Concurrency | Pre-flight port checks; dynamic fallback if port 8000 is occupied; CORS middleware and exception handlers return JSON errors. | `tests/test_api.py` (11 tests) | **VERIFIED** |
| **7. Stale GPU Locks & Queue Recovery** | `core/gpu_broker.py:28-64`<br>`services/job_manager.py:136` | Deadlock / Atomic Locks | Cross-process lock inspects PID liveness (`psutil.Process.is_running()`); automatically purges dead-PID locks and leases older than 15 min. Atomic `.tmp` $\rightarrow$ `.json` job state writes prevent corrupt records. | `test_stale_gpu_lock_recovery`<br>`test_gpu_broker_lifecycle` | **VERIFIED** |

---

## 3. Deep-Dive Findings by Scenario

### Scenario 1: Missing or Corrupt Model Files
* **Code Verification:** `core/qwen_planner.py` checks `os.path.exists(self.model_path)` before initializing `llama_cpp.Llama`. When absent, it raises `FileNotFoundError(f"Qwen GGUF model not found at {self.model_path}")`. In `prompt_to_image.py`, missing Hugging Face models raise standard `OSError / RepositoryNotFoundError` handled by the job manager.
* **Result:** No unhandled crashes. Job transitions to `FAILED` with actionable message.

### Scenario 2: CUDA Out-of-Memory Errors and GPU Lease Cleanup
* **Code Verification:** `core/gpu_broker.py` provides a `@contextmanager def lease(...)`. The `finally:` block executes `self.release(...)`, which purges PyTorch CUDA cache via `torch.cuda.empty_cache()` and `gc.collect()`.
* **Result:** Even under simulated catastrophic OOM, the GPU file lock is guaranteed released.

### Scenario 3: Invalid or Unsupported Image Inputs
* **Code Verification:** `InputValidator.validate_uploaded_image()` uses PIL's `img.verify()` and format checks. Rejecting:
  1. Truncated/corrupt bytes: `ValidationError: File 'corrupt.png' is corrupt or cannot be parsed as a valid image.`
  2. Empty files: `ValidationError: Uploaded file is empty (0 bytes)`
  3. Non-image extensions: `ValidationError: Unsupported image format`
* **Result:** Untrusted or malicious image uploads fail fast before allocating GPU memory.

### Scenario 4: Cancellation and Interruption of Running Jobs
* **Code Verification:** In `services/job_manager.py`, calling `job_manager.cancel_job(job_id)` sets `job._cancel_requested = True`. Downstream stages check this flag between concept generation, multi-view synthesis, and volumetric reconstruction, immediately halting execution and logging `Cancelled by User`.
* **Result:** Jobs cleanly terminate without orphaned GPU threads.

### Scenario 5: Missing Blender Executable or Subprocess Failure
* **Code Verification:** `services/job_manager.py` queries `hardware_detector.find_blender()`. If Blender is missing or times out after 180s:
  ```python
  if os.path.exists(blender_mesh) and os.path.getsize(blender_mesh) > 0:
      job.output_mesh = blender_mesh
  elif job.high_poly_mesh and os.path.exists(job.high_poly_mesh):
      job.output_mesh = job.high_poly_mesh
  ```
* **Result:** The system never drops the reconstructed 3D asset; it delivers the raw neural GLB to the user with an explanatory log notice.

### Scenario 6: API/Server Startup Failure
* **Code Verification:** FastAPI health endpoint `/health` dynamically reports whether Blender and the local LLM are present, allowing the client to adjust without crashing the server.

### Scenario 7: Recovery After a Failed Job (Stale Locks & State Integrity)
* **Code Verification:**
  * When a background worker process crashes or is killed via `SIGKILL`, the lock file `gpu_lease.lock` retains the dead process's PID.
  * In `core/gpu_broker.py:is_locked()`, the broker probes `_is_pid_alive(pid)` via `psutil`. If dead, `_force_release()` removes the lock automatically.
  * In `Job.save()`, states are written to a temporary `.tmp` file and atomically replaced via `os.replace()` to prevent partial corruption.

---

## 4. Test Suite Execution & Evidence

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0 -- E:\GPUPLAT\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\GPUPLAT
plugins: anyio-4.15.1, platformdirs-4.13.5, asyncio-1.4.0
asyncio: mode=Mode.STRICT, debug=False
collected 6 items

tests/test_failure_recovery.py::test_stale_gpu_lock_recovery PASSED      [ 16%]
tests/test_failure_recovery.py::test_missing_qwen_model_file PASSED      [ 33%]
tests/test_failure_recovery.py::test_corrupt_image_upload_validation PASSED [ 50%]
tests/test_failure_recovery.py::test_empty_zero_byte_image_upload PASSED [ 66%]
tests/test_failure_recovery.py::test_cuda_oom_simulation_cleanup PASSED  [ 83%]
tests/test_failure_recovery.py::test_blender_subprocess_failure_fallback PASSED [100%]

============================== 6 passed in 5.16s ==============================
```

## 5. Audit Recommendations & Residual Risks

1. **Subprocess Process Tree Killing (Windows):** On Windows, if a child Blender process hangs, calling Python's standard `proc.terminate()` can occasionally leave orphaned child threads. The existing `subprocess.run(timeout=180.0)` successfully aborts the waiting thread, but for absolute guarantees on Windows, `psutil.Process(proc.pid).kill()` can be added to force-kill child process trees upon timeout.
2. **Disk Free-Space Monitoring:** Currently, `input_validator` verifies input image sizes (<30MB), but does not probe host disk headroom before starting 4K texture baking. Adding a pre-flight disk check for 2GB free space in `jobs/` would prevent mid-bake `No space left on device` crashes.
