# Contributing to Local Mesh AI

Thank you for your interest in contributing to **Local Mesh AI**! 

## Code Style & Standards
- Python code should adhere to PEP 8 standards with type hints where applicable.
- All Blender code must support both **Blender 4.2 LTS** and **Blender 5.2 LTS**. Avoid deprecated APIs.
- Any GPU-accelerated code must respect the `gpu_broker` lease model to prevent memory contention on lower VRAM GPUs (e.g. 4GB–6GB).
- Do not introduce arbitrary code execution paths in the Qwen planner or Blender bridge. All modifications must be schema-validated via `core/command_schemas.py`.

## Running Tests
Before submitting a PR, make sure all tests pass:
```bash
pytest tests -v
```

## Reporting Issues
Please include:
- Your OS and version
- GPU model and VRAM capacity
- Blender version
- Full error logs from `jobs/<job_id>.json` or console output
