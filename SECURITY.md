# Local Mesh AI — Security Policy & Architectural Safety

**Project:** Local Mesh AI  
**Repository:** https://github.com/Kernlab-001/local-mesh-ai  

---

## 1. Reporting Security Vulnerabilities

We take the security and integrity of Local Mesh AI seriously. If you discover a security vulnerability, path traversal hazard, or execution boundary bypass, please report it responsibly:

- **Do NOT open a public GitHub issue.**
- Submit reports privately via **GitHub Security Advisories** on the repository:  
  `https://github.com/Kernlab-001/local-mesh-ai/security/advisories/new`
- Alternatively, email the maintainers directly at `security@kernlab.ai` (or repository maintainers).
- Please include detailed reproduction steps, target environment, affected version/commit, and a proof-of-concept where possible.
- We acknowledge reports within 48 hours and aim to provide verified patches within 14 days.

---

## 2. Core Architectural Safety Principles

Local Mesh AI is designed under strict defensive security guarantees to protect workstations from untrusted AI code execution, arbitrary file manipulation, and memory corruption.

### A. Non-Negotiable AI Execution Sandbox
- **No Arbitrary Python Execution:** Neural models (LLMs, vision models, classifiers) are strictly prohibited from generating executable Python scripts, shell commands, or Blender bpy scripts for automatic evaluation.
- **Strict JSON Command Schemas:** All AI technical artist operations are parsed into strongly typed, Pydantic-validated `ComponentEditPlan` schemas. Operations outside approved geometric transformations (`scale`, `separate`, `extrude`, `bevel`, `subdivide`, `normals`, `decimate`, `material`) are rejected at schema validation time.
- **Bounded Numerical Scales:** Numerical scale factors are clamped to safe domains (e.g., `[0.2, 3.0]`), preventing infinite bounding boxes or float overflow crashes.

### B. Path Security & Traversal Prevention
- All file access from API endpoints, worker threads, and Blender operators passes through `core.path_security.validate_safe_path()`.
- Access is strictly restricted to designated application subdirectories (`data/staging/`, `data/learning/`, `data/models/`, `output/`).
- Directory traversal sequences (`..`, `%2e%2e`, nested relative escapes) and system root attempts (`C:\Windows`, etc.) raise `PermissionError` and terminate the request immediately.

### C. Server-Controlled Remediation with Cryptographic Verification
- Automatic or manual staging cleanup requires a **two-stage server-controlled preview approval contract**.
- Dry-run previews compute streaming SHA-256 checksums of candidate files, generate a cryptographically random `preview_id`, and set an expiration window (`expires_at`).
- Execution verifies that the target files match both the active staging directory and their exact pre-calculated SHA-256 hashes.
- Files belonging to active generation jobs or editing sessions are automatically protected and ignored by cleanup routines.
- Every executed recovery action is permanently logged to `data/recovery_audit.json`.

### D. Cryptographic Model Artifact Verification
- Candidate neural weights and checkpoints are hashed with streaming SHA-256 upon completion.
- Candidate registration requires matching state-dict schemas, valid metadata, and cryptographic fingerprints before being accepted by `services.model_version_manager`.
- Untrusted, zero-byte, or corrupted model files cannot bypass promotion gates into active inference.

---

## 3. Supported Versions

| Version | Status | Security Patches |
|---|---|---|
| **v6.0 / Phase 9.1** (`productization/release-readiness-documentation`) | Current Release Candidate | Active Support |
| Phase 9 (`phase-9/self-healing-controlled-learning`) | Superseded | Critical Only |
| Phase 7 and earlier | Legacy Milestones | Unsupported |
