# Local Mesh AI — Privacy Policy

**Effective Date:** November 2026  
**Product:** Local Mesh AI  
**Repository:** https://github.com/Kernlab-001/local-mesh-ai  

---

## 1. Core Commitment: Local-First Privacy

Local Mesh AI is engineered as a **local-first, privacy-preserving 3D creation and technical artist environment**. We believe creators must maintain absolute sovereignty over their 3D intellectual property, artwork, training data, and workstation telemetry.

### Key Guarantees:
1. **Zero External Inference Telemetry:** All neural inference—including 3D mesh reconstruction (TripoSR), semantic component decomposition, AI Technical Artist reasoning (Qwen and Auxiliary Classifiers), texture baking, and LoRA adaptation—executes entirely on your local workstation hardware (GPU/CPU).
2. **No Secret Telemetry:** The application contains zero third-party analytic trackers, ad networks, device fingerprinting, or hidden telemetry beacons.
3. **No Mandatory Accounts or Subscriptions:** The core application operates completely offline without user registration, account creation, or internet connectivity requirements.

---

## 2. Local Data Processing & Storage

All data processed or created by Local Mesh AI resides strictly within your local machine's directory tree:

| Data Category | Storage Location | Processing Purpose | Network Access |
|---|---|---|---|
| **Input Images & Prompts** | `data/staging/` | Local input to mesh reconstruction and technical artist pipelines | **None** (Offline) |
| **Generated 3D Assets** | `output/` | Final GLB, OBJ, and FBX exports owned solely by you | **None** (Offline) |
| **Active Editing Sessions** | `data/staging/` | Local per-session snapshots for undo/redo and atomic mutations | **None** (Offline) |
| **Local Learning Samples** | `data/learning/` | Opt-in local instruction pairs for model adaptation | **None** (Offline) |
| **Candidate Checkpoints** | `data/models/candidates/` | Locally trained neural adapter weights | **None** (Offline) |
| **Audit Logs** | `data/recovery_audit.json` | Local cryptographic records of user-approved cleanup actions | **None** (Offline) |

---

## 3. Autonomous Learning & Controlled Data Collection (Strictly Opt-In)

Local Mesh AI includes an autonomous controlled-learning subsystem designed to improve technical artist assistance. This subsystem adheres to strict privacy safeguards:

- **Disabled by Default:** Data collection is completely inactive upon initial installation (`learning_consent_opted_in = False`).
- **Explicit User Consent:** Local data logging requires deliberate opt-in in the Blender addon ("Allow Local Data Collection") or backend API.
- **Family Partitioning:** Collected samples are categorized and isolated by semantic family to prevent data leakage across distinct project categories.
- **On-Device Storage Only:** Training data is stored exclusively in `data/learning/inbox/`. It is never uploaded, shared, or accessible by external entities.
- **One-Click Permanent Deletion:** Users can permanently destroy all stored learning records at any time via the "Purge All Learning Data" operator in Blender or the `/api/v1/learning/consent/purge` API endpoint.

---

## 4. Model Downloading & Network Usage

The only network activity initiated by Local Mesh AI occurs when you explicitly instruct the application to download model weights:
- **Official Model Sources:** Weights are retrieved exclusively from vetted public repositories (such as Hugging Face official model repositories: Stability AI TripoSR, Qwen/Qwen2.5-Coder-3B-Instruct).
- **Cryptographic Hash Verification:** All downloaded model artifacts are verified against pre-published SHA-256 checksums before loading into memory.
- **Offline Model Bundle Support:** For air-gapped or offline installations, models can be placed directly into `data/models/` without network access.

---

## 5. Future Optional Cloud Services (Freemium Distinction)

If optional cloud-powered features (such as high-end distributed GPU rendering, multi-view cloud diffusion, or team asset synchronization) are introduced in future releases:
- The **local core will remain 100% functional, free, and offline-capable**.
- Any cloud processing will require **explicit, per-feature opt-in and authentication**.
- Cloud data processing will be subject to a separate transparent disclosure detailing transit encryption, retention limits, and zero-data-retention training policies.

---

## 6. Contact & Inquiries

For questions regarding data privacy or security within Local Mesh AI, please submit an issue on the [GitHub Repository](https://github.com/Kernlab-001/local-mesh-ai/issues) or review our [SECURITY.md](SECURITY.md) guidelines.
