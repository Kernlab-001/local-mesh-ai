# Local Mesh AI — AI Model Licensing & Provenance Register

**Project:** Local Mesh AI  
**Repository:** https://github.com/Kernlab-001/local-mesh-ai  

---

## 1. Model Provisioning Policy

Local Mesh AI is designed with clean architectural separation between source code and large binary neural weights:
- **No Weights in Git:** Binary model checkpoints are **never committed into the git repository**, keeping git cloning fast and lean.
- **On-Demand & Offline Provisioning:** Weights are provisioned on-demand via the First-Run Model Manager or placed into `data/models/` using legal offline redistributable bundles.
- **Cryptographic Hash Verification:** Every model checkpoint loaded by Local Mesh AI is checked against verified SHA-256 digests.

---

## 2. Supported Model Register & License Details

| Model Name | Role in Local Mesh AI | Author / Organization | Upstream License | Commercial Use Allowed | Checkpoint Type |
|---|---|---|---|---|---|
| **TripoSR** | Feedforward Single-Image to 3D Mesh | Stability AI & Tripo AI | **MIT License** | **Yes** | PyTorch (`model.ckpt`) |
| **Qwen2.5-Coder 3B / 7B** | Natural Language to 3D Technical Artist Reasoning | Alibaba Cloud / Qwen Team | **Qwen License / Apache 2.0** | **Yes** (Free commercial use for <100M MAU) | GGUF (Q4_K_M) / PyTorch |
| **PartField** | 3D Shape Segmentation & Component Field Decomposition | PartField Research Authors | **MIT / Research License** | Varies by weight checkpoint | PyTorch |
| **SD-Turbo** | Instant Diffusion Texture & Concept Synthesis | Stability AI | **Stability AI Non-Commercial Research Community License** | **Non-Commercial Only** (Commercial license required from Stability AI) | SafeTensors |
| **Auxiliary Technical Artist Classifier** | Local Neural Component Classification & Parameter Predictor | Local Mesh AI Authors | **Apache License 2.0** | **Yes** | PyTorch (`adapter_model.bin`) |

---

## 3. License Specifics & Usage Conditions

### A. TripoSR
- **Repository:** https://github.com/VAST-AI-Research/TripoSR
- **License:** MIT License
- **Terms:** TripoSR weights and code may be used commercially and non-commercially with standard attribution.

### B. Qwen 2.5 Coder
- **Repository:** https://github.com/QwenLM/Qwen2.5-Coder
- **License:** Apache 2.0 / Qwen Community License Agreement
- **Terms:** Permits both research and commercial use. If your entity has monthly active users exceeding 100 million, formal authorization from Alibaba Cloud is required.

### C. SD-Turbo (Stable Diffusion Turbo)
- **Repository:** https://huggingface.co/stabilityai/sd-turbo
- **License:** Stability AI Non-Commercial Research Community License
- **Terms:** Free for academic research, education, and non-commercial personal projects. Any commercial generation or monetization requires a commercial license directly from Stability AI.

### D. Auxiliary Technical Artist Classifier
- **Authors:** Local Mesh AI Engineering Team
- **License:** Apache License 2.0
- **Terms:** 100% free for commercial and non-commercial applications.

---

## 4. User Responsibility

Users of Local Mesh AI are responsible for ensuring that their deployment and generated assets comply with the upstream terms of the specific models active in their installation.
