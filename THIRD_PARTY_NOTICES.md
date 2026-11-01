# Local Mesh AI — Third-Party Notices & Open Source Attribution

**Project:** Local Mesh AI  
**Repository:** https://github.com/Kernlab-001/local-mesh-ai  

---

## 1. Project Licensing Overview

Local Mesh AI is built with strict adherence to open-source licensing boundaries:
- **Core Orchestration, Engine & Backend Services:** Licensed under the **Apache License 2.0**.
- **Blender Studio Addon (`addon/triposr_blender_addon.py`):** Licensed under the **GNU General Public License v3.0 (GPLv3)**, in full compliance with Blender's Python API (`bpy`) licensing requirements.
- **AI Models & Weights:** Retain their respective upstream author licenses (see [MODEL_LICENSES.md](MODEL_LICENSES.md)).

---

## 2. Third-Party Python Libraries

The following open-source libraries and components are utilized by Local Mesh AI:

### FastAPI
- **License:** MIT License
- **Copyright:** (c) 2018 Sebastián Ramírez
- **URL:** https://github.com/fastapi/fastapi

### Uvicorn
- **License:** BSD-3-Clause License
- **Copyright:** (c) 2017-present, Encode OSS Ltd.
- **URL:** https://github.com/encode/uvicorn

### PyTorch (`torch`, `torchvision`)
- **License:** Modified BSD License
- **Copyright:** (c) 2016-present Meta Platforms, Inc. and affiliates.
- **URL:** https://github.com/pytorch/pytorch

### Trimesh
- **License:** MIT License
- **Copyright:** (c) 2019 Michael Dawson-Haggerty
- **URL:** https://github.com/mikedh/trimesh

### Pydantic
- **License:** MIT License
- **Copyright:** (c) 2017-present Samuel Colvin and contributors
- **URL:** https://github.com/pydantic/pydantic

### Gradio
- **License:** Apache License 2.0
- **Copyright:** (c) 2019-present Gradio, Inc. / Hugging Face
- **URL:** https://github.com/gradio-app/gradio

### NumPy
- **License:** BSD-3-Clause License
- **Copyright:** (c) 2005-2024, NumPy Developers
- **URL:** https://github.com/numpy/numpy

### SciPy
- **License:** BSD-3-Clause License
- **Copyright:** (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers
- **URL:** https://github.com/scipy/scipy

### Psutil
- **License:** BSD-3-Clause License
- **Copyright:** (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola
- **URL:** https://github.com/giampaolo/psutil

### Pillow (PIL)
- **License:** HPND (Historical Permission Notice and Disclaimer)
- **Copyright:** (c) 2010-2024 by Alex Clark and contributors
- **URL:** https://github.com/python-pillow/Pillow

---

## 3. Blender Addon Licensing (GPLv3)

The Blender Studio Addon (`addon/triposr_blender_addon.py`) imports and links with Blender's `bpy` module. In accordance with the GNU General Public License v3 (GPLv3) governing Blender extensions, this addon is distributed under GPLv3.

The backend FastAPI server and standalone Python modules communicate with the Blender addon strictly over network/IPC boundaries (HTTP REST / JSON-RPC localhost:8000), maintaining clear architectural independence between GPL-licensed UI scripts and Apache-2.0-licensed orchestration services.
