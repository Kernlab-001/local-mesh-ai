"""
FastAPI Server Setup for Local Neural 3D Studio.
Hardened for local execution with restricted CORS, rate/origin boundaries,
and clean lifespan resource shutdown.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from core.backend_identity import backend_identity
from api.routes import router
from api.self_healing_routes import router as self_healing_router
from api.learning_routes import router as learning_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: publish the versioned backend identity so that clients can
    # verify they are talking to Local Mesh AI and not to some other service
    # that happens to own port 8000.
    try:
        from core.backend_identity import backend_identity
        backend_identity.mark_ready()
        print(f"[IDENTITY] {backend_identity.application} v{backend_identity.app_version} "
              f"api={backend_identity.api_version} build={backend_identity.build_id} "
              f"pid={backend_identity.pid} started={backend_identity.started_at}")
    except Exception:
        pass
    yield
    # Shutdown: ensure background executor workers and GPU resources are safely released
    try:
        from services.job_manager import job_manager
        job_manager.executor.shutdown(wait=False)
    except Exception:
        pass

    try:
        from core.gpu_broker import gpu_broker
        gpu_broker.purge_vram()
    except Exception:
        pass

app = FastAPI(
    title="Local Neural 3D Studio API",
    description="Local-first 3D Generation and Editing Orchestrator (TripoSR + Qwen + Blender)",
    version=backend_identity.api_version,
    lifespan=lifespan
)

# The routers are mounted at import time and there is no asynchronous start-up
# work, so the backend is ready to serve as soon as the app exists. The
# lifespan calls mark_ready() again, which is idempotent.
backend_identity.mark_ready()

# Restrict to trusted local browser / frontend origins (prevent malicious cross-origin execution)
TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:7860",
    "http://localhost:7860"
]

# Allow custom origins via environment variable if user explicitly hosts remotely
custom_origins = os.environ.get("ALLOWED_ORIGINS")
if custom_origins:
    TRUSTED_ORIGINS.extend([o.strip() for o in custom_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=TRUSTED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(self_healing_router)
app.include_router(learning_router)

static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
