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
from api.routes import router
from api.self_healing_routes import router as self_healing_router
from api.learning_routes import router as learning_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing needed
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
    version="1.0.0",
    lifespan=lifespan
)

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
