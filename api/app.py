"""
FastAPI Server Setup for Local Neural 3D Studio.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Local Neural 3D Studio API",
    description="Local-first 3D Generation and Editing Orchestrator (TripoSR + Qwen + Blender)",
    version="1.0.0"
)

# Allow local connections only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
