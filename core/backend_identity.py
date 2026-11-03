"""
Backend identity contract for Local Mesh AI.

A bare HTTP 200 from ``127.0.0.1:8000/health`` does **not** prove that the Local
Mesh AI backend owns that port -- any other application could be listening there.
This module is the single source of truth for the versioned identity document
that the Windows launcher and the Blender add-on verify before they report
``Connected``.

The contract is intentionally small and secret-free:

* ``application``     -- stable product identity ("Local Mesh AI")
* ``app_version``     -- product version
* ``backend_build``   -- backend build identifier
* ``api_version``     -- API contract version the add-on depends on
* ``pid``             -- OS process id of the backend
* ``started_at``      -- ISO-8601 UTC timestamp of backend start
* ``ready``           -- whether initialisation finished
* ``model_subsystem`` -- truthful model subsystem status
* ``gpu_subsystem``   -- truthful GPU subsystem status
* ``sessions``        -- registered client sessions (see services.session_registry)
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional

APPLICATION_NAME = "Local Mesh AI"
APPLICATION_VERSION = "6.0.0"
API_VERSION = "1.0.0"
BACKEND_BUILD_ID = "6.0.0-pr28"

#: The add-on refuses to talk to a backend whose major API version differs.
SUPPORTED_API_MAJORS = ("1",)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class BackendIdentity:
    """Builds the versioned identity/health payload served by the backend."""

    def __init__(
        self,
        application: str = APPLICATION_NAME,
        app_version: str = APPLICATION_VERSION,
        api_version: str = API_VERSION,
        build_id: str = BACKEND_BUILD_ID,
    ) -> None:
        self.application = application
        self.app_version = app_version
        self.api_version = api_version
        self.build_id = build_id
        self.pid = os.getpid()
        self.started_at = _utc_now_iso()
        self._ready = False

    # -- lifecycle ---------------------------------------------------------
    def mark_ready(self) -> None:
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    # -- payload -----------------------------------------------------------
    def build_payload(
        self,
        gpu_status: Optional[Dict[str, Any]] = None,
        model_status: Optional[Dict[str, Any]] = None,
        session_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Returns the complete identity document.

        ``gpu_status`` keeps the legacy ``gpu`` block so that older clients keep
        working, but the authoritative data now also lives under
        ``gpu_subsystem``.
        """
        gpu = dict(gpu_status or {})
        payload: Dict[str, Any] = {
            "application": self.application,
            "app_version": self.app_version,
            "backend_build": self.build_id,
            "api_version": self.api_version,
            "pid": self.pid,
            "started_at": self.started_at,
            "hostname": socket.gethostname(),
            "ready": self._ready,
            "status": "healthy" if self._ready else "initialising",
            "model_subsystem": model_status or {},
            "gpu_subsystem": gpu,
            "gpu": gpu,
        }
        if session_status is not None:
            payload["sessions"] = session_status
        return payload


#: Process-wide identity instance used by the API layer and the launcher probes.
backend_identity = BackendIdentity()


def verify_identity(
    payload: Any,
    expected_application: str = APPLICATION_NAME,
    expected_api_major: Optional[str] = "1",
) -> Dict[str, Any]:
    """Validates an identity payload returned by a backend.

    Returns a dictionary with ``ok`` plus a human readable ``reason`` when the
    payload is not a compatible Local Mesh AI backend. This is the single check
    used by both the launcher and the Blender add-on so the two can never
    disagree about what "Connected" means.
    """
    result: Dict[str, Any] = {"ok": False, "reason": "", "payload": payload}
    if not isinstance(payload, dict):
        result["reason"] = "Backend did not return a JSON object."
        return result

    application = payload.get("application")
    if application != expected_application:
        result["reason"] = (
            f"Port is served by a different application (application={application!r}, "
            f"expected {expected_application!r})."
        )
        return result

    api_version = payload.get("api_version")
    if not api_version:
        result["reason"] = "Backend did not report an api_version."
        return result

    if expected_api_major is not None:
        major = str(api_version).split(".", 1)[0]
        if major not in SUPPORTED_API_MAJORS:
            result["reason"] = (
                f"Incompatible backend API version {api_version} "
                f"(this add-on supports major version {expected_api_major})."
            )
            return result

    if not payload.get("ready", False):
        result["reason"] = "Backend is running but not ready yet."
        return result

    result["ok"] = True
    return result
