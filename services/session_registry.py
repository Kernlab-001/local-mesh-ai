"""
Session-aware backend ownership registry.

The previous design tied the backend's lifetime to a single ``--parent-pid``
(the Blender instance that happened to spawn it). That produced two real bugs:

1. Closing Blender A killed a backend that Blender B was still using.
2. An instance could terminate a backend it did not own.

This module replaces the single parent-pid link with a small lease registry.
Every Blender instance registers a *session*, heartbeats while it is alive and
releases the lease on shutdown. The backend may only shut itself down when no
live session remains -- and it never kills anything it does not own.

The registry is intentionally in-process and dependency free: it is touched
from FastAPI request handlers (which run in a thread pool) and therefore all
mutations are guarded by a lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: A session that has not sent a heartbeat within this window is considered
#: stale (crashed Blender, killed process, hard power loss).
DEFAULT_STALE_AFTER_SECONDS = 90.0


@dataclass
class SessionRecord:
    """A single registered client session (one Blender instance)."""

    session_id: str
    pid: int
    label: str = ""
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    released: bool = False

    def age(self, now: Optional[float] = None) -> float:
        return (now or time.time()) - self.last_seen

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        return {
            "session_id": self.session_id,
            "pid": self.pid,
            "label": self.label,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "age_seconds": round(now - self.last_seen, 2),
            "released": self.released,
        }


class SessionRegistry:
    """Thread-safe registry of live client sessions."""

    def __init__(self, stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionRecord] = {}
        self._stale_after = stale_after_seconds

    # -- configuration -----------------------------------------------------
    @property
    def stale_after_seconds(self) -> float:
        return self._stale_after

    # -- mutation ----------------------------------------------------------
    def register(
        self,
        pid: int,
        label: str = "",
        session_id: Optional[str] = None,
    ) -> SessionRecord:
        """Registers a new client session and returns its lease record."""
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            record = SessionRecord(session_id=sid, pid=pid, label=label)
            self._sessions[sid] = record
            return record

    def heartbeat(self, session_id: str) -> bool:
        """Refreshes a session lease. Returns False when the session is unknown."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.released:
                return False
            record.last_seen = time.time()
            return True

    def release(self, session_id: str) -> bool:
        """Releases a session lease. Returns False when the session is unknown."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return False
            record.released = True
            record.last_seen = time.time()
            return True

    # -- queries -----------------------------------------------------------
    def prune_stale(self) -> List[str]:
        """Removes stale sessions and returns the ids that were pruned."""
        now = time.time()
        pruned: List[str] = []
        with self._lock:
            for sid, record in list(self._sessions.items()):
                if record.released or (now - record.last_seen) > self._stale_after:
                    pruned.append(sid)
                    self._sessions.pop(sid, None)
        return pruned

    def active_sessions(self) -> List[SessionRecord]:
        """Returns the sessions that are still alive (not released, not stale)."""
        now = time.time()
        with self._lock:
            return [
                r
                for r in self._sessions.values()
                if not r.released and (now - r.last_seen) <= self._stale_after
            ]

    def active_session_count(self) -> int:
        return len(self.active_sessions())

    def has_active_sessions(self) -> bool:
        return self.active_session_count() > 0

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._sessions.get(session_id)

    def status(self) -> Dict[str, Any]:
        """Returns a compact, secret-free status block for the health payload."""
        active = self.active_sessions()
        with self._lock:
            total = len(self._sessions)
        return {
            "active": len(active),
            "registered": total,
            "stale_after_seconds": self._stale_after,
            "sessions": [r.to_dict() for r in active],
        }

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


#: Process-wide registry used by the API layer and the backend watchdog.
session_registry = SessionRegistry()
