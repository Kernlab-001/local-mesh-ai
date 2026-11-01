"""
Opt-In Data Consent Manager for Local Mesh AI.
Enforces strictly local, privacy-first data governance.
All collection is disabled by default and requires explicit user opt-in.
Zero external network calls or telemetry are ever permitted.
"""

import json
import time
from pathlib import Path

from core.path_security import get_data_dir
from core.learning_schemas import ConsentState


class DataConsentManager:
    def __init__(self):
        self.consent_file = get_data_dir() / "learning" / "consent" / "consent.json"
        self._cached_state: ConsentState = self._load_or_create_default()

    def _load_or_create_default(self) -> ConsentState:
        if self.consent_file.exists():
            try:
                with open(self.consent_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ConsentState(**data)
            except Exception:
                pass
        # Default state: Strictly opted out
        default_state = ConsentState(opted_in=False)
        return default_state

    def is_opted_in(self) -> bool:
        """Returns True only if the user has explicitly opted into local learning data collection."""
        return bool(self._cached_state.opted_in)

    def get_consent_state(self) -> ConsentState:
        """Returns the current consent state."""
        return self._cached_state

    def set_consent(
        self,
        opted_in: bool,
        anonymize: bool = True,
        retention_days: int = 90,
        export_allowed: bool = False
    ) -> ConsentState:
        """Explicitly updates user consent settings and persists them locally."""
        self.consent_file.parent.mkdir(parents=True, exist_ok=True)
        new_state = ConsentState(
            opted_in=opted_in,
            consent_timestamp=time.time() if opted_in else None,
            retention_days=retention_days,
            anonymize_mesh_names=anonymize,
            export_allowed=export_allowed,
            storage_mode="strictly_local"
        )
        self._cached_state = new_state
        with open(self.consent_file, "w", encoding="utf-8") as f:
            json.dump(new_state.model_dump(), f, indent=2)
        return new_state

    def revoke_consent(self) -> ConsentState:
        """Revokes all consent, setting opted_in to False."""
        return self.set_consent(opted_in=False)


data_consent_manager = DataConsentManager()
