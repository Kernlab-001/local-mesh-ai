"""
Backwards-compatible import shim for the Local Mesh AI Blender add-on.

History
-------
The add-on used to live in this single flat module. Blender only discovers
add-ons that are a *package* (a directory containing ``__init__.py``), so the
implementation now lives in ``addon/local_mesh_ai/__init__.py`` and Blender
loads it as the ``local_mesh_ai`` add-on.

This file is deliberately kept so that every historical import keeps working::

    import addon.triposr_blender_addon as addon
    addon.register()

Every public *and* private name of the real add-on is re-exported here, so
existing callers (tests, scripts, documentation examples) cannot tell the
difference. The canonical add-on source is
``addon/local_mesh_ai/__init__.py``.
"""

import os
import sys

_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
if _ADDON_DIR not in sys.path:
    sys.path.insert(0, _ADDON_DIR)

import local_mesh_ai as _impl

# Re-export every attribute (including underscore-prefixed helpers) so that
# historical call sites such as ``addon._apply_in_viewport_pbr(...)`` keep
# resolving.
globals().update({
    key: value
    for key, value in vars(_impl).items()
    if not (key.startswith("__") and key.endswith("__"))
})

__all__ = [key for key in globals() if not key.startswith("_")]
