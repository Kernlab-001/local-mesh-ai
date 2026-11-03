"""
Blender start-up bootstrap for Local Mesh AI.

Executed by ``Start_Local_Mesh_AI.bat`` through::

    blender.exe --python launchers/blender_bootstrap.py

Blender 5.x removed the ``--addons-path`` command line option that the old
launcher relied on, and Blender only discovers add-ons that are a *package*
(a directory containing ``__init__.py``). This script therefore performs the
add-on installation and activation itself:

  1. copies the add-on package into Blender's own add-ons folder,
  2. disables the superseded flat-file add-on (``triposr_blender_addon``) so
     the two cannot fight over the same operator ids,
  3. enables the ``local_mesh_ai`` add-on,
  4. points the add-on at the project root so it can find and start the
     backend,
  5. verifies the backend *identity* -- an HTTP 200 on port 8000 is not proof
     that Local Mesh AI is running.

Every step is written to Blender's console and to ``logs/blender-bootstrap.log``
so that a failed start-up is visible instead of silent.
"""

import os
import shutil
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "blender-bootstrap.log")

ADDON_MODULE = "local_mesh_ai"
LEGACY_ADDON_MODULE = "triposr_blender_addon"
SOURCE_PACKAGE = os.path.join(PROJECT_ROOT, "addon", ADDON_MODULE)


def _log(message):
    """Prints to Blender's console and appends to the persistent log."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    try:
        print(line)
    except Exception:
        pass
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def _copy_package(source, target):
    """Copies the add-on package, replacing the target directory."""
    if not os.path.isdir(source):
        _log(f"Add-on source package not found: {source}")
        return False
    try:
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        return True
    except Exception as exc:
        _log(f"Could not install add-on into {target}: {exc}")
        return False


def install_addon():
    """Installs the add-on package into Blender's add-ons folder."""
    import bpy
    try:
        addons_root = bpy.utils.user_resource('SCRIPTS', path='addons', create=True)
    except Exception as exc:
        _log(f"Could not resolve Blender's add-ons folder: {exc}")
        return None
    target = os.path.join(addons_root, ADDON_MODULE)
    if _copy_package(SOURCE_PACKAGE, target):
        _log(f"Add-on installed to: {target}")
    return target


def retire_legacy_addon():
    """Retires the superseded flat-file add-on.

    Two steps are needed:

    1. ``addon_utils.disable()`` unregisters the legacy classes and clears the
       stale "enabled" flag from the user preferences.
    2. The legacy ``triposr_blender_addon.py`` file is *moved aside* (never
       deleted) so Blender cannot load it again on the next start. Both add-ons
       declare the same operator ids, so loading both would make Blender
       silently re-register the new classes over the old ones.

    The original file is preserved as ``triposr_blender_addon.py.bak`` so the
    owner's previous install is never lost.
    """
    import bpy
    import addon_utils

    try:
        if bpy.context.preferences.addons.get(LEGACY_ADDON_MODULE) is not None:
            addon_utils.disable(LEGACY_ADDON_MODULE, default_set=True)
            _log(f"Disabled legacy add-on '{LEGACY_ADDON_MODULE}' in the user preferences.")
    except Exception as exc:
        _log(f"Could not disable the legacy add-on: {exc}")

    try:
        addons_root = bpy.utils.user_resource('SCRIPTS', path='addons', create=True)
    except Exception as exc:
        _log(f"Could not resolve Blender's add-ons folder: {exc}")
        return

    legacy_file = os.path.join(addons_root, LEGACY_ADDON_MODULE + ".py")
    if os.path.exists(legacy_file):
        backup = legacy_file + ".bak"
        try:
            if os.path.exists(backup):
                os.replace(legacy_file, legacy_file + ".bak.old")
                backup = legacy_file + ".bak.old"
            else:
                os.replace(legacy_file, backup)
            _log(f"Legacy add-on file moved aside (kept as): {backup}")
        except Exception as exc:
            _log(f"Could not move the legacy add-on file aside: {exc}")


def enable_addon():
    """Enables the Local Mesh AI add-on and returns the module (or None)."""
    import addon_utils
    import bpy

    if ADDON_MODULE in bpy.context.preferences.addons:
        _log(f"Add-on '{ADDON_MODULE}' is already enabled.")
        return sys.modules.get(ADDON_MODULE)

    try:
        module = addon_utils.enable(ADDON_MODULE, default_set=True, persistent=True)
        if module is not None:
            _log(f"Add-on enabled: {ADDON_MODULE}")
            return module
    except Exception as exc:
        _log(f"Failed to enable add-on '{ADDON_MODULE}': {exc}")
        _log(traceback.format_exc())

    # Fall back to importing the package directly from the project checkout.
    addon_src = os.path.join(PROJECT_ROOT, "addon")
    if addon_src not in sys.path:
        sys.path.insert(0, addon_src)
    try:
        import local_mesh_ai as module  # type: ignore
        module.register()
        _log("Add-on registered directly from the project checkout.")
        return module
    except Exception as exc:
        _log(f"Fallback registration failed: {exc}")
        _log(traceback.format_exc())
        return None


def configure_preferences(module):
    """Points the add-on at the project root."""
    import bpy
    try:
        entry = bpy.context.preferences.addons.get(ADDON_MODULE)
        if entry is not None and getattr(entry, "preferences", None) is not None:
            entry.preferences.project_dir = PROJECT_ROOT
            _log(f"Add-on project root set to: {PROJECT_ROOT}")
    except Exception as exc:
        _log(f"Could not set add-on preferences: {exc}")


def verify_backend(module):
    """Verifies the backend identity and reports the real state."""
    if module is None:
        _log("BACKEND: unknown (add-on not loaded)")
        return False
    try:
        probe = module.probe_backend(module.DEFAULT_SERVER_URL, timeout=2.5)
    except Exception as exc:
        _log(f"BACKEND: probe failed: {exc}")
        return False
    if probe["ok"]:
        identity = probe.get("identity") or {}
        _log(
            "BACKEND: CONNECTED to {app} v{ver} (api {api}, build {build}, pid {pid}) "
            "in {lat:.1f} ms".format(
                app=identity.get("application", "?"),
                ver=identity.get("app_version", "?"),
                api=identity.get("api_version", "?"),
                build=identity.get("backend_build", "?"),
                pid=identity.get("pid", "?"),
                lat=probe["latency_ms"],
            )
        )
        return True
    _log(f"BACKEND: NOT READY - {probe['reason']}")
    _log("BACKEND: the add-on will start it automatically; use 'Start Backend Terminal' if needed.")
    return False


def main():
    import bpy

    _log("=" * 68)
    _log(f"Local Mesh AI bootstrap | Blender {bpy.app.version_string} | project {PROJECT_ROOT}")

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    # The add-on uses this to locate run_server.py without any manual setup.
    os.environ["LOCAL_MESH_AI_PROJECT_ROOT"] = PROJECT_ROOT
    os.environ.setdefault("LOCAL_MESH_AI_DIR", PROJECT_ROOT)

    install_addon()
    retire_legacy_addon()
    module = enable_addon()
    configure_preferences(module)
    verify_backend(module)

    _log("Bootstrap complete.")
    return module


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401  (Blender-only module)
    except Exception:  # pragma: no cover - running outside Blender
        bpy = None
    if bpy is not None:
        main()
