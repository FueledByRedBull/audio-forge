"""Startup preset ID helpers for the main window."""

from ..config import BUILTIN_PRESETS

STARTUP_BUILTIN_PREFIX = "builtin:"
STARTUP_CUSTOM_PREFIX = "custom:"


def startup_builtin_id(preset_key: str) -> str:
    return f"{STARTUP_BUILTIN_PREFIX}{preset_key}"


def startup_custom_id(preset_name: str) -> str:
    return f"{STARTUP_CUSTOM_PREFIX}{preset_name}"


def normalize_startup_preset_id(value: str, custom_names: tuple[str, ...] = ()) -> str:
    """Return the stable startup preset ID, accepting legacy stored display names."""
    preset_id = str(value or "")
    if not preset_id:
        return ""
    if preset_id.startswith((STARTUP_BUILTIN_PREFIX, STARTUP_CUSTOM_PREFIX)):
        return preset_id
    if preset_id in BUILTIN_PRESETS:
        return startup_builtin_id(preset_id)
    for key, preset in BUILTIN_PRESETS.items():
        if preset.name == preset_id:
            return startup_builtin_id(key)
    if preset_id in custom_names:
        return startup_custom_id(preset_id)
    return preset_id


def startup_preset_display_name(preset_id: str) -> str:
    if preset_id.startswith(STARTUP_BUILTIN_PREFIX):
        preset_key = preset_id[len(STARTUP_BUILTIN_PREFIX):]
        if preset_key in BUILTIN_PRESETS:
            return BUILTIN_PRESETS[preset_key].name
    if preset_id.startswith(STARTUP_CUSTOM_PREFIX):
        return preset_id[len(STARTUP_CUSTOM_PREFIX):]
    return preset_id
