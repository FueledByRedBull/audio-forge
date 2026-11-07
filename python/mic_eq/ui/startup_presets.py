"""Startup preset ID helpers for the main window."""

from ..config import BUILTIN_PRESETS
from ..config_parts.presets import RefusedPreset

STARTUP_BUILTIN_PREFIX = "builtin:"
STARTUP_CUSTOM_PREFIX = "custom:"
STARTUP_CUSTOM_FILE_PREFIX = "custom-file:"


def startup_builtin_id(preset_key: str) -> str:
    return f"{STARTUP_BUILTIN_PREFIX}{preset_key}"


def startup_custom_id(preset_name: str) -> str:
    return f"{STARTUP_CUSTOM_PREFIX}{preset_name}"


def startup_custom_file_id(filename: str) -> str:
    return f"{STARTUP_CUSTOM_FILE_PREFIX}{filename}"


def normalize_startup_preset_id(value: str, custom_names: tuple[str, ...] = ()) -> str:
    """Return the stable startup preset ID, accepting legacy stored display names."""
    preset_id = str(value or "")
    if not preset_id:
        return ""
    if preset_id.startswith(
        (STARTUP_BUILTIN_PREFIX, STARTUP_CUSTOM_PREFIX, STARTUP_CUSTOM_FILE_PREFIX)
    ):
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
    if preset_id.startswith(STARTUP_CUSTOM_FILE_PREFIX):
        return preset_id[len(STARTUP_CUSTOM_FILE_PREFIX):]
    return preset_id


def refusal_reason_for_preset_id(
    preset_id: str, refused: list[RefusedPreset]
) -> str | None:
    """Why the one preset file behind a custom preset ID is refused for age, if it is."""
    if preset_id.startswith(STARTUP_CUSTOM_FILE_PREFIX):
        identity = preset_id[len(STARTUP_CUSTOM_FILE_PREFIX):]
        matches = [item for item in refused if item.path.name == identity]
    elif preset_id.startswith(STARTUP_CUSTOM_PREFIX):
        identity = preset_id[len(STARTUP_CUSTOM_PREFIX):]
        matches = [
            item for item in refused if item.name == identity or item.path.name == identity
        ]
    else:
        matches = [item for item in refused if item.name == preset_id]
    return matches[0].reason if len(matches) == 1 else None
