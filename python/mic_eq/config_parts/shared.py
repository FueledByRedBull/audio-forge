"""Shared config primitives, versioning, and persistence paths."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


CURRENT_VERSION = "1.8.4"
APPDATA_DIR_NAME = "AudioForge"
LEGACY_APPDATA_DIR_NAME = "MicEq"


class PresetValidationError(Exception):
    """Raised when preset or config validation fails with actionable detail."""


def _reject_json_constant(value: str) -> None:
    raise PresetValidationError(f"Invalid JSON constant in preset/config: {value}")


def _version_tuple(version: str) -> tuple[int, int, int]:
    """Convert semantic version string to tuple for safe ordering."""
    parts = str(version).split(".")
    normalized: list[int] = []
    for index in range(3):
        try:
            normalized.append(int(parts[index]))
        except (IndexError, ValueError):
            normalized.append(0)
    return tuple(normalized)  # type: ignore[return-value]


def _config_base_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home()))
    return Path.home() / ".config"


def _config_dir() -> Path:
    base_dir = _config_base_dir()
    config_dir = base_dir / APPDATA_DIR_NAME
    legacy_dir = base_dir / LEGACY_APPDATA_DIR_NAME
    if not config_dir.exists() and legacy_dir.exists():
        try:
            shutil.copytree(legacy_dir, config_dir)
        except OSError:
            pass
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_presets_dir() -> Path:
    """Get the presets directory, creating it if necessary."""
    presets_dir = _config_dir() / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


def get_preset_imports_dir() -> Path:
    """Get the preset imports directory, creating it if necessary."""
    imports_dir = get_presets_dir().parent / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    return imports_dir


def get_config_file() -> Path:
    """Get the main config file path."""
    return _config_dir() / "config.json"


@dataclass(slots=True)
class DeviceIdentity:
    """Persisted audio device identity used by the UI/config layer."""

    name: str = ""
    is_default: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, data: dict | str | DeviceIdentity | None) -> DeviceIdentity | None:
        if isinstance(data, cls):
            return data if data.name else None
        if isinstance(data, str):
            name = data.strip()
            return cls(name=name) if name else None
        if not isinstance(data, dict):
            return None

        name = str(data.get("name", "")).strip()
        if not name:
            return None
        return cls(name=name, is_default=bool(data.get("is_default", False)))


def coerce_device_identity(data: object) -> DeviceIdentity | None:
    """Normalize persisted device identity data from legacy or structured inputs."""
    if isinstance(data, (DeviceIdentity, dict, str)) or data is None:
        return DeviceIdentity.from_dict(data)
    return None


def legacy_latency_profile_key(input_name: str, output_name: str) -> str:
    """Legacy latency profile key based on friendly device names."""
    return f"{input_name}||{output_name}"


def build_latency_profile_key(
    input_device: DeviceIdentity | None,
    output_device: DeviceIdentity | None,
) -> str:
    """Build a deterministic latency profile key from structured device identities."""
    payload = {
        "input": input_device.to_dict() if input_device is not None else None,
        "output": output_device.to_dict() if output_device is not None else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_latency_profile_key(
    key: str,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None] | None:
    """Parse a latency profile key from either legacy or structured format."""
    text = str(key)
    if "||" in text:
        input_name, output_name = text.split("||", 1)
        return (
            coerce_device_identity(input_name),
            coerce_device_identity(output_name),
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    return (
        coerce_device_identity(payload.get("input")),
        coerce_device_identity(payload.get("output")),
    )


__all__ = [
    "APPDATA_DIR_NAME",
    "CURRENT_VERSION",
    "DeviceIdentity",
    "LEGACY_APPDATA_DIR_NAME",
    "PresetValidationError",
    "_reject_json_constant",
    "_version_tuple",
    "build_latency_profile_key",
    "coerce_device_identity",
    "get_config_file",
    "get_preset_imports_dir",
    "get_presets_dir",
    "legacy_latency_profile_key",
    "parse_latency_profile_key",
]
