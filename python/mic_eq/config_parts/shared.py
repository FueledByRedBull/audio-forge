"""Shared config primitives, versioning, and persistence paths."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


CURRENT_VERSION = "1.11.2"
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
    # Default-route status is transient policy, not part of endpoint identity.
    is_default: bool = field(default=False, compare=False)
    endpoint_id: str = ""
    host_api: str = ""
    direction: str = ""
    # Mutable Windows format fields are diagnostic evidence, not route-key
    # material; changing endpoint format must not orphan a binding.
    sample_rate: int | None = None
    channels: int | None = None
    name_ordinal: int | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_default": self.is_default,
            "endpoint_id": self.endpoint_id,
            "host_api": self.host_api,
            "direction": self.direction,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "name_ordinal": self.name_ordinal,
        }

    def stable_dict(self) -> dict:
        """Return rename-stable fields suitable for route/profile keys."""
        if self.endpoint_id:
            return {
                "endpoint_id": self.endpoint_id,
                "host_api": self.host_api.casefold(),
                "direction": self.direction.casefold(),
            }
        result: dict[str, object] = {
            "name": " ".join(self.name.casefold().split()),
        }
        if self.host_api:
            result["host_api"] = self.host_api.casefold()
        if self.direction:
            result["direction"] = self.direction.casefold()
        if self.name_ordinal is not None:
            result["name_ordinal"] = self.name_ordinal
        return result

    @classmethod
    def from_dict(
        cls, data: dict | str | DeviceIdentity | None
    ) -> DeviceIdentity | None:
        if isinstance(data, cls):
            return data if data.name else None
        if isinstance(data, str):
            name = data.strip()
            return cls(name=name) if name else None
        if not isinstance(data, dict):
            return None

        raw_name = data.get("name", "")
        if not isinstance(raw_name, str):
            return None
        name = raw_name.strip()
        if not name:
            return None

        def optional_non_negative_int(
            value: object,
            maximum: int,
        ) -> int | None:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(numeric) or not numeric.is_integer():
                return None
            parsed = int(numeric)
            return parsed if 0 <= parsed <= maximum else None

        raw_direction = data.get("direction", "")
        direction = (
            raw_direction.strip().casefold()
            if isinstance(raw_direction, str)
            else ""
        )
        if direction not in {"", "input", "output"}:
            direction = ""
        endpoint_id = data.get("endpoint_id", "")
        host_api = data.get("host_api", "")
        return cls(
            name=name,
            is_default=(
                data.get("is_default", False)
                if isinstance(data.get("is_default", False), bool)
                else False
            ),
            endpoint_id=(endpoint_id.strip() if isinstance(endpoint_id, str) else ""),
            host_api=(host_api.strip() if isinstance(host_api, str) else ""),
            direction=direction,
            sample_rate=optional_non_negative_int(data.get("sample_rate"), 0xFFFFFFFF),
            channels=optional_non_negative_int(data.get("channels"), 0xFFFF),
            name_ordinal=optional_non_negative_int(
                data.get("name_ordinal"), 0xFFFFFFFF
            ),
        )


def coerce_device_identity(data: object) -> DeviceIdentity | None:
    """Normalize persisted device identity data from legacy or structured inputs."""
    if isinstance(data, (DeviceIdentity, dict, str)) or data is None:
        return DeviceIdentity.from_dict(data)
    return None


def legacy_latency_profile_key(input_name: str, output_name: str) -> str:
    """Legacy latency profile key based on friendly device names."""
    return f"{input_name}||{output_name}"


def build_device_route_key(
    input_device: DeviceIdentity | None,
    output_device: DeviceIdentity | None,
) -> str:
    """Build a deterministic key for an input/output endpoint pair."""
    payload = {
        "input": input_device.stable_dict() if input_device is not None else None,
        "output": output_device.stable_dict() if output_device is not None else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_latency_profile_key(
    input_device: DeviceIdentity | None,
    output_device: DeviceIdentity | None,
) -> str:
    """Build a deterministic latency key without transient default-route state."""
    return build_device_route_key(input_device, output_device)


def parse_latency_profile_key(
    key: str,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None] | None:
    """Parse a latency profile key from either legacy or structured format."""
    text = str(key)
    if "||" in text:
        input_name, output_name = text.split("||", 1)
        input_device = coerce_device_identity(input_name)
        output_device = coerce_device_identity(output_name)
        if input_device is None or output_device is None:
            return None
        return input_device, output_device

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or set(payload) != {"input", "output"}:
        return None

    def parse_route_identity(value: object) -> tuple[DeviceIdentity | None, bool]:
        if value is None:
            return None, True
        candidate = value
        if (
            isinstance(value, dict)
            and value.get("endpoint_id")
            and not value.get("name")
        ):
            # Stable endpoint route keys intentionally omit rename-prone names.
            # Supply a non-persisted placeholder so the ordinary identity
            # validator can canonicalize the endpoint fields again.
            candidate = {
                **value,
                "name": f"endpoint:{value['endpoint_id']}",
            }
        parsed = coerce_device_identity(candidate)
        return parsed, parsed is not None

    input_device, input_valid = parse_route_identity(payload.get("input"))
    output_device, output_valid = parse_route_identity(payload.get("output"))
    if (
        not input_valid
        or not output_valid
        or input_device is None
        or output_device is None
    ):
        return None
    return input_device, output_device


__all__ = [
    "APPDATA_DIR_NAME",
    "CURRENT_VERSION",
    "DeviceIdentity",
    "LEGACY_APPDATA_DIR_NAME",
    "PresetValidationError",
    "_reject_json_constant",
    "_version_tuple",
    "build_latency_profile_key",
    "build_device_route_key",
    "coerce_device_identity",
    "get_config_file",
    "get_preset_imports_dir",
    "get_presets_dir",
    "legacy_latency_profile_key",
    "parse_latency_profile_key",
]
