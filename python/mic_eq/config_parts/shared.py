"""Shared config primitives, versioning, and persistence paths."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__ as CURRENT_VERSION


logger = logging.getLogger(__name__)


APPDATA_DIR_NAME = "AudioForge"
LEGACY_APPDATA_DIR_NAME = "MicEq"
_MIGRATION_MARKER_NAME = f".{APPDATA_DIR_NAME}.migration-pending"


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


def replace_file(source: Path, destination: Path) -> None:
    """os.replace that rides out a scanner or editor briefly holding the target open."""
    attempts = 4
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            # A read-only target stays refused; waiting would only stall the GUI thread.
            if attempt == attempts or (destination.exists() and not os.access(destination, os.W_OK)):
                raise
            time.sleep(0.05 * attempt)


def _config_base_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home()))
    return Path.home() / ".config"


def _migration_marker(base_dir: Path) -> Path:
    return base_dir / _MIGRATION_MARKER_NAME


def _marker_state(marker: Path) -> str:
    """Return "" with no marker, "staged" for a published copy, else "pending".

    Empty, unreadable or unrecognised content (a crash mid-write, another build)
    counts as pending so it is retried rather than left to block saves.
    """
    try:
        text = marker.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError):
        return "pending"
    return "staged" if text == "staged" else "pending"


def migration_pending() -> bool:
    """Return whether a legacy config migration needs a safe retry."""
    base_dir = _config_base_dir()
    state = _marker_state(_migration_marker(base_dir))
    # A staged marker beside a published destination is complete even if it
    # could not be deleted.
    return bool(state) and not (
        state == "staged" and (base_dir / APPDATA_DIR_NAME).exists()
    )


def _write_migration_marker(marker: Path, state: str) -> None:
    marker.write_text(state, encoding="ascii")


def _clear_migration_marker(marker: Path) -> None:
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not clear config migration marker", exc_info=True)


def _restore_stranded_config(base_dir: Path, config_dir: Path) -> None:
    """Put back a destination a failed migration moved aside, if nothing replaced it."""
    kept = sorted(
        base_dir.glob(f".{APPDATA_DIR_NAME}.partial-*"),
        key=lambda path: path.stat().st_mtime,
    )
    if config_dir.exists() or not kept:
        return
    try:
        kept[-1].replace(config_dir)
    except OSError:
        logger.warning("Could not restore config folder %s", kept[-1], exc_info=True)
        return
    logger.warning("Restored the AudioForge config folder from %s", kept[-1])
    if len(kept) > 1:
        logger.warning("Older config folders were left in place: %s", ", ".join(map(str, kept[:-1])))


def _config_dir() -> Path:
    base_dir = _config_base_dir()
    config_dir = base_dir / APPDATA_DIR_NAME
    legacy_dir = base_dir / LEGACY_APPDATA_DIR_NAME
    marker = _migration_marker(base_dir)
    migration_failed = False
    marker_state = _marker_state(marker)

    if marker_state == "staged" and config_dir.exists():
        # A process may have been interrupted after the atomic directory swap
        # and before removing the marker. The destination is complete.
        _clear_migration_marker(marker)
        marker_state = ""
    elif marker_state and not legacy_dir.exists():
        # Nothing is left to migrate from, so the marker can only block saves.
        try:
            _restore_stranded_config(base_dir, config_dir)
        except OSError:
            logger.warning("Could not look for a stranded config folder", exc_info=True)
        _clear_migration_marker(marker)
        marker_state = ""

    needs_migration = legacy_dir.exists() and (
        not config_dir.exists() or marker_state == "pending"
    )
    if needs_migration:
        staged_dir: Path | None = None
        partial_dir: Path | None = None
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            _write_migration_marker(marker, "pending")
            # Copy into a sibling staging directory first so a failed copy
            # cannot make a partial destination look migrated on the next run.
            staged_dir = Path(
                tempfile.mkdtemp(prefix=f".{APPDATA_DIR_NAME}.migrate-", dir=base_dir)
            )
            staged_dir.rmdir()
            shutil.copytree(legacy_dir, staged_dir)
            if config_dir.exists():
                partial_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f".{APPDATA_DIR_NAME}.partial-", dir=base_dir
                    )
                )
                partial_dir.rmdir()
                config_dir.replace(partial_dir)
            staged_dir.replace(config_dir)
            if partial_dir is not None:
                logger.warning(
                    "Kept the previous AudioForge config folder at %s", partial_dir
                )
            # Mark the directory as staged only after the atomic publication.
            # A crash before this write leaves the pending marker, so the next
            # process retries instead of accepting a partial destination.
            try:
                _write_migration_marker(marker, "staged")
            except OSError:
                logger.warning("Could not mark config migration as staged", exc_info=True)
            _clear_migration_marker(marker)
        except OSError:
            migration_failed = True
            logger.warning("Legacy AudioForge config migration failed", exc_info=True)
            if staged_dir is not None and staged_dir.exists():
                try:
                    shutil.rmtree(staged_dir)
                except OSError:
                    logger.warning("Could not clean failed config migration staging", exc_info=True)
            if partial_dir is not None and partial_dir.exists() and not config_dir.exists():
                try:
                    partial_dir.replace(config_dir)
                except OSError:
                    logger.warning("Could not restore partial config destination", exc_info=True)
    # Leave the destination absent after a failed migration.  This keeps the
    # next process (or the next call) eligible to retry instead of treating an
    # empty/partial destination as an already completed migration.
    if not migration_failed:
        config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_presets_dir() -> Path:
    """Get the presets directory, creating it if necessary."""
    presets_dir = _config_dir() / "presets"
    if migration_pending():
        return presets_dir
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


def get_preset_imports_dir() -> Path:
    """Get the preset imports directory, creating it if necessary."""
    imports_dir = get_presets_dir().parent / "imports"
    if migration_pending():
        return imports_dir
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


def parse_latency_profile_key(
    key: str,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None] | None:
    """Parse a latency profile key from either legacy or structured format."""
    text = str(key)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        payload = None

    if isinstance(payload, dict) and set(payload) == {"input", "output"}:

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

    if "||" not in text:
        return None
    input_name, output_name = text.split("||", 1)
    input_device = coerce_device_identity(input_name)
    output_device = coerce_device_identity(output_name)
    if input_device is None or output_device is None:
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
    "build_device_route_key",
    "coerce_device_identity",
    "get_config_file",
    "get_preset_imports_dir",
    "get_presets_dir",
    "legacy_latency_profile_key",
    "parse_latency_profile_key",
    "migration_pending",
    "replace_file",
]
