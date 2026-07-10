"""Application config schema and persistence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .settings import LatencyCalibrationProfile
from .shared import (
    DeviceIdentity,
    PresetValidationError,
    _reject_json_constant,
    build_latency_profile_key,
    coerce_device_identity,
    get_config_file,
    parse_latency_profile_key,
)
from .validation import _coerce_config_bool, _coerce_window_geometry

INPUT_CHANNEL_MODES = frozenset(
    {"average", "left", "right", "max_rms", "phase_safe_mono"}
)
INPUT_CLEANUP_MODES = frozenset({"off", "gentle", "strong"})


def _coerce_input_channel_mode(value: object) -> str:
    return value if isinstance(value, str) and value in INPUT_CHANNEL_MODES else "average"


def _coerce_input_cleanup_mode(value: object) -> str:
    return value if isinstance(value, str) and value in INPUT_CLEANUP_MODES else "off"


@dataclass
class AppConfig:
    """Application configuration (persisted settings)."""

    last_input_device: str = ""
    last_output_device: str = ""
    last_input_device_identity: DeviceIdentity | None = None
    last_output_device_identity: DeviceIdentity | None = None
    input_channel_mode: str = "average"
    input_cleanup_mode: str = "off"
    last_preset: str = ""
    startup_preset: str = ""
    window_geometry: dict | None = None
    main_splitter_sizes: list[int] | None = None
    main_control_tab_index: int = 0
    use_measured_latency: bool = True
    latency_calibration_profiles: dict[str, LatencyCalibrationProfile] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "last_input_device": self.last_input_device,
            "last_output_device": self.last_output_device,
            "last_input_device_identity": (
                self.last_input_device_identity.to_dict()
                if self.last_input_device_identity is not None
                else None
            ),
            "last_output_device_identity": (
                self.last_output_device_identity.to_dict()
                if self.last_output_device_identity is not None
                else None
            ),
            "input_channel_mode": self.input_channel_mode,
            "input_cleanup_mode": self.input_cleanup_mode,
            "last_preset": self.last_preset,
            "startup_preset": self.startup_preset,
            "window_geometry": self.window_geometry,
            "main_splitter_sizes": self.main_splitter_sizes,
            "main_control_tab_index": self.main_control_tab_index,
            "use_measured_latency": self.use_measured_latency,
            "latency_calibration_profiles": {
                key: profile.to_dict()
                for key, profile in self.latency_calibration_profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, data: object) -> "AppConfig":
        if not isinstance(data, dict):
            return cls()

        input_identity = coerce_device_identity(data.get("last_input_device_identity"))
        if input_identity is None:
            input_identity = coerce_device_identity(data.get("last_input_device"))

        output_identity = coerce_device_identity(data.get("last_output_device_identity"))
        if output_identity is None:
            output_identity = coerce_device_identity(data.get("last_output_device"))

        raw_profiles = data.get("latency_calibration_profiles", {}) or {}
        parsed_profiles: dict[str, LatencyCalibrationProfile] = {}
        if isinstance(raw_profiles, dict):
            for key, value in raw_profiles.items():
                if not isinstance(value, dict):
                    continue
                try:
                    profile = LatencyCalibrationProfile.from_dict(value)
                except (TypeError, ValueError):
                    continue
                parsed_key = str(key)
                parsed_devices = parse_latency_profile_key(parsed_key)
                if parsed_devices is not None:
                    parsed_key = build_latency_profile_key(*parsed_devices)
                parsed_profiles[parsed_key] = profile

        return cls(
            last_input_device=str(data.get("last_input_device", "") or (input_identity.name if input_identity else "")),
            last_output_device=str(data.get("last_output_device", "") or (output_identity.name if output_identity else "")),
            last_input_device_identity=input_identity,
            last_output_device_identity=output_identity,
            input_channel_mode=_coerce_input_channel_mode(data.get("input_channel_mode")),
            input_cleanup_mode=_coerce_input_cleanup_mode(data.get("input_cleanup_mode")),
            last_preset=data.get("last_preset", "") if isinstance(data.get("last_preset", ""), str) else "",
            startup_preset=data.get("startup_preset", ""),
            window_geometry=_coerce_window_geometry(data.get("window_geometry")),
            main_splitter_sizes=[
                int(size)
                for size in (data.get("main_splitter_sizes") or [])
                if isinstance(size, (int, float))
            ] or None,
            main_control_tab_index=int(data.get("main_control_tab_index", 0) or 0),
            use_measured_latency=_coerce_config_bool(data.get("use_measured_latency", True), True),
            latency_calibration_profiles=parsed_profiles,
        )


def save_config(config: AppConfig) -> None:
    filepath = get_config_file()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{filepath.name}.",
        suffix=".tmp",
        dir=filepath.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, filepath)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def load_config() -> AppConfig:
    filepath = get_config_file()
    if not filepath.exists():
        return AppConfig()

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=_reject_json_constant)
        return AppConfig.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, PresetValidationError):
        return AppConfig()


__all__ = [
    "AppConfig",
    "INPUT_CHANNEL_MODES",
    "INPUT_CLEANUP_MODES",
    "load_config",
    "save_config",
]
