"""Application config schema and persistence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .settings import LatencyCalibrationProfile
from .shared import (
    DeviceIdentity,
    PresetValidationError,
    _reject_json_constant,
    build_device_route_key,
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
DYNAMICS_INTENSITIES = frozenset({"gentle", "balanced", "dense", "custom"})
DEVICE_PRESET_PROVENANCE = frozenset({"explicit_user", "legacy_migration"})
FIRST_RUN_SETUP_STATES = frozenset(
    {"not_started", "in_progress", "completed", "completed_with_skips"}
)
FIRST_RUN_SETUP_STEPS = ("devices", "route", "latency", "voice")
FIRST_RUN_STEP_STATES = frozenset({"pending", "completed", "skipped"})
_PRE_SETUP_CONFIG_FIELDS = frozenset(
    {
        "last_input_device",
        "last_output_device",
        "last_input_device_identity",
        "last_output_device_identity",
        "input_channel_mode",
        "input_cleanup_mode",
        "last_preset",
        "startup_preset",
        "window_geometry",
        "main_splitter_sizes",
        "main_control_tab_index",
        "use_measured_latency",
        "voice_setup_dynamics_intensity",
        "voice_setup_custom_p95_db",
        "voice_setup_custom_peak_cap_db",
        "latency_calibration_profiles",
        "auto_apply_device_presets",
        "device_preset_bindings",
    }
)


@dataclass(frozen=True, slots=True)
class DevicePresetBinding:
    """Reference a saved DSP preset without embedding route preferences in it."""

    preset_id: str
    provenance: str = "explicit_user"

    def to_dict(self) -> dict[str, str]:
        return {"preset_id": self.preset_id, "provenance": self.provenance}

    @classmethod
    def from_value(cls, value: object) -> "DevicePresetBinding | None":
        if isinstance(value, str):
            preset_id = value.strip()
            provenance = "legacy_migration"
        elif isinstance(value, dict):
            preset_id = str(value.get("preset_id", "")).strip()
            provenance = str(value.get("provenance", "explicit_user")).strip()
        else:
            return None
        if not preset_id.startswith(("builtin:", "custom:")):
            return None
        if provenance not in DEVICE_PRESET_PROVENANCE:
            provenance = "legacy_migration"
        return cls(preset_id=preset_id, provenance=provenance)


def _coerce_input_channel_mode(value: object) -> str:
    return value if isinstance(value, str) and value in INPUT_CHANNEL_MODES else "average"


def _coerce_input_cleanup_mode(value: object) -> str:
    return value if isinstance(value, str) and value in INPUT_CLEANUP_MODES else "off"


def _coerce_float(value: object, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not low <= parsed <= high:
        return default
    return parsed


def _coerce_int(value: object, default: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric) or not numeric.is_integer():
        return default
    parsed = int(numeric)
    return parsed if low <= parsed <= high else default


def _coerce_splitter_sizes(value: object) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    parsed = [_coerce_int(size, -1, 0, 1_000_000) for size in value]
    return parsed if all(size >= 0 for size in parsed) else None


def _coerce_dynamics_intensity(value: object) -> str:
    return value if isinstance(value, str) and value in DYNAMICS_INTENSITIES else "balanced"


def _coerce_first_run_steps(value: object) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    return {
        step: (
            str(source.get(step))
            if str(source.get(step)) in FIRST_RUN_STEP_STATES
            else "pending"
        )
        for step in FIRST_RUN_SETUP_STEPS
    }


def _coerce_device_name(
    value: object,
    identity: DeviceIdentity | None,
) -> str:
    if isinstance(value, str):
        name = value.strip()
        if name and len(name) <= 4096 and "\x00" not in name:
            return name
    return identity.name if identity is not None else ""


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
    voice_setup_dynamics_intensity: str = "balanced"
    voice_setup_custom_p95_db: float = 3.5
    voice_setup_custom_peak_cap_db: float = 8.0
    latency_calibration_profiles: dict[str, LatencyCalibrationProfile] = field(default_factory=dict)
    auto_apply_device_presets: bool = True
    device_preset_bindings: dict[str, DevicePresetBinding] = field(default_factory=dict)
    first_run_setup_state: str = "not_started"
    first_run_setup_step: str = "devices"
    first_run_setup_steps: dict[str, str] = field(
        default_factory=lambda: {step: "pending" for step in FIRST_RUN_SETUP_STEPS}
    )

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
            "voice_setup_dynamics_intensity": self.voice_setup_dynamics_intensity,
            "voice_setup_custom_p95_db": self.voice_setup_custom_p95_db,
            "voice_setup_custom_peak_cap_db": self.voice_setup_custom_peak_cap_db,
            "latency_calibration_profiles": {
                key: profile.to_dict()
                for key, profile in self.latency_calibration_profiles.items()
            },
            "auto_apply_device_presets": self.auto_apply_device_presets,
            "device_preset_bindings": {
                key: binding.to_dict()
                for key, binding in self.device_preset_bindings.items()
            },
            "first_run_setup_state": self.first_run_setup_state,
            "first_run_setup_step": self.first_run_setup_step,
            "first_run_setup_steps": dict(self.first_run_setup_steps),
        }

    @classmethod
    def from_dict(cls, data: object) -> "AppConfig":
        if not isinstance(data, dict):
            return cls()

        migrated_existing_install = (
            "first_run_setup_state" not in data
            and bool(_PRE_SETUP_CONFIG_FIELDS.intersection(data))
        )

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
                if parsed_devices is None:
                    continue
                parsed_key = build_latency_profile_key(*parsed_devices)
                parsed_profiles[parsed_key] = profile

        raw_bindings = data.get("device_preset_bindings", {}) or {}
        parsed_bindings: dict[str, DevicePresetBinding] = {}
        if isinstance(raw_bindings, dict):
            for key, value in raw_bindings.items():
                binding = DevicePresetBinding.from_value(value)
                if binding is None:
                    continue
                parsed_key = str(key)
                parsed_devices = parse_latency_profile_key(parsed_key)
                if parsed_devices is None:
                    continue
                parsed_key = build_device_route_key(*parsed_devices)
                parsed_bindings[parsed_key] = binding

        first_run_steps = _coerce_first_run_steps(
            data.get("first_run_setup_steps")
        )
        if migrated_existing_install and "first_run_setup_steps" not in data:
            first_run_steps = {
                step: "skipped" for step in FIRST_RUN_SETUP_STEPS
            }

        return cls(
            last_input_device=_coerce_device_name(
                data.get("last_input_device"), input_identity
            ),
            last_output_device=_coerce_device_name(
                data.get("last_output_device"), output_identity
            ),
            last_input_device_identity=input_identity,
            last_output_device_identity=output_identity,
            input_channel_mode=_coerce_input_channel_mode(data.get("input_channel_mode")),
            input_cleanup_mode=_coerce_input_cleanup_mode(data.get("input_cleanup_mode")),
            last_preset=data.get("last_preset", "") if isinstance(data.get("last_preset", ""), str) else "",
            startup_preset=(
                data.get("startup_preset", "")
                if isinstance(data.get("startup_preset", ""), str)
                else ""
            ),
            window_geometry=_coerce_window_geometry(data.get("window_geometry")),
            main_splitter_sizes=_coerce_splitter_sizes(
                data.get("main_splitter_sizes")
            ),
            main_control_tab_index=_coerce_int(
                data.get("main_control_tab_index", 0), 0, 0, 64
            ),
            use_measured_latency=_coerce_config_bool(data.get("use_measured_latency", True), True),
            voice_setup_dynamics_intensity=_coerce_dynamics_intensity(
                data.get("voice_setup_dynamics_intensity")
            ),
            voice_setup_custom_p95_db=_coerce_float(
                data.get("voice_setup_custom_p95_db"),
                3.5,
                1.0,
                8.0,
            ),
            voice_setup_custom_peak_cap_db=_coerce_float(
                data.get("voice_setup_custom_peak_cap_db"),
                8.0,
                1.5,
                12.0,
            ),
            latency_calibration_profiles=parsed_profiles,
            auto_apply_device_presets=_coerce_config_bool(
                data.get("auto_apply_device_presets", True), True
            ),
            device_preset_bindings=parsed_bindings,
            first_run_setup_state=(
                "completed_with_skips"
                if migrated_existing_install
                else (
                    str(data.get("first_run_setup_state"))
                    if str(data.get("first_run_setup_state"))
                    in FIRST_RUN_SETUP_STATES
                    else "not_started"
                )
            ),
            first_run_setup_step=(
                str(data.get("first_run_setup_step"))
                if str(data.get("first_run_setup_step")) in FIRST_RUN_SETUP_STEPS
                else "devices"
            ),
            first_run_setup_steps=first_run_steps,
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
            json.dump(config.to_dict(), handle, indent=2, allow_nan=False)
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
    except (
        OSError,
        OverflowError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        PresetValidationError,
    ):
        return AppConfig()


__all__ = [
    "AppConfig",
    "DevicePresetBinding",
    "INPUT_CHANNEL_MODES",
    "INPUT_CLEANUP_MODES",
    "DYNAMICS_INTENSITIES",
    "FIRST_RUN_SETUP_STATES",
    "FIRST_RUN_SETUP_STEPS",
    "load_config",
    "save_config",
]
