"""Privacy-safe, versioned AudioForge support snapshots."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import platform
import secrets
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "audioforge-support-snapshot"
SCHEMA_VERSION = 1
MAX_SERIALIZED_BYTES = 128 * 1024
MIN_PSEUDONYM_KEY_BYTES = 16

_CONFIG_ENUMS = {
    "input_channel_mode": frozenset(
        {"average", "left", "right", "max_rms", "phase_safe_mono"}
    ),
    "input_cleanup_mode": frozenset({"off", "gentle", "strong"}),
    "voice_setup_dynamics_intensity": frozenset(
        {"gentle", "balanced", "dense", "custom"}
    ),
    "first_run_setup_state": frozenset(
        {"not_started", "in_progress", "completed", "completed_with_skips"}
    ),
    "first_run_setup_step": frozenset(
        {"devices", "route", "latency", "voice"}
    ),
}
_CONFIG_NUMBERS = (
    "main_control_tab_index",
    "voice_setup_custom_p95_db",
    "voice_setup_custom_peak_cap_db",
)
_CONFIG_BOOLS = ("use_measured_latency", "auto_apply_device_presets")

_PROCESSING_FIELDS = {
    "gate": frozenset(
        {
            "enabled",
            "threshold_db",
            "attack_ms",
            "release_ms",
            "gate_mode",
            "vad_threshold",
            "vad_hold_time_ms",
            "vad_pre_gain",
            "auto_threshold_enabled",
            "gate_margin_db",
        }
    ),
    "eq": frozenset({"schema_version", "enabled", "bands"}),
    "rnnoise": frozenset({"enabled", "strength", "model"}),
    "deesser": frozenset(
        {
            "enabled",
            "auto_enabled",
            "auto_amount",
            "low_cut_hz",
            "high_cut_hz",
            "threshold_db",
            "ratio",
            "attack_ms",
            "release_ms",
            "max_reduction_db",
        }
    ),
    "compressor": frozenset(
        {
            "enabled",
            "threshold_db",
            "ratio",
            "attack_ms",
            "release_ms",
            "makeup_gain_db",
            "adaptive_release",
            "base_release_ms",
            "auto_makeup_enabled",
            "target_lufs",
            "sidechain_highpass_enabled",
        }
    ),
    "limiter": frozenset(
        {
            "enabled",
            "ceiling_db",
            "release_ms",
            "careful_output_enabled",
        }
    ),
}
_PROCESSING_ENUMS = {
    ("rnnoise", "model"): frozenset(
        {"rnnoise", "deepfilter-ll", "deepfilter"}
    ),
}
_EQ_BAND_FIELDS = frozenset(
    {
        "filter_type",
        "frequency_hz",
        "gain_db",
        "q",
        "bandwidth_mode",
        "bandwidth_octaves",
        "slope_db_per_octave",
        "stage",
        "enabled",
    }
)
_EQ_BAND_ENUMS = {
    "filter_type": frozenset(
        {
            "bell",
            "notch",
            "low_shelf",
            "high_shelf",
            "high_pass",
            "low_pass",
        }
    ),
    "bandwidth_mode": frozenset({"q", "octaves"}),
    "stage": frozenset({"combined", "correction", "tone"}),
}
_RUNTIME_NUMBERS = frozenset(
    {
        "input_dropped_samples",
        "input_backlog_dropped_samples",
        "input_backlog_recovery_count",
        "input_callback_error_count",
        "input_fixed_buffer_frames",
        "input_phase_estimated_delay_samples",
        "input_phase_warning_count",
        "input_stereo_correlation",
        "jitter_dropped_samples",
        "lock_contention_count",
        "output_callback_error_count",
        "output_clip_event_count",
        "output_clip_peak_db",
        "output_fixed_buffer_frames",
        "output_recovery_count",
        "output_recovery_event_count",
        "output_resampler_delay_samples",
        "output_retime_adjustment_count",
        "output_sample_rate",
        "output_short_term_lufs",
        "output_short_write_dropped_samples",
        "output_true_peak_db",
        "output_true_peak_event_count",
        "output_true_peak_gain_reduction_db",
        "output_true_peak_headroom_db",
        "output_underrun_streak",
        "output_underrun_total",
        "rt_buffer_overflow_count",
        "stream_restart_count",
        "suppressor_non_finite_count",
        "total_latency_ms",
        "engine_latency_ms",
        "route_latency_ms",
        "gate_chatter_event_count",
        "gate_fused_score",
        "deesser_detector_confidence",
        "limiter_gain_reduction_db",
        "limiter_peak_gain_reduction_db",
        "noise_attenuation_limit_db",
        "noise_post_filter_beta",
    }
)
_RUNTIME_BOOLS = frozenset(
    {
        "gate_auto_relax_active",
        "input_cleanup_hum_detected",
        "input_cleanup_rumble_detected",
        "input_phase_polarity_flipped",
        "input_resampler_active",
        "limiter_careful_output_enabled",
        "noise_backend_available",
        "noise_backend_failed",
        "output_resampler_active",
        "raw_monitor_enabled",
        "recovery_suppressed",
    }
)
_RUNTIME_ENUMS = {
    "input_channel_mode": frozenset(
        {"average", "left", "right", "max_rms", "phase_safe_mono"}
    ),
    "input_cleanup_mode": frozenset({"off", "gentle", "strong"}),
    "input_phase_rescue_strategy": frozenset(
        {"none", "delay_left", "delay_right", "polarity_flip"}
    ),
    "noise_model": frozenset({"rnnoise", "deepfilter-ll", "deepfilter"}),
    "rt_error_name": frozenset(
        {
            "none",
            "input_queue_full",
            "output_queue_full",
            "non_finite",
            "processor_unavailable",
        }
    ),
}
_ROOT_FIELDS = frozenset(
    {
        "schema",
        "generated_at_utc",
        "application",
        "system",
        "audio_engine",
        "configuration",
        "processing",
        "runtime",
        "privacy",
    }
)
_SYSTEM_FIELDS = frozenset(
    {
        "operating_system",
        "os_release",
        "os_version",
        "architecture",
        "python_version",
        "python_implementation",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        *_CONFIG_ENUMS,
        *_CONFIG_NUMBERS,
        *_CONFIG_BOOLS,
        "saved_latency_profile_count",
        "device_preset_binding_count",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        *_RUNTIME_NUMBERS,
        *_RUNTIME_BOOLS,
        *_RUNTIME_ENUMS,
        "backend_error_present",
        "stream_error_present",
        "restart_reason_present",
    }
)


def _mapping_value(source: object, key: str) -> object:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _safe_enum(value: object, allowed: frozenset[str]) -> str:
    return str(value) if isinstance(value, str) and value in allowed else "other"


def _device_value(device: object, key: str, default: object = None) -> object:
    if isinstance(device, Mapping):
        return device.get(key, default)
    return getattr(device, key, default)


def _device_fields(device: object) -> tuple[str, bool] | None:
    if device is None:
        return None
    if isinstance(device, str):
        name = device.strip()
        return (f"name:{name.casefold()}", False) if name else None

    name = str(_device_value(device, "name", "")).strip()
    if not name:
        return None
    is_default = bool(_device_value(device, "is_default", False))
    endpoint_id = str(_device_value(device, "endpoint_id", "") or "").strip()
    host_api = str(_device_value(device, "host_api", "") or "").strip().casefold()
    direction = str(_device_value(device, "direction", "") or "").strip().casefold()
    if endpoint_id:
        identity = f"endpoint:{host_api}:{direction}:{endpoint_id}"
    else:
        ordinal = _device_value(device, "name_ordinal", None)
        sample_rate = _device_value(device, "sample_rate", None)
        channels = _device_value(device, "channels", None)
        identity = (
            f"fallback:{host_api}:{direction}:{name.casefold()}:"
            f"{ordinal!r}:{sample_rate!r}:{channels!r}"
        )
    return identity, is_default


def _pseudonymized_device(
    device: object,
    pseudonym_key: bytes,
) -> dict[str, str | bool] | None:
    fields = _device_fields(device)
    if fields is None:
        return None
    private_identity, is_default = fields
    digest = hmac.new(
        pseudonym_key,
        private_identity.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return {
        "pseudonym": f"device-{digest}",
        "is_default": is_default,
    }


def _sanitized_config(config: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, allowed in _CONFIG_ENUMS.items():
        result[key] = _safe_enum(_mapping_value(config, key), allowed)
    for key in _CONFIG_BOOLS:
        value = _mapping_value(config, key)
        if isinstance(value, bool):
            result[key] = value
    for key in _CONFIG_NUMBERS:
        value = _finite_number(_mapping_value(config, key))
        if value is not None:
            result[key] = value
    profiles = _mapping_value(config, "latency_calibration_profiles")
    result["saved_latency_profile_count"] = (
        len(profiles) if isinstance(profiles, Mapping) else 0
    )
    bindings = _mapping_value(config, "device_preset_bindings")
    result["device_preset_binding_count"] = (
        len(bindings) if isinstance(bindings, Mapping) else 0
    )
    return result


def _sanitized_processing(
    processing_settings: object,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for section, allowed_fields in _PROCESSING_FIELDS.items():
        raw_section = _mapping_value(processing_settings, section)
        if not isinstance(raw_section, Mapping):
            continue
        clean_section: dict[str, object] = {}
        for key in sorted(allowed_fields):
            value = raw_section.get(key)
            enum_values = _PROCESSING_ENUMS.get((section, key))
            if enum_values is not None:
                clean_section[key] = _safe_enum(value, enum_values)
            elif section == "eq" and key == "bands":
                if isinstance(value, list) and len(value) <= 32:
                    bands: list[dict[str, object]] = []
                    for raw_band in value:
                        if not isinstance(raw_band, Mapping):
                            continue
                        band: dict[str, object] = {}
                        for band_key in sorted(_EQ_BAND_FIELDS):
                            band_value = raw_band.get(band_key)
                            band_enum = _EQ_BAND_ENUMS.get(band_key)
                            if band_enum is not None:
                                band[band_key] = _safe_enum(
                                    band_value,
                                    band_enum,
                                )
                            elif isinstance(band_value, bool):
                                band[band_key] = band_value
                            elif band_value is None:
                                band[band_key] = None
                            else:
                                number = _finite_number(band_value)
                                if number is not None:
                                    band[band_key] = number
                        bands.append(band)
                    clean_section[key] = bands
            elif isinstance(value, bool):
                clean_section[key] = value
            elif isinstance(value, (list, tuple)):
                numbers = [_finite_number(item) for item in value[:32]]
                if all(item is not None for item in numbers):
                    clean_section[key] = numbers
            else:
                number = _finite_number(value)
                if number is not None:
                    clean_section[key] = number
        result[section] = clean_section
    bypass = _mapping_value(processing_settings, "bypass")
    if isinstance(bypass, bool):
        result["bypass"] = bypass
    return result


def _sanitized_runtime(
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(_RUNTIME_NUMBERS):
        if key not in diagnostics:
            continue
        result[key] = _finite_number(diagnostics[key])
    for key in sorted(_RUNTIME_BOOLS):
        value = diagnostics.get(key)
        if isinstance(value, bool):
            result[key] = value
    for key, allowed in sorted(_RUNTIME_ENUMS.items()):
        if key in diagnostics:
            result[key] = _safe_enum(diagnostics[key], allowed)
    result["backend_error_present"] = bool(
        diagnostics.get("noise_backend_error")
    )
    result["stream_error_present"] = bool(
        diagnostics.get("last_stream_error")
    )
    result["restart_reason_present"] = bool(
        diagnostics.get("last_restart_reason")
    )
    return result


def _system_snapshot() -> dict[str, str]:
    return {
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _safe_system_text(value: object) -> str:
    text = str(value)[:256]
    lowered = text.casefold()
    if (
        ":\\" in text
        or ":/" in text
        or "\\\\" in text
        or "/home/" in lowered
        or "/users/" in lowered
        or "\n" in text
        or "\r" in text
    ):
        return "redacted"
    return text


def _timestamp_utc(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("diagnostics timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_diagnostics_snapshot(
    *,
    app_version: str,
    runtime_diagnostics: Mapping[str, object],
    config: object,
    processing_settings: object,
    input_device: object,
    output_device: object,
    processing_sample_rate_hz: int,
    output_sample_rate_hz: int,
    running: bool,
    generated_at: datetime | None = None,
    pseudonym_key: bytes | None = None,
    system_info: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build an allowlisted support snapshot without local identifiers."""
    key = pseudonym_key or secrets.token_bytes(32)
    if len(key) < MIN_PSEUDONYM_KEY_BYTES:
        raise ValueError("diagnostics pseudonym key is too short")
    processing_rate = _finite_number(processing_sample_rate_hz)
    output_rate = _finite_number(output_sample_rate_hz)
    if processing_rate is None or output_rate is None:
        raise ValueError("diagnostics sample rates must be finite integers")
    safe_system = {
        key: _safe_system_text(value)
        for key, value in (system_info or _system_snapshot()).items()
        if key in _SYSTEM_FIELDS
    }
    snapshot: dict[str, Any] = {
        "schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
        },
        "generated_at_utc": _timestamp_utc(generated_at),
        "application": {
            "name": "AudioForge",
            "version": str(app_version),
        },
        "system": safe_system,
        "audio_engine": {
            "running": bool(running),
            "processing_sample_rate_hz": processing_rate,
            "output_sample_rate_hz": output_rate,
            "input_device": _pseudonymized_device(input_device, key),
            "output_device": _pseudonymized_device(output_device, key),
        },
        "configuration": _sanitized_config(config),
        "processing": _sanitized_processing(processing_settings),
        "runtime": _sanitized_runtime(runtime_diagnostics),
        "privacy": {
            "raw_audio_included": False,
            "environment_variables_included": False,
            "arbitrary_paths_included": False,
            "raw_device_names_included": False,
            "secrets_included": False,
            "pseudonyms_are_report_local": True,
        },
    }
    serialize_diagnostics_snapshot(snapshot)
    return snapshot


def serialize_diagnostics_snapshot(snapshot: Mapping[str, object]) -> bytes:
    if set(snapshot) != _ROOT_FIELDS:
        raise ValueError("diagnostics snapshot contains unexpected root fields")
    schema = snapshot.get("schema")
    if not isinstance(schema, Mapping):
        raise ValueError("diagnostics snapshot is missing its schema")
    if (
        schema.get("name") != SCHEMA_NAME
        or schema.get("version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported diagnostics snapshot schema")
    sections = {
        "application": frozenset({"name", "version"}),
        "system": _SYSTEM_FIELDS,
        "audio_engine": frozenset(
            {
                "running",
                "processing_sample_rate_hz",
                "output_sample_rate_hz",
                "input_device",
                "output_device",
            }
        ),
        "configuration": _CONFIG_FIELDS,
        "processing": frozenset({*_PROCESSING_FIELDS, "bypass"}),
        "runtime": _RUNTIME_FIELDS,
        "privacy": frozenset(
            {
                "raw_audio_included",
                "environment_variables_included",
                "arbitrary_paths_included",
                "raw_device_names_included",
                "secrets_included",
                "pseudonyms_are_report_local",
            }
        ),
    }
    for section_name, allowed_fields in sections.items():
        section = snapshot.get(section_name)
        if not isinstance(section, Mapping) or not set(section) <= allowed_fields:
            raise ValueError(
                f"diagnostics snapshot contains unexpected {section_name} fields"
            )
    processing = snapshot["processing"]
    assert isinstance(processing, Mapping)
    for section_name, allowed_fields in _PROCESSING_FIELDS.items():
        section = processing.get(section_name)
        if section is not None and (
            not isinstance(section, Mapping)
            or not set(section) <= allowed_fields
        ):
            raise ValueError(
                "diagnostics snapshot contains unexpected processing fields"
            )
    eq = processing.get("eq")
    if isinstance(eq, Mapping):
        bands = eq.get("bands")
        if bands is not None:
            if not isinstance(bands, list) or any(
                not isinstance(band, Mapping)
                or not set(band) <= _EQ_BAND_FIELDS
                for band in bands
            ):
                raise ValueError(
                    "diagnostics snapshot contains unexpected EQ band fields"
                )
    try:
        encoded = (
            json.dumps(
                snapshot,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("diagnostics snapshot is not finite JSON") from error
    if len(encoded) > MAX_SERIALIZED_BYTES:
        raise ValueError("diagnostics snapshot exceeds the size limit")
    return encoded


def diagnostics_filename(app_version: str, generated_at: datetime | None = None) -> str:
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("diagnostics filename timestamp must be timezone-aware")
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_version = "".join(
        character
        for character in str(app_version)
        if character.isascii() and (character.isalnum() or character in ".-_")
    )
    return f"AudioForge-diagnostics-v{safe_version or 'unknown'}-{stamp}.json"


def write_diagnostics_snapshot(
    path: str | Path,
    snapshot: Mapping[str, object],
) -> None:
    """Atomically write one validated snapshot."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_diagnostics_snapshot(snapshot)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "MAX_SERIALIZED_BYTES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "build_diagnostics_snapshot",
    "diagnostics_filename",
    "serialize_diagnostics_snapshot",
    "write_diagnostics_snapshot",
]
