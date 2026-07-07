"""Validation helpers for preset and app config persistence."""

from __future__ import annotations

import math

from .shared import PresetValidationError


VALIDATION_RANGES = {
    "gate": {
        "threshold_db": (-80.0, -10.0),
        "attack_ms": (0.1, 100.0),
        "release_ms": (10.0, 1000.0),
        "gate_mode": (0, 2),
        "vad_threshold": (0.3, 0.7),
        "vad_hold_time_ms": (0.0, 500.0),
        "vad_pre_gain": (1.0, 10.0),
        "auto_threshold_enabled": (bool, None),
        "gate_margin_db": (0.0, 20.0),
    },
    "eq": {
        "band_freq": (20.0, 20000.0),
        "band_gain": (-12.0, 12.0),
        "band_q": (0.1, 10.0),
    },
    "rnnoise": {
        "strength": (0.0, 1.0),
        "model": ["rnnoise", "deepfilter-ll", "deepfilter"],
    },
    "deesser": {
        "auto_amount": (0.0, 1.0),
        "low_cut_hz": (2000.0, 12000.0),
        "high_cut_hz": (2200.0, 16000.0),
        "threshold_db": (-60.0, -6.0),
        "ratio": (1.0, 20.0),
        "attack_ms": (0.1, 50.0),
        "release_ms": (5.0, 500.0),
        "max_reduction_db": (0.0, 24.0),
    },
    "compressor": {
        "threshold_db": (-60.0, 0.0),
        "ratio": (1.0, 20.0),
        "attack_ms": (0.1, 100.0),
        "release_ms": (10.0, 1000.0),
        "makeup_gain_db": (0.0, 24.0),
        "adaptive_release": (bool, None),
        "auto_makeup_enabled": (bool, None),
        "target_lufs": (-24.0, -12.0),
        "sidechain_highpass_enabled": (bool, None),
    },
    "limiter": {
        "ceiling_db": (-12.0, 0.0),
        "release_ms": (10.0, 500.0),
    },
}


def _validate_bool(value: object, param_name: str, section: str) -> bool:
    if isinstance(value, bool):
        return value
    raise PresetValidationError(
        f"Invalid {param_name} in {section}: {value!r} (must be true or false)"
    )


def _coerce_config_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _coerce_window_geometry(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None

    coerced: dict[str, int] = {}
    for key in ("x", "y", "width", "height"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        numeric = float(raw)
        if not math.isfinite(numeric):
            return None
        coerced[key] = int(round(numeric))

    coerced["width"] = max(640, coerced["width"])
    coerced["height"] = max(480, coerced["height"])
    return coerced


def _parse_range_args(args: tuple[object, ...]) -> tuple[float, float, str, str]:
    if len(args) != 4:
        raise PresetValidationError("Invalid validation range definition")
    min_val, max_val, param_name, section = args
    if not isinstance(param_name, str) or not isinstance(section, str):
        raise PresetValidationError("Invalid validation range definition")
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise PresetValidationError("Invalid validation range definition")
    return float(min_val), float(max_val), param_name, section


def _validate_range(value: object, *args: object) -> float:
    min_val, max_val, param_name, section = _parse_range_args(args)
    if isinstance(value, bool):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        )
    if not isinstance(value, (int, float, str)):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        ) from exc
    if not math.isfinite(numeric_value):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        )

    tolerance = (max_val - min_val) * 0.1
    if numeric_value < min_val - tolerance or numeric_value > max_val + tolerance:
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {numeric_value} "
            f"(must be between {min_val} and {max_val})"
        )
    return max(min_val, min(max_val, numeric_value))


def _validate_fixed_float_list(
    values: object,
    expected_len: int,
    *args: object,
) -> list[float]:
    min_val, max_val, param_name, section = _parse_range_args(args)
    if not isinstance(values, (list, tuple)):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: expected list of {expected_len} values"
        )
    if len(values) != expected_len:
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: expected {expected_len} values, got {len(values)}"
        )
    return [
        _validate_range(value, min_val, max_val, f"{param_name}[{index}]", section)
        for index, value in enumerate(values)
    ]


__all__ = [
    "VALIDATION_RANGES",
    "_coerce_config_bool",
    "_coerce_window_geometry",
    "_validate_bool",
    "_validate_fixed_float_list",
    "_validate_range",
]
