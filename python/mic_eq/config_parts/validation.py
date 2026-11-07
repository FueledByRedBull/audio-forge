"""Validation helpers for preset and app config persistence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from .. import noise_model_ids as _noise_model_ids
from .. import processing_ranges as _processing_ranges
from .shared import PresetValidationError


@cache
def validation_ranges() -> Mapping[str, Mapping[str, tuple[float, float]]]:
    """The native core's numeric ranges by section and key (booleans have none).

    Read on first use, so this module imports without the core and the
    missing-core ImportError names the call that needed it. The table is
    shared and read-only: a caller cannot change what presets validate against.
    """
    return MappingProxyType(
        {
            section: MappingProxyType(ranges)
            for section, ranges in _processing_ranges().items()
        }
    )


@cache
def noise_model_ids() -> list[str]:
    """The noise model ids a saved preset may name."""
    return _noise_model_ids()


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


# Qt rectangles hold C ints; stay far below that so a hand-edited value cannot
# stop the main window from opening.
_MAX_WINDOW_COORDINATE = 100_000


def _coerce_window_geometry(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None

    coerced: dict[str, int] = {}
    for key in ("x", "y", "width", "height"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        try:
            numeric = float(raw)
        except OverflowError:
            return None
        if not math.isfinite(numeric) or abs(numeric) > _MAX_WINDOW_COORDINATE:
            return None
        coerced[key] = int(round(numeric))

    coerced["width"] = max(640, coerced["width"])
    coerced["height"] = max(480, coerced["height"])
    if value.get("maximized") is True:
        coerced["maximized"] = True
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


def _validate_range(value: object, *args: object, strict: bool = False) -> float:
    min_val, max_val, param_name, section = _parse_range_args(args)
    if isinstance(value, bool):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        )
    accepted_types = (int, float) if strict else (int, float, str)
    if not isinstance(value, accepted_types):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        ) from exc
    if not math.isfinite(numeric_value):
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value!r} "
            f"(must be a finite number between {min_val} and {max_val})"
        )

    if strict:
        if numeric_value < min_val or numeric_value > max_val:
            raise PresetValidationError(
                f"Invalid {param_name} in {section}: {numeric_value} "
                f"(must be between {min_val} and {max_val})"
            )
        return numeric_value

    tolerance = (max_val - min_val) * 0.1
    if numeric_value < min_val - tolerance or numeric_value > max_val + tolerance:
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {numeric_value} "
            f"(must be between {min_val} and {max_val})"
        )
    return max(min_val, min(max_val, numeric_value))


__all__ = [
    "_coerce_config_bool",
    "_coerce_window_geometry",
    "_validate_bool",
    "_validate_range",
    "noise_model_ids",
    "validation_ranges",
]
