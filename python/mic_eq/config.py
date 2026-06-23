"""Public config facade built from focused config_parts modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .config_parts.app_config import AppConfig, load_config, save_config
from .config_parts.presets import (
    Preset,
    generate_auto_eq_preset_name,
    list_presets,
    load_preset,
    save_preset,
)
from .config_parts.settings import (
    ANALYSIS_MAX_SPECTRAL_FLATNESS,
    ANALYSIS_MIN_DYNAMIC_RANGE,
    ANALYSIS_MIN_PEAK_COUNT,
    ANALYSIS_MIN_SNR,
    AUTO_EQ_DEFAULT_Q,
    CompressorSettings,
    DeEsserSettings,
    EQSettings,
    EQ_FREQUENCIES,
    GateSettings,
    LatencyCalibrationProfile,
    LimiterSettings,
    RNNoiseSettings,
    TargetCurve,
)
from .config_parts.shared import (
    APPDATA_DIR_NAME,
    CURRENT_VERSION,
    DeviceIdentity,
    PresetValidationError,
    build_latency_profile_key,
    coerce_device_identity,
    get_config_file,
    get_preset_imports_dir,
    get_presets_dir,
    legacy_latency_profile_key,
    parse_latency_profile_key,
)
from .config_parts.validation import (
    VALIDATION_RANGES,
    _coerce_config_bool,
    _coerce_window_geometry,
    _validate_bool,
    _validate_fixed_float_list,
    _validate_range,
)


def _load_catalog_builders():
    """Load catalog builders in packages, with a direct-test file fallback."""
    if "mic_eq" in sys.modules:
        from .config_parts.catalogs import build_builtin_presets, build_target_curves

        return build_builtin_presets, build_target_curves

    module_path = Path(__file__).with_name("config_parts") / "catalogs.py"
    spec = importlib.util.spec_from_file_location("mic_eq_config_catalogs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load config catalog builders")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_builtin_presets, module.build_target_curves


_build_builtin_presets, _build_target_curves = _load_catalog_builders()

BUILTIN_PRESETS = _build_builtin_presets(
    Preset,
    GateSettings,
    EQSettings,
    RNNoiseSettings,
)

TARGET_CURVES = _build_target_curves(TargetCurve)


__all__ = [
    "ANALYSIS_MAX_SPECTRAL_FLATNESS",
    "ANALYSIS_MIN_DYNAMIC_RANGE",
    "ANALYSIS_MIN_PEAK_COUNT",
    "ANALYSIS_MIN_SNR",
    "APPDATA_DIR_NAME",
    "AUTO_EQ_DEFAULT_Q",
    "AppConfig",
    "BUILTIN_PRESETS",
    "CURRENT_VERSION",
    "CompressorSettings",
    "DeEsserSettings",
    "DeviceIdentity",
    "EQSettings",
    "EQ_FREQUENCIES",
    "GateSettings",
    "LatencyCalibrationProfile",
    "LimiterSettings",
    "Preset",
    "PresetValidationError",
    "RNNoiseSettings",
    "TARGET_CURVES",
    "TargetCurve",
    "VALIDATION_RANGES",
    "_coerce_config_bool",
    "_coerce_window_geometry",
    "_validate_bool",
    "_validate_fixed_float_list",
    "_validate_range",
    "build_latency_profile_key",
    "coerce_device_identity",
    "generate_auto_eq_preset_name",
    "get_config_file",
    "get_preset_imports_dir",
    "get_presets_dir",
    "legacy_latency_profile_key",
    "list_presets",
    "load_config",
    "load_preset",
    "parse_latency_profile_key",
    "save_config",
    "save_preset",
]
