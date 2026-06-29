"""Public config facade built from focused config_parts modules."""

from __future__ import annotations

from .config_parts.app_config import AppConfig, load_config, save_config
from .config_parts.catalogs import build_builtin_presets, build_target_curves
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
    LEGACY_APPDATA_DIR_NAME,
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


BUILTIN_PRESETS = build_builtin_presets(
    Preset,
    GateSettings,
    EQSettings,
    RNNoiseSettings,
)

TARGET_CURVES = build_target_curves(TargetCurve)


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
    "LEGACY_APPDATA_DIR_NAME",
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
