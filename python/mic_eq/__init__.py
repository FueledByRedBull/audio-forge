# ruff: noqa: E402

"""
AudioForge - Low-latency microphone audio processor

Provides real-time noise suppression and equalization for voice communication.
"""

__version__ = "1.11.2"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mic_eq_core import (
        AudioProcessor,
        DeviceInfo,
        analyze_vad_probabilities,
        configure_deepfilter_runtime_paths,
        eq_magnitude_response,
        eq_magnitude_response_v2,
        list_input_devices,
        list_output_devices,
        measure_integrated_loudness,
        product_resampler_configuration,
        simulate_auto_eq_chain,
        simulate_auto_makeup_control,
        simulate_gate_suppressor_order,
        simulate_eq_v2,
        simulate_product_resampler,
    )

    CORE_AVAILABLE: bool
else:
    # Keep package import resilient for tooling/tests that only need the
    # pure-Python modules and do not require the native extension.
    _CORE_IMPORT_ERROR = None
    _core_module = None
    try:
        from . import mic_eq_core as _core_module
    except ImportError:
        try:
            # Compatibility for environments that expose the extension outside
            # the package namespace.
            import mic_eq_core as _core_module
        except ImportError as error:
            _CORE_IMPORT_ERROR = error

    def _missing_core(*args, **kwargs):
        raise ImportError(
            "mic_eq_core is unavailable or missing this API. Make sure to "
            "build with: maturin develop --release"
        ) from _CORE_IMPORT_ERROR

    if _core_module is not None:
        AudioProcessor = _core_module.AudioProcessor
        DeviceInfo = _core_module.DeviceInfo
        list_input_devices = _core_module.list_input_devices
        list_output_devices = _core_module.list_output_devices
        CORE_AVAILABLE = True
        for _name in (
            "configure_deepfilter_runtime_paths",
            "analyze_vad_probabilities",
            "simulate_auto_eq_chain",
            "simulate_auto_makeup_control",
            "simulate_gate_suppressor_order",
            "simulate_eq_v2",
            "simulate_product_resampler",
            "product_resampler_configuration",
            "eq_magnitude_response",
            "eq_magnitude_response_v2",
            "measure_integrated_loudness",
        ):
            globals()[_name] = getattr(_core_module, _name, _missing_core)
    else:
        CORE_AVAILABLE = False

        class _MissingCoreType:
            def __init__(self, *args, **kwargs):
                _missing_core()

        AudioProcessor = type("AudioProcessor", (_MissingCoreType,), {"__module__": __name__})
        DeviceInfo = type("DeviceInfo", (_MissingCoreType,), {"__module__": __name__})
        for _name in (
            "list_input_devices",
            "list_output_devices",
            "simulate_auto_eq_chain",
            "simulate_auto_makeup_control",
            "simulate_gate_suppressor_order",
            "simulate_eq_v2",
            "simulate_product_resampler",
            "product_resampler_configuration",
            "configure_deepfilter_runtime_paths",
            "analyze_vad_probabilities",
            "eq_magnitude_response",
            "eq_magnitude_response_v2",
            "measure_integrated_loudness",
        ):
            globals()[_name] = _missing_core


__all__ = [
    "AudioProcessor",
    "DeviceInfo",
    "list_input_devices",
    "list_output_devices",
    "simulate_auto_eq_chain",
    "simulate_auto_makeup_control",
    "simulate_gate_suppressor_order",
    "simulate_eq_v2",
    "simulate_product_resampler",
    "product_resampler_configuration",
    "eq_magnitude_response",
    "eq_magnitude_response_v2",
    "measure_integrated_loudness",
    "analyze_vad_probabilities",
    "configure_deepfilter_runtime_paths",
    "CORE_AVAILABLE",
    "Preset",
    "save_preset",
    "load_preset",
    "list_presets",
    "BUILTIN_PRESETS",
]

# Also export config utilities
from .config import (
    Preset,
    save_preset,
    load_preset,
    list_presets,
    BUILTIN_PRESETS,
)
