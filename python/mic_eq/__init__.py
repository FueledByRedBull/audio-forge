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

    def _raise_core_import_error():
        raise ImportError(
            "Failed to import mic_eq_core. Make sure to build with: "
            "maturin develop --release"
        ) from _CORE_IMPORT_ERROR

    if _core_module is not None:
        AudioProcessor = _core_module.AudioProcessor
        DeviceInfo = _core_module.DeviceInfo
        list_input_devices = _core_module.list_input_devices
        list_output_devices = _core_module.list_output_devices
        configure_deepfilter_runtime_paths = getattr(
            _core_module,
            "configure_deepfilter_runtime_paths",
            lambda *_args, **_kwargs: None,
        )
        analyze_vad_probabilities = getattr(
            _core_module,
            "analyze_vad_probabilities",
            None,
        )
        CORE_AVAILABLE = True

        def _raise_missing_simulation_helper(*args, **kwargs):
            raise ImportError(
                "mic_eq_core was imported, but simulate_auto_eq_chain is missing. "
                "Rebuild with: maturin develop --release"
            )

        simulate_auto_eq_chain = getattr(
            _core_module,
            "simulate_auto_eq_chain",
            _raise_missing_simulation_helper,
        )
        simulate_auto_makeup_control = getattr(
            _core_module,
            "simulate_auto_makeup_control",
            _raise_missing_simulation_helper,
        )
        simulate_gate_suppressor_order = getattr(
            _core_module,
            "simulate_gate_suppressor_order",
            _raise_missing_simulation_helper,
        )
        simulate_product_resampler = getattr(
            _core_module,
            "simulate_product_resampler",
            _raise_missing_simulation_helper,
        )
        product_resampler_configuration = getattr(
            _core_module,
            "product_resampler_configuration",
            _raise_missing_simulation_helper,
        )

        def _raise_missing_eq_response(*args, **kwargs):
            raise ImportError(
                "mic_eq_core was imported, but eq_magnitude_response is missing. "
                "Rebuild with: maturin develop --release"
            )

        eq_magnitude_response = getattr(
            _core_module,
            "eq_magnitude_response",
            _raise_missing_eq_response,
        )
        eq_magnitude_response_v2 = getattr(
            _core_module,
            "eq_magnitude_response_v2",
            _raise_missing_eq_response,
        )
        simulate_eq_v2 = getattr(
            _core_module,
            "simulate_eq_v2",
            _raise_missing_simulation_helper,
        )
        measure_integrated_loudness = getattr(
            _core_module,
            "measure_integrated_loudness",
            _raise_missing_simulation_helper,
        )
    else:
        CORE_AVAILABLE = False

        class AudioProcessor:
            def __init__(self, *args, **kwargs):
                _raise_core_import_error()

        class DeviceInfo:
            def __init__(self, *args, **kwargs):
                _raise_core_import_error()

        def list_input_devices():
            _raise_core_import_error()

        def list_output_devices():
            _raise_core_import_error()

        def simulate_auto_eq_chain(*args, **kwargs):
            _raise_core_import_error()

        def simulate_auto_makeup_control(*args, **kwargs):
            _raise_core_import_error()

        def simulate_gate_suppressor_order(*args, **kwargs):
            _raise_core_import_error()

        def simulate_eq_v2(*args, **kwargs):
            _raise_core_import_error()

        def simulate_product_resampler(*args, **kwargs):
            _raise_core_import_error()

        def product_resampler_configuration(*args, **kwargs):
            _raise_core_import_error()

        def configure_deepfilter_runtime_paths(*args, **kwargs):
            _raise_core_import_error()

        def analyze_vad_probabilities(*args, **kwargs):
            _raise_core_import_error()

        def eq_magnitude_response(*args, **kwargs):
            _raise_core_import_error()

        def eq_magnitude_response_v2(*args, **kwargs):
            _raise_core_import_error()

        def measure_integrated_loudness(*args, **kwargs):
            _raise_core_import_error()


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
