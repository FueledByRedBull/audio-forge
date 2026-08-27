"""AudioForge low-latency microphone processing."""

__version__ = "1.11.3"

try:
    from .mic_eq_core import (
        AudioProcessor,
        DeviceInfo,
        analyze_vad_probabilities,
        configure_deepfilter_runtime_paths,
        eq_magnitude_response,
        eq_magnitude_response_v2,
        list_input_devices,
        list_output_devices,
    )

    CORE_AVAILABLE = True
except ImportError as error:
    _CORE_IMPORT_ERROR = error
    CORE_AVAILABLE = False

    def _missing_core(*args, **kwargs):
        raise ImportError(
            "mic_eq_core is unavailable. Build it with: maturin develop --release"
        ) from _CORE_IMPORT_ERROR

    class _MissingCoreType:
        def __init__(self, *args, **kwargs):
            _missing_core()

    AudioProcessor = type("AudioProcessor", (_MissingCoreType,), {"__module__": __name__})
    DeviceInfo = type("DeviceInfo", (_MissingCoreType,), {"__module__": __name__})
    list_input_devices = _missing_core
    list_output_devices = _missing_core
    configure_deepfilter_runtime_paths = _missing_core
    analyze_vad_probabilities = _missing_core
    eq_magnitude_response = _missing_core
    eq_magnitude_response_v2 = _missing_core

from .config import (
    BUILTIN_PRESETS,
    Preset,
    list_presets,
    load_preset,
    save_preset,
)

__all__ = [
    "AudioProcessor",
    "DeviceInfo",
    "list_input_devices",
    "list_output_devices",
    "eq_magnitude_response",
    "eq_magnitude_response_v2",
    "analyze_vad_probabilities",
    "configure_deepfilter_runtime_paths",
    "CORE_AVAILABLE",
    "Preset",
    "save_preset",
    "load_preset",
    "list_presets",
    "BUILTIN_PRESETS",
]
