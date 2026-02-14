"""
AudioForge - Low-latency microphone audio processor

Provides real-time noise suppression and equalization for voice communication.
"""

__version__ = "1.5.0"

# Import the Rust core module.
#
# Keep package import resilient for tooling/tests that only need the pure-Python
# modules (config/analysis) and do not require the native extension.
_CORE_IMPORT_ERROR = None
try:
    from .mic_eq_core import (
        AudioProcessor,
        DeviceInfo,
        list_input_devices,
        list_output_devices,
    )
    CORE_AVAILABLE = True
except ImportError:
    try:
        # Fallback for environments that install the extension as top-level module.
        from mic_eq_core import (
            AudioProcessor,
            DeviceInfo,
            list_input_devices,
            list_output_devices,
        )
        CORE_AVAILABLE = True
    except ImportError as e:
        _CORE_IMPORT_ERROR = e
        CORE_AVAILABLE = False

        def _raise_core_import_error():
            raise ImportError(
                "Failed to import mic_eq_core. Make sure to build with: "
                "maturin develop --release"
            ) from _CORE_IMPORT_ERROR

        class AudioProcessor:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                _raise_core_import_error()

        class DeviceInfo:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                _raise_core_import_error()

        def list_input_devices():
            _raise_core_import_error()

        def list_output_devices():
            _raise_core_import_error()

__all__ = [
    "AudioProcessor",
    "DeviceInfo",
    "list_input_devices",
    "list_output_devices",
    "CORE_AVAILABLE",
]

# Also export config utilities
from .config import (
    Preset,
    save_preset,
    load_preset,
    list_presets,
    BUILTIN_PRESETS,
)
