# ruff: noqa: E402

"""
AudioForge - Low-latency microphone audio processor

Provides real-time noise suppression and equalization for voice communication.
"""

__version__ = "1.8.6"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mic_eq_core import (
        AudioProcessor,
        DeviceInfo,
        configure_deepfilter_runtime_paths,
        list_input_devices,
        list_output_devices,
        simulate_auto_eq_chain,
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

        def configure_deepfilter_runtime_paths(*args, **kwargs):
            _raise_core_import_error()

__all__ = [
    "AudioProcessor",
    "DeviceInfo",
    "list_input_devices",
    "list_output_devices",
    "simulate_auto_eq_chain",
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
