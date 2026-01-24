"""
MicEq - Low-latency microphone audio processor

Provides real-time noise suppression and equalization for voice communication.
"""

__version__ = "0.1.0"

# Import the Rust core module
try:
    from mic_eq_core import (
        AudioProcessor,
        DeviceInfo,
        list_input_devices,
        list_output_devices,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import mic_eq_core. Make sure to build with: "
        "maturin develop --release"
    ) from e

__all__ = [
    "AudioProcessor",
    "DeviceInfo",
    "list_input_devices",
    "list_output_devices",
]

# Also export config utilities
from .config import (
    Preset,
    save_preset,
    load_preset,
    list_presets,
    BUILTIN_PRESETS,
)
